import math

import torch

from src.modules.metrics import acc_metric
from src.modules.regularizers import FisherPenaly, BatchGradCovariancePenalty
from src.utils import common


class ClassificationLoss(torch.nn.Module):
    def __init__(self, criterion_name):
        super().__init__()
        self.criterion = common.LOSS_NAME_MAP[criterion_name]()

    def forward(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        acc = acc_metric(y_pred, y_true)
        evaluators = {
            'loss': loss.item(),
            'acc': acc
        }
        return loss, evaluators


class FisherPenaltyLoss(torch.nn.Module):
    def __init__(self, model, general_criterion_name, num_classes, whether_record_trace=False, fpw=0.0):
        super().__init__()
        self.criterion = ClassificationLoss(common.LOSS_NAME_MAP[general_criterion_name]())
        self.regularizer = FisherPenaly(model, common.LOSS_NAME_MAP[general_criterion_name](), num_classes)
        self.whether_record_trace = whether_record_trace
        self.fpw = fpw
        #przygotowanie do logowania co n kroków
        self.overall_trace_buffer = None
        self.traces = None

    def forward(self, y_pred, y_true):
        traces = {}
        loss, evaluators = self.criterion(y_pred, y_true)
        if self.whether_record_trace:# and self.regularizer.model.training:
            overall_trace, traces = self.regularizer(y_pred)
            evaluators['overall_trace'] = overall_trace.item()
            if self.fpw > 0:
                loss += self.fpw * overall_trace
        return loss, evaluators, traces
    

class BatchGradCovarianceLoss(torch.nn.Module):
    def __init__(self, model, loader, general_criterion_name, bgcw=0.0, n=10):
        super().__init__()
        self.criterion = ClassificationLoss(common.LOSS_NAME_MAP[general_criterion_name]())
        self.regularizer = BatchGradCovariancePenalty(model, loader, common.LOSS_NAME_MAP[general_criterion_name])
        self.bgcw = bgcw
        self.n = n

    def forward(self, y_pred, y_true):
        loss, evaluators = self.criterion(y_pred, y_true)
        if self.whether_record_logdet:
            log_det = self.regularizer(self.n) # ten regulizer nie ma sensu
            evaluators['bgc_log_det'] = log_det.item()
            if self.bgcw > 0:
                loss -= self.bgcw * log_det
        return loss, evaluators
    

class MSESoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = torch.nn.MSELoss()
   

    def forward(self, y_pred, y_true):
        y_true = torch.nn.functional.one_hot(y_true, num_classes=10).float()
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        loss = self.criterion(y_pred, y_true)
        return loss

from torch.func import functional_call, vmap, grad
from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
class BADGELoss(torch.nn.Module):
    def __init__(self, model, general_criterion_name, whether_record_logdet=False, badge_coeff=0.0, normalize=False):
        super().__init__()
        self.model = model
        self.criterion = ClassificationLoss(common.LOSS_NAME_MAP[general_criterion_name]()).criterion
        # self.ft_criterion = vmap(grad(self.compute_loss, has_aux=True), in_dims=(None, None, 0, 0))
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.whether_record_logdet = whether_record_logdet
        self.normalize = normalize
        self.badge_coeff = badge_coeff

    def forward(self, x_true, y_true):
        traces = {}
        if self.whether_record_logdet:
            loss, y_pred, graham_det, normalized_graham_det = self.regularizer(x_true, y_true)
            # print(y_pred.requires_grad, loss.requires_grad, log_det.requires_grad)
            acc = acc_metric(y_pred, y_true)
            evaluators = {
                'loss': loss.item(),
                'acc': acc
            }
            evaluators['det/graham_det'] = graham_det.item()
            evaluators['det/normalized_graham_det'] = normalized_graham_det.item()
            # if self.badge_coeff > 0:
            #     loss -= self.badge_coeff * log_det
        else:
            1/0
            loss, evaluators = self.criterion(x_true, y_true)
        # print(evaluators)
        return loss, evaluators, traces
    
    def compute_grad(self, prediction, target):
        prediction = prediction.unsqueeze(0)  # prepend batch dimension for processing
        target = target.unsqueeze(0)

        loss = self.criterion(prediction, target)
        
        params = [v for k, v in self.model.named_parameters() if k in self.allowed_parameter_names]
        
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True)
        
        grads = torch.cat([g.flatten() for g in grads])
        grads = [grad for grad in grads if grad is not None]

        return grads, loss
    
    def regularizer(self, x_true, y_true):
        losses = []
        sample_grads = []
        batch_size = x_true.shape[0]
        predictions = self.model(x_true)
        for i in range(batch_size):
            grad, loss = self.compute_grad(predictions[i], y_true[i])
            losses.append(loss)
            sample_grads.append(grad)
            
        loss = torch.stack(losses).mean()
        concatenated_weights = torch.stack(sample_grads)
        graham_matrix = concatenated_weights @ concatenated_weights.T
        graham_det = torch.det(graham_matrix)
        # if self.normalize:
        concatenated_weights = concatenated_weights / (1e-9 + concatenated_weights.norm(dim=1, keepdim=True).detach())
        normalized_graham_matrix = concatenated_weights @ concatenated_weights.T
        normalized_graham_det = torch.det(normalized_graham_matrix)
        
        # log_det = torch.log(det + 1e-9) 
        return loss, predictions, graham_det, normalized_graham_det
    
    # def compute_loss(self, params, buffers, sample, target):
    #     batch = sample.unsqueeze(0)
    #     targets = target.unsqueeze(0)
    #     predictions = functional_call(self.model, (params, buffers), (batch,))
    #     loss = self.criterion(predictions, targets)
    #     return loss, (loss, predictions)
    
    # def regularizer(self, x_true, y_true):
    #     params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.allowed_parameter_names}
    #     buffers = {}
    #     ft_per_sample_grads, (loss, predictions) = self.ft_criterion(params, buffers, x_true, y_true)
    #     print(loss.shape, predictions.shape)
    #     concatenated_weights = torch.empty((x_true.shape[0], 0), device=x_true.device)
    #     loss = loss.mean()
    #     predictions = predictions.squeeze()
        
    #     for tag in ft_per_sample_grads:
    #         grad = ft_per_sample_grads[tag]
    #         grad = grad.reshape(grad.shape[0], -1)
    #         concatenated_weights = torch.cat((concatenated_weights, grad), dim=1)
    #     if self.normalize:
    #         concatenated_weights = concatenated_weights / (1e-9 + concatenated_weights.norm(dim=1, keepdim=True))
    #     graham_matrix = concatenated_weights @ concatenated_weights.T
    #     det = torch.det(graham_matrix)
    #     log_det = torch.log(det)
    #     return loss, predictions, log_det