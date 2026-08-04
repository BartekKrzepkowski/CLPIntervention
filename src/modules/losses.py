
import torch

from src.modules.metrics import acc_metric
from src.modules.regularizers import BalancePenaly, FisherPenaly


CRITERION_NAME_MAP = {
    "ce": torch.nn.CrossEntropyLoss,
    "nll": torch.nn.NLLLoss,
}


def build_classification_criterion(criterion_name, weight=None):
    try:
        criterion = CRITERION_NAME_MAP[criterion_name]
    except KeyError as error:
        raise ValueError(f"Unsupported classification criterion: {criterion_name}") from error
    return criterion(weight=weight)


class ClassificationLoss(torch.nn.Module):
    def __init__(self, criterion_name, weight=None):
        super().__init__()
        self.criterion = build_classification_criterion(criterion_name, weight=weight)

    def forward(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        acc = acc_metric(y_pred, y_true)
        evaluators = {
            'loss': loss.item(),
            'acc': acc
        }
        return loss, evaluators


class MSESoftmaxLoss(torch.nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()

        self.criterion = torch.nn.MSELoss()
        self.num_classes = num_classes
   

    def forward(self, y_pred, y_true):
        num_classes = self.num_classes or y_pred.size(1)
        y_true = torch.nn.functional.one_hot(y_true, num_classes=num_classes).float()
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        loss = self.criterion(y_pred, y_true)
        return loss, {"loss": loss.item(), "acc": acc_metric(y_pred, y_true.argmax(dim=1))}
    

class FisherPenaltyLoss(torch.nn.Module):
    def __init__(self, model, general_criterion_name, num_classes, whether_record_trace=False, fpw=0.0):
        super().__init__()
        self.criterion = ClassificationLoss(general_criterion_name)
        self.regularizer = FisherPenaly(model, build_classification_criterion(general_criterion_name), num_classes)
        self.whether_record_trace = whether_record_trace
        self.fpw = fpw
        #przygotowanie do logowania co n kroków
        self.overall_trace_buffer = None
        self.traces = None

    def forward(self, y_pred, y_true):
        traces = {}
        loss, evaluators = self.criterion(y_pred, y_true)
        if self.training and torch.is_grad_enabled() and self.whether_record_trace:
            overall_trace, traces = self.regularizer(y_pred)
            evaluators['overall_trace'] = overall_trace.item()
            if self.fpw > 0:
                loss += self.fpw * overall_trace
        evaluators.update({f"fisher_penalty/{name}": value for name, value in traces.items()})
        return loss, evaluators
    
    
class BalancePenaltyLoss(torch.nn.Module):
    def __init__(self, model, general_criterion_name, num_classes, weight=None, fpw=0.0, use_log=False):
        super().__init__()
        self.criterion = ClassificationLoss(general_criterion_name, weight=weight)
        self.regularizer = BalancePenaly(model, build_classification_criterion(general_criterion_name, weight=weight), num_classes)
        self.fpw = fpw
        self.use_log = use_log
        #przygotowanie do logowania co n kroków
        self.overall_trace_buffer = None
        self.traces = None

    def forward(self, y_pred, y_true):
        loss, evaluators = self.criterion(y_pred, y_true)
        if not self.training or not torch.is_grad_enabled():
            return loss, evaluators
        mean_ratio, sum_of_fims, traces = self.regularizer(y_pred)
        evaluators['balance_penalty/mean_ratio'] = mean_ratio.item()
        evaluators = evaluators | traces
        if self.fpw > 0:
            reg_part = torch.log(mean_ratio) if self.use_log else (mean_ratio - 1)
            loss += self.fpw * (reg_part - sum_of_fims)
            evaluators['balance_penalty/combined_loss'] = loss.item()
        return loss, evaluators
    
