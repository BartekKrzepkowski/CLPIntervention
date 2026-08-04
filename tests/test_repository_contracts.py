import pathlib
import subprocess
import sys

import pytest


def test_resume_rng_is_restored_after_logger_initialization(monkeypatch):
    import src.trainer.trainer_classification_mm_clp as trainer_module

    trainer = object.__new__(trainer_module.TrainerClassification)
    events = []
    trainer.resume_rng_state = {"sentinel": True}
    trainer.manual_seed = lambda _config: events.append("manual_seed")
    trainer.at_exp_start = lambda _config: events.append("at_exp_start")
    monkeypatch.setattr(
        trainer_module,
        "restore_rng_state",
        lambda state: events.append(("restore_rng_state", state)),
    )

    trainer._initialize_run(object())

    assert events == [
        "manual_seed",
        "at_exp_start",
        ("restore_rng_state", {"sentinel": True}),
    ]
    assert trainer.resume_rng_state is None


def test_core_modules_can_be_imported_independently():
    command = (
        "import src.modules.losses; "
        "import src.modules.architectures.mm_mlp; "
        "import src.modules.architectures.models; "
        "import src.utils.common"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_frozen_left_active_resolves_to_two_active_modalities():
    from omegaconf import OmegaConf

    from src.trainer.trainer_validation_clp import ValidationControlledTrainer

    config = OmegaConf.create(
        {"phase3_intervention": {"mode": "frozen_left_active"}}
    )
    stage = ValidationControlledTrainer._phase_stage(3, config)
    assert stage.enable_left_branch is True
    assert stage.enable_right_branch is True
    assert stage.left_branch_intervention is None
    assert stage.right_branch_intervention is None

    historical = ValidationControlledTrainer._phase_stage(
        3, OmegaConf.create({})
    )
    assert historical.enable_left_branch is False
    assert historical.left_branch_intervention == "deactivation"


def test_full_active_umt_resolves_to_two_active_modalities():
    from omegaconf import OmegaConf

    from src.trainer.trainer_validation_clp import ValidationControlledTrainer

    stage = ValidationControlledTrainer._phase_stage(
        3,
        OmegaConf.create({"phase3_intervention": {"mode": "full_active"}}),
    )
    assert stage.enable_left_branch is True
    assert stage.enable_right_branch is True
    assert stage.left_branch_intervention is None
    assert stage.right_branch_intervention is None


def test_active_launchers_do_not_contain_host_specific_checkpoint_paths_or_dynamic_modules():
    root = pathlib.Path(__file__).parents[1]
    launchers = list(root.glob("*.sh")) + list((root / "scripts/bash").glob("*.sh"))
    content = "\n".join(path.read_text(encoding="utf-8") for path in launchers)
    assert "/net/pr2" not in content
    assert "scripts.python_new.$1" not in content
    assert "PATH_TO_CHECKPOINT" not in content


def test_environment_file_does_not_contain_a_literal_wandb_key():
    root = pathlib.Path(__file__).parents[1]
    content = (root / "src/configs/env_variables.sh").read_text(encoding="utf-8")
    api_key_line = next(line for line in content.splitlines() if line.startswith("export WANDB_API_KEY="))
    assert "${WANDB_API_KEY:-}" in api_key_line


def test_single_runner_replaces_phase_and_pretraining_copies():
    root = pathlib.Path(__file__).parents[1]
    wrappers = [
        "run_all_at_once.py",
        "run_all_at_once_umt.py",
        "run_normal.py",
        "run_phase1.py",
        "run_phase1_umt.py",
        "run_phase2.py",
        "run_phase3.py",
        "run_phase4.py",
        "run_pretrain_modality1_proper.py",
        "run_pretrain_modality2_blurred.py",
        "run_pretrain_modality2_proper.py",
    ]
    for name in wrappers:
        content = (root / "scripts/python_new" / name).read_text(encoding="utf-8")
        assert "scripts.python_new.run_single import main" in content
        assert len(content.splitlines()) <= 10
    for name in ("run_all_at_once_umt.py", "run_phase1_umt.py"):
        content = (root / "scripts/python_new" / name).read_text(encoding="utf-8")
        assert "umt=True" in content
        assert "def objective" not in content


def test_shell_launchers_use_only_the_repo_gh200_environment():
    root = pathlib.Path(__file__).parents[1]
    launchers = list((root / "scripts/bash").glob("*.sh"))
    content = "\n".join(path.read_text(encoding="utf-8") for path in launchers)
    assert "conda activate clpi_env" not in content
    assert "plgrid-gpu-a100" not in content
    assert "partition=rtx3080" not in content
    assert (root / "scripts/bash/run_experiment.sh").exists()


def test_unified_runner_computes_phase_bounds_and_checkpoint_contract():
    from omegaconf import OmegaConf

    from scripts.python_new.run_single import (
        MODE_SPECS,
        _checkpoint_restore_policy,
        _run_bounds,
        _validate_resume_state,
        _validate_resume_protocol,
    )

    config = OmegaConf.create(
        {
            "phase1": 10,
            "phase2": 20,
            "phase3": 5,
            "phase4": 7,
            "model_checkpoint": "model.pth",
        }
    )
    assert _run_bounds("all_at_once", MODE_SPECS["all_at_once"], config) == (
        0, 42, None,
    )
    assert _run_bounds("phase1", MODE_SPECS["phase1"], config) == (0, 10, None)
    assert _run_bounds("phase2", MODE_SPECS["phase2"], config) == (
        10,
        30,
        "model.pth",
    )
    assert _run_bounds("phase3", MODE_SPECS["phase3"], config) == (
        30,
        35,
        "model.pth",
    )
    assert _checkpoint_restore_policy(config, explicit_resume=False) is False
    assert _checkpoint_restore_policy(config, explicit_resume=True) is True
    config.transfer_training_state = True
    config.restore_training_state = False
    assert _checkpoint_restore_policy(config, explicit_resume=False) is True
    assert _checkpoint_restore_policy(config, explicit_resume=True) is False

    legacy_state = {"is_training_checkpoint": False}
    with pytest.raises(ValueError, match="versioned training checkpoint"):
        _validate_resume_state(
            legacy_state,
            explicit_resume=True,
            restore_training_state=True,
            resume_start_epoch=None,
        )

    training_state = {"is_training_checkpoint": True}
    with pytest.raises(ValueError, match="explicit resume_start_epoch"):
        _validate_resume_state(
            training_state,
            explicit_resume=True,
            restore_training_state=False,
            resume_start_epoch=None,
        )

    _validate_resume_state(
        training_state,
        explicit_resume=True,
        restore_training_state=False,
        resume_start_epoch=30,
    )

    manifest = {
        "version": 1,
        "loader": {"batch_size": 64},
        "training": {"seed": 83},
    }
    resume_with_manifest = {
        "metadata": {"protocol_manifest": manifest}
    }
    _validate_resume_protocol(
        resume_with_manifest, manifest, explicit_resume=True
    )
    changed_manifest = {
        **manifest,
        "loader": {"batch_size": 128},
    }
    with pytest.raises(ValueError, match=r"loader\.batch_size"):
        _validate_resume_protocol(
            resume_with_manifest, changed_manifest, explicit_resume=True
        )
    with pytest.raises(ValueError, match="has no protocol manifest"):
        _validate_resume_protocol(
            {"metadata": {}}, manifest, explicit_resume=True
        )
    _validate_resume_protocol(
        {"metadata": {}},
        manifest,
        explicit_resume=True,
        allow_missing_manifest=True,
    )


def test_tfim_explicit_epoch_schedule_overrides_periodic_cadence():
    from omegaconf import OmegaConf

    from src.trainer.trainer_classification_mm_clp import fim_measurement_due

    schedule = [5, 8, 11, 14, 17]
    observed = [
        epoch
        for epoch in range(1, 21)
        if fim_measurement_due(epoch, 1, schedule)
    ]
    assert observed == schedule
    assert fim_measurement_due(10, 5) is True
    assert fim_measurement_due(11, 5) is False
    with pytest.raises(ValueError, match="strictly increasing"):
        fim_measurement_due(5, 1, [5, 5, 8])

    root = pathlib.Path(__file__).parents[1] / "configs" / "experiments"
    refinement = OmegaConf.load(
        root / "cifar10_phase4_tfim_refinement_p1_120.yaml"
    )
    assert refinement.phase4 == 200
    assert list(refinement.fim_eval_epochs) == schedule
    assert refinement.fim_chunk_size == 256
    assert refinement.phase4_test_policy == "disabled"

    candidate = OmegaConf.load(
        root / "cifar10_phase3_oracle_candidate_p1_120.yaml"
    )
    assert candidate.mode == "phase3"
    assert candidate.transfer_training_state is True
    assert candidate.phase3_intervention.mode == "deactivation"
    assert candidate.phase4 == 0

    probe = OmegaConf.load(
        root / "cifar10_phase4_tfim_probe_p1_120.yaml"
    )
    assert probe.phase1 == 120 and probe.phase2 == 200
    assert probe.phase3 == 0 and probe.phase4 == 17
    assert list(probe.fim_eval_epochs) == schedule
    assert probe.phase4_test_policy == "disabled"

    gold_probe = OmegaConf.load(
        root / "cifar10_gold_tfim_p2_17.yaml"
    )
    assert gold_probe.mode == "all_at_once"
    assert gold_probe.phase1 == 0 and gold_probe.phase2 == 17
    assert gold_probe.phase3 == 0 and gold_probe.phase4 == 0
    assert list(gold_probe.fim_eval_epochs) == schedule
    assert gold_probe.phase2_stopping.mode == "disabled"
    assert gold_probe.phase2_stopping.duration_policy == "diagnostic_fixed"
    assert gold_probe.phase2_test_policy == "disabled"
    assert gold_probe.phase4_test_policy == "disabled"

    assert refinement.phase4_selection.primary_metric == "accuracy"


def test_recovery_ranking_uses_one_point_full_gap_equivalence_band():
    from scripts.python_new.analyze_tfim_p4_refinement import (
        _select_best_recovery,
    )

    branch_perfect = {
        "e3": 10,
        "full_accuracy_gap_abs": 0.010,
        "branch_accuracy_gap_mean_abs": 0.0,
        "dominant_accuracy_gap_abs": 0.0,
        "weak_accuracy_gap_abs": 0.0,
    }
    full_closer = {
        "e3": 20,
        "full_accuracy_gap_abs": 0.005,
        "branch_accuracy_gap_mean_abs": 0.2,
        "dominant_accuracy_gap_abs": 0.2,
        "weak_accuracy_gap_abs": 0.2,
    }
    secondary_best = {
        **full_closer,
        "e3": 30,
        "branch_accuracy_gap_mean_abs": 0.03,
        "dominant_accuracy_gap_abs": 0.02,
        "weak_accuracy_gap_abs": 0.04,
    }
    outside_band = {
        "e3": 40,
        "full_accuracy_gap_abs": 0.016,
        "branch_accuracy_gap_mean_abs": 0.0,
        "dominant_accuracy_gap_abs": 0.0,
        "weak_accuracy_gap_abs": 0.0,
    }

    assert _select_best_recovery((branch_perfect, full_closer)) is branch_perfect
    assert _select_best_recovery((full_closer, secondary_best)) is secondary_best
    assert _select_best_recovery((full_closer, outside_band)) is full_closer


def test_tfim_replay_resolves_one_log_across_artifact_roots(tmp_path):
    from scripts.python_new.analyze_tfim_p4_refinement import (
        _find_log_in_directories,
    )

    old_root = tmp_path / "old"
    storage_root = tmp_path / "storage"
    old_root.mkdir()
    storage_root.mkdir()
    expected = storage_root / "probe-123.out"
    expected.touch()

    assert _find_log_in_directories([old_root, storage_root], "123") == expected


def test_gold_slope_stopper_replay_stops_at_first_crossing():
    from scripts.python_new.replay_tfim_gold_slope_stopper import replay_seed

    rows = [
        {"e3": 20, "slope_log_ratio": 0.30},
        {"e3": 40, "slope_log_ratio": 0.10},
        {"e3": 60, "slope_log_ratio": -0.05},
        {"e3": 80, "slope_log_ratio": 0.02},
    ]

    replay = replay_seed(rows, gold_slope=0.0)

    assert replay["status"] == "crossing_bracket_found"
    assert replay["bracket_left_e3"] == 40
    assert replay["bracket_right_e3"] == 60
    assert [row["e3"] for row in replay["revealed"]] == [20, 40, 60]
    assert replay["revealed"][-1]["decision"] == "stop_and_refine_first_crossing"


def test_phase3_materialization_checkpoints_do_not_require_evaluation():
    source = pathlib.Path(
        "src/trainer/trainer_validation_clp.py"
    ).read_text()

    assert 'section.get(\n                "materialization_checkpoint_epochs", []' in source
    assert '"checkpoint_role": "materialization_without_evaluation"' in source
    materialize = source.index("if local_epoch in materialization_epochs:")
    evaluate = source.index("if not should_evaluate_phase_epoch(", materialize)
    assert materialize < evaluate


def test_publication_validation_cadence_and_wandb_mode_are_explicit():
    from omegaconf import OmegaConf

    root = pathlib.Path(__file__).parents[1]
    names = (
        "cifar10_validation_protocol_adaptive_seed83.yaml",
        "cifar10_validation_protocol_observe_seed83.yaml",
        "cifar10_pais_calibration.yaml",
    )
    for name in names:
        config = OmegaConf.load(root / "configs" / "experiments" / name)
        assert config.phase1_validation_interval_epochs == 5
        assert config.phase2_stopping.eval_interval_epochs == 5
        assert config.phase3_stopping.eval_interval_epochs == 5
        assert config.phase4_selection.eval_interval_epochs == 5
        assert config.logger == "wandb"
        assert config.logger_mode == "online"


def test_recovery_configs_use_dedicated_wandb_project_and_rule():
    from omegaconf import OmegaConf

    root = pathlib.Path(__file__).parents[1]
    names = (
        "cifar10_pais_recovery_calibration.yaml",
        "cifar10_validation_protocol_recovery_seed83.yaml",
    )
    for name in names:
        config = OmegaConf.load(root / "configs" / "experiments" / name)
        assert config.phase3_stopping.decision_rule == "weak_recovery"
        assert config.phase3_stopping.emergency_stop_mode == "numerical_only"
        assert config.phase3_stopping.eval_interval_epochs == 5
        assert config.wandb_entity == "bartekk"
        assert config.wandb_project == "CLPIntervention_PAIS"
        assert config.logger_mode == "online"

    calibration = OmegaConf.load(
        root
        / "configs"
        / "experiments"
        / "cifar10_pais_recovery_calibration.yaml"
    )
    assert calibration.phase3_stopping.shadow_continue_after_stop
    assert list(calibration.phase3_stopping.calibration_milestone_epochs) == [
        20,
        40,
        60,
        80,
        120,
    ]
    assert calibration.phase3_stopping.confidence_family_size == 9

    shadow_smoke = OmegaConf.load(
        root
        / "configs"
        / "experiments"
        / "cifar10_pais_recovery_shadow_smoke.yaml"
    )
    assert shadow_smoke.phase3_stopping.mode == "observe_only"
    assert shadow_smoke.phase3_stopping.shadow_continue_after_stop
    assert list(
        shadow_smoke.phase3_stopping.calibration_milestone_epochs
    ) == [2, 4]
    assert shadow_smoke.wandb_project == "CLPIntervention_PAIS"


def test_local_accuracy_profile_uses_dense_prefix_and_four_look_window():
    from omegaconf import OmegaConf

    root = pathlib.Path(__file__).parents[1]
    config = OmegaConf.load(
        root
        / "configs"
        / "experiments"
        / "cifar10_phase3_stopper_observe_p1_40.yaml"
    )
    stopping = config.phase3_stopping
    assert stopping.min_epochs == 4
    assert stopping.initial_dense_eval_epochs == 4
    assert stopping.eval_interval_epochs == 4
    assert stopping.minimum_exposure_evaluations == 4
    assert stopping.trend_window == 4


def test_relative_unimodal_parity_configs_are_explicit_and_test_safe():
    from omegaconf import OmegaConf

    root = pathlib.Path(__file__).parents[1] / "configs" / "experiments"
    reference = OmegaConf.load(root / "cifar10_unimodal_reference.yaml")
    assert reference.protocol_name == "cifar10_unimodal_reference_v2"
    assert reference.unimodal_reference_training
    assert (
        reference.unimodal_initialization_policy
        == "canonical_bimodal_components_v2"
    )
    assert reference.epochs == 200
    assert reference.unimodal_reference_eval_interval_epochs == 5
    assert reference.phase2_test_policy == "disabled"
    assert reference.phase4_test_policy == "disabled"
    assert reference.wandb_project == "CLPIntervention_UnimodalParity"

    parity = OmegaConf.load(
        root / "cifar10_relative_unimodal_parity_p1_40.yaml"
    )
    stopping = parity.phase3_stopping
    assert stopping.mode == "enforce"
    assert stopping.decision_rule == "relative_unimodal_parity"
    assert stopping.parity_patience == 2
    assert stopping.recovery_fraction_threshold == 1.0
    assert stopping.initial_dense_eval_epochs == 4
    assert stopping.eval_interval_epochs == 4
    assert parity.phase4_test_policy == "final_only"
    assert parity.unimodal_references.left_checkpoint is None
    assert parity.unimodal_references.right_checkpoint is None
    assert parity.wandb_project == "CLPIntervention_UnimodalParity"

    trajectory = OmegaConf.load(
        root / "cifar10_relative_unimodal_parity_trajectory_p1_40.yaml"
    )
    trajectory_stopping = trajectory.phase3_stopping
    assert trajectory_stopping.mode == "observe_only"
    assert trajectory_stopping.recovery_fraction_threshold == 1.0
    assert trajectory.phase3 == 200
    assert trajectory.phase4 == 0
    assert trajectory_stopping.shadow_continue_after_stop is False
    assert trajectory_stopping.calibration_milestone_epochs[:4] == [1, 2, 3, 4]
    assert trajectory_stopping.calibration_milestone_epochs[-1] == 200
    assert len(trajectory_stopping.calibration_milestone_epochs) == 53
    assert trajectory.phase4_test_policy == "disabled"

    recovery_phase4 = OmegaConf.load(
        root / "cifar10_phase4_unimodal_recovery_fraction.yaml"
    )
    assert recovery_phase4.mode == "phase4"
    assert recovery_phase4.phase4 == 200
    assert recovery_phase4.phase4_selection.primary_metric == "accuracy"
    assert recovery_phase4.phase4_test_policy == "disabled"
    assert (
        recovery_phase4.wandb_project
        == "CLPIntervention_UnimodalParity"
    )

    shared_trajectory = OmegaConf.load(
        root / "cifar10_shared_trunk_trajectory_p1_120.yaml"
    )
    shared_stopping = shared_trajectory.phase3_stopping
    assert shared_trajectory.phase1 == 120
    assert shared_trajectory.phase2 == 200
    assert shared_trajectory.phase3 == 200
    assert shared_trajectory.phase4 == 0
    assert shared_stopping.mode == "observe_only"
    assert shared_stopping.decision_rule == "local_accuracy"
    assert shared_stopping.observe_phase4_transition == "endpoint"
    assert shared_stopping.shadow_continue_after_stop is True
    assert shared_stopping.eval_interval_epochs == 4
    assert list(shared_stopping.calibration_milestone_epochs[:4]) == [
        1, 2, 3, 4
    ]
    assert list(shared_stopping.calibration_milestone_epochs[-10:]) == [
        20, 40, 60, 80, 100, 120, 140, 160, 180, 200
    ]
    assert shared_trajectory.fim_measurements_per_epoch == 0
    assert shared_trajectory.phase2_test_policy == "disabled"
    assert shared_trajectory.phase4_test_policy == "disabled"

    umt = OmegaConf.load(root / "cifar10_umt_phase3_p1_120.yaml")
    assert umt.mode == "phase3"
    assert umt.umt is True
    assert umt.phase1 == 120 and umt.phase2 == 200
    assert umt.phase3 == 200 and umt.phase4 == 0
    assert umt.phase3_intervention.mode == "full_active"
    assert umt.phase3_stopping.mode == "observe_only"
    assert umt.phase3_stopping.observe_phase4_transition == "endpoint"
    assert umt.phase3_stopping.eval_interval_epochs == 20
    assert umt.phase3_stopping.initial_dense_eval_epochs == 0
    assert list(umt.phase3_stopping.calibration_milestone_epochs) == list(
        range(20, 201, 20)
    )
    assert umt.phase2_test_policy == "disabled"
    assert umt.phase4_test_policy == "disabled"

    frozen_left = OmegaConf.load(
        root / "cifar10_frozen_left_active_phase3_p1_120.yaml"
    )
    assert frozen_left.mode == "phase3"
    assert frozen_left.phase3_intervention.mode == "frozen_left_active"
    assert frozen_left.phase3 == 200 and frozen_left.phase4 == 0
    assert frozen_left.phase3_stopping.mode == "observe_only"
    assert frozen_left.phase4_test_policy == "disabled"

    frozen_left_phase4 = OmegaConf.load(
        root / "cifar10_phase4_from_frozen_left_active_p1_120.yaml"
    )
    assert frozen_left_phase4.mode == "phase4"
    assert frozen_left_phase4.phase4 == 200
    assert frozen_left_phase4.fim_measurements_per_epoch == 1
    assert frozen_left_phase4.fim_eval_interval_epochs == 10
    assert frozen_left_phase4.fim_chunk_size == 256

    from scripts.python_new.run_single import MODE_SPECS

    assert MODE_SPECS["phase4"].trace_fim is True

    oracle = OmegaConf.load(
        root / "cifar10_phase4_tfim_oracle_p1_120.yaml"
    )
    assert oracle.phase4 == 40
    assert oracle.fim_eval_interval_epochs == 5
    assert oracle.fim_chunk_size == 256
    assert oracle.phase4_test_policy == "disabled"

    clean_fim = OmegaConf.load(root / "cifar10_clean_p2_tfim_50.yaml")
    assert clean_fim.phase1 == 0 and clean_fim.phase2 == 50
    assert clean_fim.phase2_stopping.duration_policy == "diagnostic_fixed"
    assert clean_fim.fim_eval_interval_epochs == 5
    assert clean_fim.fim_chunk_size == 256
    assert clean_fim.phase2_test_policy == "disabled"

    unimodal_fim = OmegaConf.load(root / "cifar10_unimodal_tfim_50.yaml")
    assert unimodal_fim.protocol_name == "cifar10_unimodal_tfim_50_v2"
    assert unimodal_fim.epochs == 50
    assert unimodal_fim.unimodal_reference_training is True
    assert (
        unimodal_fim.unimodal_initialization_policy
        == "canonical_bimodal_components_v2"
    )
    assert unimodal_fim.fim_eval_interval_epochs == 5
    assert unimodal_fim.fim_chunk_size == 256

    from src.trainer.trainer_classification_mm_clp_umt import (
        TrainerClassification as UMTTrainer,
        validation_controlled_umt_trainer_class,
    )
    from src.trainer.trainer_validation_clp import ValidationControlledTrainer

    combined = validation_controlled_umt_trainer_class()
    assert issubclass(combined, UMTTrainer)
    assert issubclass(combined, ValidationControlledTrainer)
    assert combined.compute_loss is UMTTrainer.compute_loss

    diagnostic = OmegaConf.load(
        root / "cifar10_phase4_compatibility_diagnostic.yaml"
    )
    assert diagnostic.mode == "phase4"
    assert diagnostic.phase4 == 10
    assert diagnostic.phase4_test_policy == "disabled"
    assert diagnostic.phase4_lr_warmup_epochs == 4
    assert diagnostic.phase4_lr_warmup_start_factor == 0.1
    assert diagnostic.phase4_diagnostics.enabled is True
    assert list(diagnostic.phase4_diagnostics.dense_eval_epochs) == [
        0, 1, 2, 3, 4, 5, 10
    ]
    assert list(diagnostic.phase4_diagnostics.hybrid_eval_epochs) == [
        0, 1, 2, 3, 4, 5, 10
    ]
    assert diagnostic.phase4_staged_unfreezing.enabled is False
    assert diagnostic.phase4_staged_unfreezing.shared_only_epochs == 4
    auxiliary = OmegaConf.load(
        root / "cifar10_phase4_weak_auxiliary_loss.yaml"
    )
    assert auxiliary.mode == "phase4"
    assert auxiliary.phase4 == 10
    assert auxiliary.phase4_auxiliary_loss.enabled is True
    assert auxiliary.phase4_auxiliary_loss.weak_weight == 1.0
    assert auxiliary.phase4_auxiliary_loss.dominant_weight == 0.0
    assert auxiliary.phase4_staged_unfreezing.enabled is False
    run_source = (
        pathlib.Path(__file__).parents[1]
        / "scripts/python_new/run_single.py"
    ).read_text(encoding="utf-8")
    assert '"phase4_staged_unfreezing": config.get(' in run_source
    assert '"phase4_auxiliary_loss": config.get(' in run_source


def test_gold_standard_minimal_exposure_and_grid_contracts():
    from omegaconf import OmegaConf

    root = pathlib.Path(__file__).parents[1] / "configs" / "experiments"
    gold = OmegaConf.load(
        root / "cifar10_clean_gold_standard_p1_0_p2_200.yaml"
    )
    assert [gold.phase1, gold.phase2, gold.phase3, gold.phase4] == [0, 200, 0, 0]
    assert gold.phase2_stopping.mode == "enforce"
    assert gold.phase2_stopping.selection_scope == "global"
    assert gold.phase2_stopping.primary_metric == "accuracy"
    assert gold.phase2_test_policy == "posthoc_final"
    assert gold.phase4_test_policy == "disabled"
    assert gold.wandb_project == "CLPIntervention_Phase3Stopping"

    minimal = OmegaConf.load(
        root / "cifar10_minimal_blurred_exposure_p1_1_p2_200.yaml"
    )
    assert [
        minimal.phase1,
        minimal.phase2,
        minimal.phase3,
        minimal.phase4,
    ] == [1, 200, 0, 0]
    assert minimal.phase2_test_policy == "disabled"

    observe = OmegaConf.load(
        root / "cifar10_phase3_stopper_observe_p1_40.yaml"
    )
    assert observe.phase3_stopping.mode == "observe_only"
    assert observe.phase4 == 200
    assert (
        observe.phase3_stopping.observe_phase4_transition
        == "hypothetical_selected"
    )
    assert list(
        observe.phase3_stopping.calibration_milestone_epochs
    ) == [20, 40, 60, 80, 200]
    assert observe.phase3_stopping.recovery_primary_metric == "accuracy"
    assert observe.phase3_lr_warmup_epochs == 4
    assert observe.phase3_lr_warmup_start_factor == 0.1
    assert observe.phase2_test_policy == "disabled"
    assert observe.phase4_test_policy == "disabled"

    grid = OmegaConf.load(root / "cifar10_phase4_recovery_grid.yaml")
    assert grid.validation_protocol is True
    assert grid.mode == "phase4"
    assert grid.phase4 == 200
    assert grid.phase4_test_policy == "disabled"
    assert grid.phase2_stopping.mode == "disabled"
    assert grid.phase3_stopping.mode == "disabled"
    assert grid.wandb_project == "CLPIntervention_Phase3Stopping"

    milestone = OmegaConf.load(
        root / "cifar10_phase4_milestone_p1_40.yaml"
    )
    assert milestone.validation_protocol is True
    assert milestone.mode == "phase4"
    assert milestone.phase1 == 40
    assert milestone.phase4 == 200
    assert milestone.phase4_test_policy == "final_only"
    assert milestone.phase4_selection.primary_metric == "accuracy"


def test_phase3_and_phase4_route_to_validation_controller_when_enabled():
    from omegaconf import OmegaConf

    from scripts.python_new.run_single import MODE_SPECS, _uses_validation_protocol

    enabled = OmegaConf.create({"validation_protocol": True})
    disabled = OmegaConf.create({"validation_protocol": False})
    assert _uses_validation_protocol(MODE_SPECS["all_at_once"], enabled)
    assert _uses_validation_protocol(MODE_SPECS["phase4"], enabled)
    assert _uses_validation_protocol(MODE_SPECS["phase3"], enabled)
    assert not _uses_validation_protocol(MODE_SPECS["phase4"], disabled)
