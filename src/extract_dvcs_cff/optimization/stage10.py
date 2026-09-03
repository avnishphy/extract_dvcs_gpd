"""Persistent Optuna search for the Stage 10 neural posterior.

Optuna changes only neural architecture and optimizer controls.  It never
changes GPD physics, pseudodata, covariance, priors, or validation gates.
Each trial trains against the already generated simulation bank and returns
the best validation negative log density.  The held-out test partition is not
evaluated during selection.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


def optimize_hyperparameters(
    *,
    configuration_path: Path,
    workspace: Path,
    profile: str,
    n_trials: int | None = None,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Run or resume a deterministic, validation-only Optuna study."""

    import optuna

    from extract_dvcs_cff.workflows.pseudodata import (
        load_configuration,
        sha256,
        train_model,
        write_json,
    )
    from extract_dvcs_cff.progress import progress_enabled

    config = load_configuration(configuration_path)
    controls = config["hyperparameter_optimization"]
    trial_count = int(
        n_trials if n_trials is not None else controls["trials"][profile]
    )
    if not 1 <= trial_count <= 1000:
        raise ValueError("Optuna trial count must be in [1,1000]")
    generated = workspace / "generated"
    if not generated.is_dir():
        raise RuntimeError("generate must run before optimize")

    output = workspace / "optimization"
    output.mkdir(parents=True, exist_ok=True)
    storage_path = (output / "study.sqlite3").resolve()
    storage = f"sqlite:///{storage_path}"
    study_name = f"{config['workflow']}__{profile}"
    sampler = optuna.samplers.TPESampler(
        seed=int(config["seeds"]["optimization"])
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(controls["pruner_startup_trials"]),
        n_warmup_steps=int(controls["pruner_warmup_epochs"]),
    )
    storage_backend = optuna.storages.RDBStorage(
        url=storage,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_backend,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )
    search = controls["search_space"]

    def objective(trial: optuna.Trial) -> float:
        trial_config = deepcopy(config)
        network = trial_config["network"]
        for key in (
            "point_hidden",
            "dataset_hidden",
            "embedding_features",
            "flow_hidden_features",
            "num_transforms",
            "training_batch_size",
        ):
            network[key] = trial.suggest_categorical(key, search[key])
        network["learning_rate"] = trial.suggest_float(
            "learning_rate",
            float(search["learning_rate"][0]),
            float(search["learning_rate"][1]),
            log=True,
        )
        trial_config["profiles"][profile]["ensemble_seeds"] = [
            int(config["seeds"]["optimization"])
        ]
        trial_config["profiles"][profile][
            "active_ensemble_member_count"
        ] = 1

        trial_root = output / "trials" / f"trial_{trial.number:04d}"
        trial_root.mkdir(parents=True, exist_ok=False)
        trial_workspace = trial_root / "workspace"
        trial_workspace.mkdir()
        (trial_workspace / "generated").symlink_to(
            generated.resolve(), target_is_directory=True
        )
        trial_config_path = trial_root / "workflow.json"
        write_json(trial_config_path, trial_config)
        training = train_model(
            configuration_path=trial_config_path,
            workspace=trial_workspace,
            profile=profile,
            evaluate_test=False,
            show_progress=show_progress,
        )
        member = next(iter(training["members"].values()))
        history = member["validation_loss_by_epoch"]
        for epoch, value in enumerate(history):
            trial.report(float(value), step=epoch)
        objective_value = float(
            member["best_validation_negative_log_density"]
        )
        trial.set_user_attr("test_set_evaluated", False)
        trial.set_user_attr("training_metrics", str(
            trial_workspace / "training" / "training_summary.json"
        ))
        # sbi exposes the validation history after train() returns, so this
        # release records pruning decisions but does not claim compute-saving
        # mid-epoch interruption.
        if trial.should_prune():
            raise optuna.TrialPruned()
        return objective_value

    try:
        study.optimize(
            objective,
            n_trials=trial_count,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=progress_enabled(show_progress),
        )
    except BaseException:
        storage_backend.remove_session()
        storage_backend.engine.dispose()
        raise
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError("Optuna study has no completed trial")
    best_network = deepcopy(config["network"])
    best_network.update(study.best_trial.params)
    record = {
        "schema_version": 1,
        "optimizer": "Optuna",
        "optuna_version": optuna.__version__,
        "study_name": study.study_name,
        "storage": str(storage_path),
        "configuration_sha256": sha256(configuration_path),
        "profile": profile,
        "objective": "best validation negative log density",
        "direction": "minimize",
        "test_set_used_for_selection": False,
        "parallel_trials": 1,
        "requested_new_trials": trial_count,
        "total_trials": len(study.trials),
        "completed_trials": len(completed),
        "best_trial_number": int(study.best_trial.number),
        "best_objective": float(study.best_value),
        "best_parameters": dict(study.best_params),
        "recommended_neural_posterior": best_network,
        "application_policy": (
            "copy recommended_neural_posterior into experiment.json in a "
            "new project, then generate/train/evaluate once on held-out test"
        ),
        "pruning_compute_savings_claimed": False,
        "real_data_used": False,
    }
    write_json(output / "optimization_summary.json", record)
    (output / "best_parameters.json").write_text(
        json.dumps(best_network, indent=4, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Python 3.13 warns when SQLite connections survive until interpreter
    # teardown. Explicitly release both Optuna's thread-local session and the
    # SQLAlchemy pool after all study fields have been serialized.
    storage_backend.remove_session()
    storage_backend.engine.dispose()
    return record
