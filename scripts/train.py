# project-root(demo-ml-project)/scripts/train.py

from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from demo_ml_project.configs.training.schema import RootConfig
from demo_ml_project.pipelines.model_training import TrainingPipeline
from demo_ml_project.utils.logging import configure_logging, get_logger


# ──────────────────────────────────────────────
# Safety check – ensure we're in project root
# ──────────────────────────────────────────────
PROJECT_MARKERS = [
    #Path("conf/training/default.yaml"),
    Path("src/demo_ml_project/__init__.py"),
    Path("scripts/train.py"),
]

missing = [p for p in PROJECT_MARKERS if not p.is_file()]

if missing:
    raise RuntimeError(
        f"Project root markers not found:\n  " + "\n  ".join(str(p) for p in missing) +
        "\nAre you sure you're running from the project root?"
    )


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="training_conf" 
)
def main(hydra_cfg: RootConfig):

    # Eagerly resolves all interpolations in place.
    OmegaConf.resolve(hydra_cfg)
    
    # Convert → pydantic (this runs all validation & path resolution)
    cfg = RootConfig.model_validate(OmegaConf.to_container(hydra_cfg, resolve=True))

    # Log resolved config (while still DictConfig)
    logger = get_logger(__name__)
    logger.debug("Resolved config:\n%s", cfg)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    configure_logging(
        log_dir=run_dir / "logs",
        log_file=f"train_{timestamp}.log",
        default_level="INFO",
        json_format=False,
    )

    logger = get_logger(__name__)
    logger.info("Output directory: %s", run_dir)

    # No conversion needed — cfg is already structured & native-typed
    pipeline = TrainingPipeline(cfg)

    try:
        pipeline.run()
        logger.info("Training pipeline completed ✓")
    except Exception:
        logger.exception("Training failed")
        raise

if __name__ == "__main__":
    main()

# look into `from hydra_zen import zen` for more advance sugar features


# # Dependency
# project-root(demo-ml-project)/conf/config.yaml
# /src/demo_ml_project/pipelines/model_training.py

