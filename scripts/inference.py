# project-root(demo-ml-project)/scripts/inference.py

from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from demo_ml_project.configs.training.schema import RootConfig
from demo_ml_project.pipelines.inference_pipeline import InferencePipeline
from demo_ml_project.utils.logging import configure_logging, get_logger


@hydra.main(version_base=None, config_path="../conf/inference", config_name="default")
def main(hydra_cfg):
    OmegaConf.resolve(hydra_cfg)
    cfg = RootConfig.model_validate(OmegaConf.to_container(hydra_cfg, resolve=True))

    # Setup logging (similar to train.py)
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    configure_logging(
        log_dir=run_dir / "logs",
        log_file=f"inference_{timestamp}.log",
        default_level="INFO",
    )

    logger = get_logger(__name__)
    logger.info("Inference pipeline starting...")

    pipeline = InferencePipeline(cfg)
    try:
        pipeline.run()
        logger.info("Inference completed ✓")
    except Exception:
        logger.exception("Inference failed")
        raise


if __name__ == "__main__":
    main()







# python scripts/inference.py \
#   input.path=data/processed/model_df.parquet \
#   output.path=results/my_preds.csv \
#   batch_size=512