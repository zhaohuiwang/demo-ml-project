
# tests/test_pipeline_smoke.py
import pytest

from demo_ml_project.pipelines.model_training import TrainingPipeline
from demo_ml_project.configs.schema import RootConfig


@pytest.mark.integration
def test_pipeline_smoke(sample_cfg):
    # Minimal smoke test — just check it initializes without crashing
    pipeline = TrainingPipeline(sample_cfg)
    assert pipeline is not None
    assert pipeline.device.type in ("cpu", "cuda", "mps")