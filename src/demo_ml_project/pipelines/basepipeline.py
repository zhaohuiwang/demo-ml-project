

"""Base pipeline."""
from abc import ABC
from typing import Any


class BasePipeline(ABC):
    """Abstract Class for an pipeline."""

    def __init__(self) -> None:
        """Initialize a pipeline."""

    def run(self) -> Any:
        """
        Call the pipeline.

        Raises
        ------
        NotImplementedError
            not implemented yet

        """
        raise NotImplementedError("functional __call__ not implemented")


# TrainingPipeline
# InferencePipeline
# DataIngestionPipeline
# ModelPipeline: A general name often used to encapsulate the end-to-end model creation and evaluation. 

# These classes generally contain a main method like run() or execute() that orchestrates various components (DataIngestion, ModelTrainer, etc.). 

# Option I
# class BasePipeline(ABC):
#     def run(self):
#         self.setup()
#         try:
#             result = self._run()
#             self.teardown()
#             return result
#         except Exception:
#             self.on_failure()
#             raise

# Option II
# from abc import ABC, abstractmethod

# class BasePipeline(ABC):

#     @abstractmethod
#     def run(self):
#         pass
# Without that, it's more of a convention than a contract.
# BasePipeline is a scaling tool, not a starter tool
# Use it when you need: consistency, orchestration, extensibility

# # Option III
# # Functions for core logic; Thin pipeline classes for orchestration
# def train_model(cfg):
#     ...

# class TrainingPipeline(BasePipeline):
#     def run(self):
#         return train_model(self.cfg)
