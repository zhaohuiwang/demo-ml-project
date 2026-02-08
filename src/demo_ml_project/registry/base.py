# Registry Interface

from abc import ABC, abstractmethod
from typing import Any

class ModelRegistry(ABC):

    @abstractmethod
    def save(self, model: Any, name: str, version: str) -> None:
        pass

    @abstractmethod
    def load(self, name: str, version: str) -> Any:
        pass



# No PyTorch import
# Storage-agnostic
# Testable with mocks