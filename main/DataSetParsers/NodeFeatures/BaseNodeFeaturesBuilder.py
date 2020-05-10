from abc import ABCMeta, abstractmethod
from typing import Dict, TypeVar
from ...Dtos.NodeFeatures import NodeFeatures

class BaseDataSetBuilder(BaseFactorizableClass, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def build() -> NodeFeatures:
        pass

