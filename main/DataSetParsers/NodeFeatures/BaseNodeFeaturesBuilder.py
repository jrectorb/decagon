from abc import ABCMeta, abstractmethod
from typings import Dict, TypeVar

class BaseDataSetBuilder(BaseFactorizableClass, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def build() -> NodeFeatures:
        pass

