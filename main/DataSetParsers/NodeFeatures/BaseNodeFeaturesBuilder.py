from abc import ABCMeta, abstractmethod
from typing import Dict, TypeVar
from ...Dtos.NodeFeatures import NodeFeatures
from ...Utils.BaseFactorizableClass import BaseFactorizableClass
from ...Utils.Config import Config

class BaseNodeFeaturesBuilder(BaseFactorizableClass, dataSetType=None, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def build() -> NodeFeatures:
        pass

