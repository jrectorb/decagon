from abc import ABCMeta, abstractmethod
from ...Dtos.AdjacencyMatrices import AdjacencyMatrices
from ...Dtos.NodeLists import NodeLists
from ...Utils.BaseFactorizableClass import BaseFactorizableClass
from ...Utils.Config import Config

class BaseAdjacencyMatricesBuilder(BaseFactorizableClass, dataSetType=None, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        pass

    @abstractmethod
    def build(self) -> AdjacencyMatrices:
        pass

