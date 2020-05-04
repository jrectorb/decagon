from abc import ABCMeta, abstractmethod
from typings import Map

class BaseAdjacencyMatricesBuilder(BaseFactorizableClass, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        pass

    @abstractmethod
    def build(self) -> AdjacencyMatrices:
        pass

