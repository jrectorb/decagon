from abc import ABCMeta, abstractmethod
from ...Dtos.NodeLists import NodeLists
from ...Utils.BaseFactorizableClass import BaseFactorizableClass
from ...Utils.Config import Config

class BaseNodeListsBuilder(BaseFactorizableClass, dataSetType=None, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def build(self) -> NodeLists:
        pass

