from abc import ABCMeta, abstractmethod
from ..Dtos.DataSet import DataSet
from ..Dtos.Trainable import Trainable
from ..Utils.BaseFactorizableClass import BaseFactorizableClass
from ..Utils.Config import Config

class BaseTrainableBuilder(BaseFactorizableClass, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, dataSet: DataSet, config: Config) -> None:
        pass

    @abstractmethod
    def build(self) -> Trainable
        pass

