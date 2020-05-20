from abc import ABCMeta, abstractmethod
from typing import Type
from ..Dtos.DataSet import DataSet
from ..Dtos.IterationResults import IterationResults
from ..Dtos.Trainable import Trainable
from ..Utils.BaseFactorizableClass import BaseFactorizableClass
from ..Utils.Config import Config

class BaseTrainableBuilder(BaseFactorizableClass, functionalityType=None, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, dataSet: DataSet, config: Config) -> None:
        pass

    @abstractmethod
    def build(self) -> Type[Trainable]:
        pass

