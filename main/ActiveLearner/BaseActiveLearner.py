from abc import ABCMeta, abstractmethod
from ..Dtos.DataSet import DataSet
from ..Dtos.IterationResults import IterationResults
from ..Utils.Config import Config

class BaseActiveLearner(BaseFactorizableClass, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def hasUpdate(
        self,
        dataSet: DataSet,
        iterResults: IterationResults
    ) -> bool:
        pass

    @abstractmethod
    def getUpdate(
        self,
        dataSet: DataSet,
        iterResults: IterationResults
    ) -> DataSet:
        pass

