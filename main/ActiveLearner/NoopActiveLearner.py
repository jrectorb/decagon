from .BaseActiveLearner import BaseActiveLearner
from ..Dtos.DataSet import DataSet
from ..Dtos.IterationResults import IterationResults
from ..Utils.Config import Config

class NoopActiveLearner(BaseActiveLearner, dataSetType=None):
    def __init__(self, config: Config) -> None:
        # Increment this everytime hasUpdate is called
        self.numIters: int = 0

    def hasUpdate(
        self,
        dataSet: DataSet,
        iterResults: IterationResults
    ) -> bool:
        return self.numIters == 0 and iterResults is None

    def getUpdate(
        self,
        dataSet: DataSet,
        iterResults: IterationResults
    ) -> DataSet:
        self.numIters += 1

        return dataSet

