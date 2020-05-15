from abc import ABCMeta, abstractmethod
from typing import Type
from ..Dtos.TrainingIterationResults import TrainingIterationResults
from ..Utils.BaseFactorizableClass import BaseFactorizableClass
from ..Utils.Config import Config

class BaseLogger(BaseFactorizableClass, functionalityType=None, metaclass=ABCMeta):
    def __init__(self, config: Config) -> None:
        self.numIterationsPerLog: int = int(config.getSetting('NumIterationsPerLog'))
        self.numIterationsDone: int = 0

    def incrementIterations(self):
        self.numIterationsDone += 1

    @property
    def shouldLog(self):
        return (self.numIterationsDone % self.numIterationsPerLog) == 0

    @abstractmethod
    def log(self, iterationResults: Type[TrainingIterationResults]) -> None:
        pass

