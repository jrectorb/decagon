from abc import ABCMeta, abstractmethod
from ..Utils.Config import Config

class BaseCheckpointer(metaclass=ABCMeta):
    def __init__(self, config: Config) -> None:
        self.numIterationsDone: int = 0
        self.numIterationsPerCheckpoint: int = int(
            config.getSetting('NumIterationsPerCheckpoint')
        )

    def incrementIterations(self):
        self.numIterationsDone += 1

    @property
    def shouldCheckpoint(self):
        return (self.numIterationsDone % self.numIterationsPerCheckpoint) == 0

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def restore(self):
        pass

