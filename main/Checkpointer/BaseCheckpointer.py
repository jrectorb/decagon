from abc import ABCMeta, abstractmethod

class BaseCheckpointer(metaclass=ABCMeta):
    def __init__(self, config: Config) -> None:
        self.numIterationsPerLog: int = int(config.getSetting('NumIterationsPerLog'))
        self.numIterationsDone: int = 0

    def incrementIterations(self):
        self.numIterationsDone += 1

    @property
    def shouldCheckpoint(self):
        return (self.numIterationsDone % self.numItersPerCheckpoint) == 0

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def restore(self):
        pass

