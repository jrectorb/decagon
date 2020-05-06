from abc import ABCMeta, abstractmethod
from ..Trainable import Trainable
from ..Utils.BaseFactorizableClass import BaseFactorizableClass
from ..Utils.Config import Config

class BaseTrainer(BaseFactorizableClass, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, trainable: Trainable, config: Config) -> None:
        pass

    @abstractmethod
    def train(self):
        pass

