from ..Dtos.DataSet import DataSet
from ..Dtos.Trainable import Trainable
from ..BaseTrainableBuilder import BaseTrainableBuilder
from ..Utils.Config import Config

class DecagonTrainableBuilder(BaseTrainableBuilder):
    def __init__(self, dataSet: DataSet, config: Config) -> None:
        self.dataSet: DataSet = dataSet
        self.config: Config = config

    def build(self) -> Trainable:
        return Trainable(idk)

