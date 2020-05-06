from ..Dtos.DataSet import DataSet
from ..Dtos.Trainable import Trainable
from ..Utils.Config import Config

PlaceholdersDict = Dict[str, tf.placeholder]

class DecagonTrainable(Trainable):
    def __init__(
        self,
        dataSetIterator,
        optimizer,
        model,
        placeholders: PlaceholdersDict
    ) -> None:
        super().__init__(dataSetIterator, optimizer, model)
        self.placeholders: PlaceholdersDict = placeholders

