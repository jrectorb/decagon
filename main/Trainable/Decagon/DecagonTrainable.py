from ..Dtos.DataSet import DataSet
from ..Dtos.Trainable import TensorflowTrainable
from ..Utils.Config import Config

PlaceholdersDict = Dict[str, tf.placeholder]

class DecagonTrainable(TensorflowTrainable):
    def __init__(
        self,
        dataSetIterator,
        optimizer,
        model,
        placeholders: PlaceholdersDict
    ) -> None:
        super().__init__(dataSetIterator, optimizer, model)
        self.placeholders: PlaceholdersDict = placeholders

