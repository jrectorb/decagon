from typing import Dict
from ...Dtos.DataSet import DataSet
from ...Dtos.IterationResults import IterationResults
from ...Dtos.TensorflowTrainable import TensorflowTrainable
from ...Dtos.TypeShortcuts import PlaceholdersDict
from ...Utils.Config import Config
import tensorflow as tf

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

    def getIterationResults(self) -> IterationResults:
        return IterationResults()

