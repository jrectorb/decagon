from .Dtos.Enums.TrainableType import TrainableType
from .Dtos.Trainable import Trainable
from .GreedyActiveLearner import GreedyActiveLearner
from .Utils.ObjectFactory import ObjectFactory
from typing import Type

class PretrainedGreedyActiveLearner(GreedyActiveLearner, functionalityType=None):
    def __init__(self, initDataSet, config):
        super().__init__(initDataSet, config)

        self.trainable: Type[Trainable] = self._getTrainable(initDataSet, config)
        self.predictionsTensor = self._getPredictionsTensor(
            config.getSetting('PretrainedModelSavePath')
        )

        self.placeholdersDict = self.trainable.model.placeholders
        self.feedDict = self._getBaseFeedDict()

    def _getTrainable(self, initDataSet, config) -> Type[Trainable]:
        trainableBuilder = ObjectFactory.build(
            BaseTrainableBuilder,
            TrainableType[config.getSetting('TrainableType')],
            dataSet=initDataSet,
            drugDrugTestEdges=self.testEdges,
            config=config,
        )

        return trainableBuilder.build()

    def _getPredictionsTensor(self, pretrainedModelSavePath: str) -> tf.Tensor:
        saver = None
        if isinstance(tf.train.Saver, object):
            saver = tf.train.Saver
        else:
            saver = tf.Train.Saver()

        saver.restore(self.session, pretrainedModelSavePath)

        return self.trainable.optimizer.predictions

    def _getBaseFeedDict(self) -> Dict[str, object]:
        result = {}

        self.trainable.dataSetIterator.update_feed_dict(
            result,
            dropout=0,
            placeholders=self.placeholdersDict
        )

        return result

    def getUpdate(self, dataSet, iterResults):
        # All these things are set in the initializer, so just pass None here
        return super().getUpdate(
            predsTensor=None,
            placeholders=None,
            feedDict=None,
            session=None,
            dataSet=dataSet,
            iterResults=iterResults,
        )

