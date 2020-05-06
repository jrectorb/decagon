from ..Dtos.DataSet import DataSet
from ..Dtos.Trainable import Trainable
from ..BaseTrainableBuilder import BaseTrainableBuilder
from ..Utils.Config import Config

import tensorflow as tf

class DecagonTrainableBuilder(BaseTrainableBuilder):
    def __init__(self, dataSet: DataSet, config: Config) -> None:
        self.dataSet: DecagonDataSet = DecagonDataSet.fromDataSet(dataSet)
        self.config: Config = config

    def build(self) -> Trainable:
        dataSetIterator = self._getDataSetIterator()
        model = self._getModel()
        optimizer = self._getOptimizer(model)

        return DecagonTrainable(
            dataSetIterator,
            optimizer,
            model,
            self.dataSet.placeholders
        )

    def _getDataSetIterator(self) -> EdgeMinibatchIterator:
        return EdgeMinibatchIterator(
            self.dataSet.adjacencyMatrixDict,
            self.dataSet.featuresDict,
            self.dataSet.edgeTypeNumMatricesDict,
            self.dataSet.flags.batch_size,
            float(self.config.getSetting('ValidationSetProportion')),
        )

    def _getModel(self) -> DecagonModel:
        subGraphToFeaturesDimDict = {
            subGraphIdx: featureMtx.shape[1]
            for subGraphIdx, featureMtx in self.dataSet.featuresDict.items()
        }

        subGraphToNumNonZeroValsDict = {
            subGraphIdx: featureMtx.sum()
            for subGraphIdx, featureMtx in self.dataSet.featuresDict.items()
        }

        return DecagonModel(
            self.dataSet.placeholders,
            subGraphToFeaturersDimDict,
            subGraphToNumNonZeroValsDict,
            self.dataSet.edgeTypeNumMatricesDict,
            self.dataSet.edgeTypeDecoderDict,
        )

    def _getOptimizer(self, model: DecagonModel) -> DecagonOptimizer:
        optimizer = None
        with tf.name_scope('optimizer'):
            optimizer = DecagonOptimizer(
                model.embeddings,
                model.latent_interrs,
                model.latent_varies,
                self.dataSet.degreesDict,
                self.dataSet.edgeTypeNumMatricesDict,
                self.dataSet.edgeTypeMatrixDimDict,
                self.dataSet.placeholders,
                self.dataSet.flags.batch_size,
                self.dataSet.flags.max_margin
            )

        return optimizer

