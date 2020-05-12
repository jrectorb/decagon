from .DecagonDataSet import DecagonDataSet
from .DecagonTrainable import DecagonTrainable
from ..BaseTrainableBuilder import BaseTrainableBuilder
from ...Dtos.DataSet import DataSet
from ...Dtos.Enums.DataSetType import DataSetType
from ...Dtos.IterationResults import IterationResults
from ...Dtos.Trainable.Trainable import Trainable
from ...Utils.Config import Config
from .decagon.deep.minibatch import EdgeMinibatchIterator
from .decagon.deep.model import DecagonModel
from .decagon.deep.optimizer import DecagonOptimizer
from typing import Type

import tensorflow as tf

class DecagonTrainableBuilder(BaseTrainableBuilder, dataSetType=None):
    def __init__(self, dataSet: DataSet, config: Config) -> None:
        self.dataSet: DecagonDataSet = DecagonDataSet.fromDataSet(dataSet, config)
        self.config: Config = config

    def build(self) -> Type[Trainable]:
        dataSetIterator = self._getDataSetIterator()
        model = self._getModel()
        optimizer = self._getOptimizer(model)

        return DecagonTrainable(
            dataSetIterator,
            optimizer,
            model,
            self.dataSet.placeholdersDict
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
        FEATURE_TPL_MTX_IDX   = 1
        FEATURE_TPL_SHAPE_IDX = 2

        subGraphToFeaturesDimDict = {
            subGraphIdx: featureTpl[FEATURE_TPL_SHAPE_IDX][1]
            for subGraphIdx, featureTpl in self.dataSet.featuresDict.items()
        }

        subGraphToNumNonZeroValsDict = {
            subGraphIdx: int(featureTpl[FEATURE_TPL_MTX_IDX].sum())
            for subGraphIdx, featureTpl in self.dataSet.featuresDict.items()
        }

        return DecagonModel(
            self.dataSet.placeholdersDict,
            subGraphToFeaturesDimDict,
            subGraphToNumNonZeroValsDict,
            self.dataSet.edgeTypeNumMatricesDict,
            self.dataSet.edgeTypeDecoderDict,
        )

    def _getOptimizer(self, model: DecagonModel) -> DecagonOptimizer:
        optimizer = None
        with tf.name_scope('optimizer'):
            optimizer = DecagonOptimizer(
                model.embeddings,
                model.latent_inters,
                model.latent_varies,
                self.dataSet.degreesDict,
                self.dataSet.edgeTypeNumMatricesDict,
                self.dataSet.edgeTypeMatrixDimDict,
                self.dataSet.placeholdersDict,
                margin=self.dataSet.flags.max_margin,
                neg_sample_weights=self.dataSet.flags.neg_sample_size,
                batch_size=self.dataSet.flags.batch_size
            )

        return optimizer

    def getIterationResults(self) -> IterationResults:
        return IterationResults()

class DecagonDummyTrainableBuilder(DecagonTrainableBuilder, dataSetType=DataSetType.DecagonDummyData):
    pass

class DecagonPublicTrainableBuilder(DecagonTrainableBuilder, dataSetType=DataSetType.DecagonPublicData):
    pass

