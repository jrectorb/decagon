from .DecagonDataSet import DecagonDataSet
from .DecagonTrainable import DecagonTrainable
from ..BaseTrainableBuilder import BaseTrainableBuilder
from ...Dtos.DataSet import DataSet
from ...Dtos.Enums.TrainableType import TrainableType
from ...Dtos.IterationResults import IterationResults
from ...Dtos.Trainable import Trainable
from ...Utils.Config import Config
from .decagon.deep.minibatch import EdgeMinibatchIterator
from .decagon.deep.model import DecagonModel
from .decagon.deep.optimizer import DecagonOptimizer
from typing import Type, Dict

import tensorflow as tf
import numpy as np

class DecagonTrainableBuilder(
    BaseTrainableBuilder,
    functionalityType=TrainableType.DecagonTrainable
):
    def __init__(
        self,
        dataSet: DataSet,
        drugDrugTestEdges: Dict[int, Dict[str, np.array]],
        config: Config,
        decagonDataSet: DecagonDataSet = None,
        placeholdersDict=None
    ) -> None:
        self.dataSet: DecagonDataSet = None
        if decagonDataSet is not None:
            self.dataSet = decagonDataSet
        else:
            self.dataSet = DecagonDataSet.fromDataSet(dataSet, config)

        self.placeholdersDict = \
            placeholdersDict if placeholdersDict is not None else self.dataSet.placeholdersDict

        self.drugDrugTestEdges: Dict[int, Dict[str, np.array]] = drugDrugTestEdges
        self.config: Config = config

    def build(self) -> Type[Trainable]:
        dataSetIterator = self.getDataSetIterator()
        model = self.getModel()
        optimizer = self.getOptimizer(model)

        return DecagonTrainable(
            dataSetIterator,
            optimizer,
            model,
            self.dataSet.placeholdersDict
        )

    def getDataSetIterator(self) -> EdgeMinibatchIterator:
        return EdgeMinibatchIterator(
            self.dataSet.adjacencyMatrixDict,
            self.dataSet.featuresDict,
            self.dataSet.edgeTypeNumMatricesDict,
            self.drugDrugTestEdges,
            self.dataSet.flags.batch_size,
            float(self.config.getSetting('TestSetProportion'))
        )

    def getModel(self) -> DecagonModel:
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

    def getOptimizer(self, model: DecagonModel) -> DecagonOptimizer:
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

