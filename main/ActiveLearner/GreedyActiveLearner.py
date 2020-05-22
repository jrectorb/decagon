from .RandomMaskingActiveLearner import RandomMaskingActiveLearner
from ..Dtos.TypeShortcuts import PlaceholdersDict
from ..Trainable.Decagon.DecagonDataSet import DecagonDataSet
from ..Trainable.Decagon.DecagonTrainableBuilder import DecagonTrainableBuilder
from ..Utils import MathUtils
from typing import Dict, Tuple, List

import tensorflow as tf
import numpy as np

RelationCoordinate = Tuple[int, int, int]
EdgeTypeToIdx = Dict[RelationCoordinate, int]

COL_SHAPE_IDX = 1

class GreedyActiveLearner(RandomMaskingActiveLearner, functionalityType=None):
    def __init__(self, initDataSet, config):
        self.session: tf.Session = tf.Session()

        self.decagonDataSet = DecagonDataSet.fromDataSet(initDataSet, config)
        self.edgeTypeToIdx = self._constructEdgeTypeToIdx(self.decagonDataSet, config)

        self.predictionsTensor = None
        self.placeholdersDict = None
        self.feedDict = None

        super().__init__(initDataSet, config)

    def getUpdate(self, predsTensor, placeholders, feedDict, session, dataSet, iterResults):
        if predsTensor is not None:
            self.predictionsTensor = predsTensor

        if placeholders is not None:
            self.placeholdersDict = placeholders

        if feedDict is not None:
            self.feedDict = feedDict

        if session is not None:
            self.session = session

        return super().getUpdate(dataSet, iterResults)

    def _getPredictionsTensor(self, decagonDataSet, config) -> tf.Tensor:
        trainableBuilder = DecagonTrainableBuilder(
            None,
            config,
            decagonDataSet=decagonDataSet
        )

        model = trainableBuilder.getModel()
        optimizer = trainableBuilder.getOptimizer(model)

        return optimizer.predictions

    def _constructEdgeTypeToIdx(self, decagonDataSet, config):
        result = {}
        idx = 0
        for i, j in decagonDataSet.edgeTypeNumMatricesDict.keys():
            for k in range(decagonDataSet.edgeTypeNumMatricesDict[i, j]):
                result[i, j, k] = idx
                idx += 1

        return result

    def _getNewSampleIdxs(self, numToUnmask: int) -> List[Tuple[int, int, int]]:
        # If no iterations happened yet, just pick a random unmask set
        if self.numIters == 0 or self._noRelsExist():
            return super()._getNewSampleIdxs(numToUnmask)

        self._updateFeedDict()
        decoderOutput = self.session.run(self.predictionsTensor, feed_dict=self.feedDict)
        predictions = MathUtils.sigmoid(decoderOutput)

        rankedPossibilities = self._getRankedPossibilities(predictions)
        bestPossibilityIdxs = rankedPossibilities[:numToUnmask]

        return bestPossibilityIdxs

    def _noRelsExist(self):
        return all((
            mtx.sum() == 0
            for mtx in self.decagonDataSet.adjacencyMatrixDict[(1, 1)]
        ))

    def _getRankedPossibilities(self, predictions):
        linearPossibilityIdxs = self._linearizeIndices(
            self.possibilities,
            predictions.shape[COL_SHAPE_IDX]
        )

        possibilityPredictions = np.take(predictions, linearPossibilityIdxs)
        return np.argsort(possibilityPredictions)[::-1]

    def _linearizeIndices(self, indexMtx, numCols):
        return (indexMtx[:, 1] * numCols) + indexMtx[:, 2]

    def _updateFeedDict(self) -> Dict:
        self.feedDict[self.placeholdersDict['dropout']] = 0
        self.feedDict[self.placeholdersDict['batch_edge_type_idx']] = self.edgeTypeToIdx[(1,1,0)]
        self.feedDict[self.placeholdersDict['batch_row_edge_type']] = 1
        self.feedDict[self.placeholdersDict['batch_col_edge_type']] = 1

