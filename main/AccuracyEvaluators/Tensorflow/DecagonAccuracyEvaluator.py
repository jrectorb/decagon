from ...Dtos.TypeShortcuts import EdgeType, PlaceholdersDict
import tensorflow as tf
import numpy as np

# A 2-tuple representing an indexing into an M x N matrix
Coordinate = tuple
EdgeSamples = Dict[EdgeType, List[Coordinate]]

FROM_GRAPH_IDX = 0
TO_GRAPH_IDX   = 1

ROW_SHAPE_IDX  = 0
COL_SHAPE_IDX  = 1

class LossElementsContainer:
    def __init__(self, predictions: np.ndarray, labels: np.ndarraay):
        self.predictions: np.ndarray = predictions
        self.labels: np.ndarray = labels

class DecagonAccuracyEvaluator:
    def __init__(
        self,
        session: tf.Session,
        placeholdersDict: PlaceholdersDict,
        predictionsTensor: tf.Tensor,
        edgeTypeToIdx: EdgeTypeToIdx
    ) -> None:
        self.session: tf.Session = session
        self.placeholdersDict: PlaceholdersDict = placeholdersDict
        self.predictionsTensor: tf.Tensor  = predictionsTensor
        self.edgeTypeToIdx : EdgeTypeToIdx = edgeTypeToIdx

    def evaluate(
        self,
        feedDict: Dict,
        edgeType: EdgeType,
        positiveEdgeSamples: EdgeSamples,
        negativeEdgeSamples: EdgeSamples,
    ) -> AccuracyScores:
        self._updateFeedDictForEval(feedDict, edgeType)

        lossElements: LossElementsContainer = \
            self._computePredictions(feedDict, edgeType)

        auroc = metrics.roc_auc_score(lossElements.labels, lossElements.predictions)
        auprc = metrics.average_precision_score(lossElements.labels, lossElements.predictions)
        apk   = self._computeApk(lossElements)

        return AccuracyScores(auroc, auprc, apk)

    def _computePredictions(
        self,
        feedDict: Dict,
        edgeType: EdgeType,
        positiveEdgeSamples: EdgeSamples,
        negativeEdgeSamples: EdgeSamples,
    ) -> LossElementsContainer:
        decoderOutput = self.session.run(self.predictionsTensor, feed_dict=feedDict)
        predictions = MathUtils.sigmoid(decoderOutput)

        positiveSamplePredictions = self._getSampledPredictions(
            predictions,
            positiveEdgeSamples,
            edgeType
        )

        negativeSamplePredictions = self._getSampledPredictions(
            predictions,
            negativeEdgeSamples,
            edgeType
        )

        sampledPredictions = np.hstack(
            [positiveSamplePredictions, negativeSamplePredictions]
        )

        sampledLabels = np.hstack([
            np.ones(len(positiveSamplePredictions)),
            np.zeros(len(negativeSamplePredictions))
        ])

        return LossElementsContainer(
            predictions=sampledPredictions,
            labels=sampledLabels
        )

    def _getSampledPredictions(
        self,
        predictions: np.ndarray,
        edgeSamples: EdgeSamples,
        edgeType: EdgeType
    ) -> np.ndarray:
        linearizedSampleIdxs = self._linearizeSampleIdxs(
            edgeSamples,
            edgeType,
            predictions.shape[COL_SHAPE_IDX]
        )

        # np.take here is equivalent to predictions.ravel()[linearizedSampleIdxs]
        return np.take(predictions, linearizedSampleIdxs)

    def _linearizeSampleIdxs(
        self,
        edgeSamples: EdgeSamples,
        edgeType: EdgeType,
        numCols: int
    ) -> np.ndarray:
        # edgeType is a 3-tuple wherein the first two indices represent the
        # subgraph type, while the last index represents the kth adjacency
        # matrix for that subgraph
        subGraphType = edgeType[:2]
        relationType = edgeType[2]

        twoDimSampleIndexes = edgeSamples[subGraphType][relationType]

        return (twoDimSampleIndexes[:, 0] * numCols) + twoDimSampleIndexes[:, 1]

    def _updateFeedDictForEval(self, feedDict: Dict, edgeType: EdgeType) -> None:
        feedDict[placeholders['dropout']] = 0
        feedDict[placeholders['batch_edge_type_idx']] = self.edgeTypeToIdx[edgeType]
        feedDict[placeholders['batch_row_edge_type']] = edgeType[FROM_GRAPH_IDX]
        feedDict[placeholders['batch_col_edge_type']] = edgeType[TO_GRAPH_IDX]

        return

    @staticmethod
    def _toLinearIdxs(twoDimIdxs, predTensorDim) -> np.ndarray:
        return (twoDimIdxs[:,0] * predTensorDim) + idxs[:,1]

