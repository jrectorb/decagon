from .BaseActiveLearner import BaseActiveLearner
from ..Dtos.AdjacencyMatrices import AdjacencyMatrices
from ..Dtos.DataSet import DataSet
from ..Dtos.Enums.ActiveLearnerType import ActiveLearnerType
from ..Dtos.TestEdgesContainer import TestEdgesContainer
from ..Utils.Config import Config
from ..Utils.Sparse import RelationCsrMatrix
from typing import List, Tuple
from operator import itemgetter
import numpy as np
import scipy.sparse as sp

class RandomMaskingActiveLearner(
    BaseActiveLearner,
    functionalityType=ActiveLearnerType.RandomMaskingActiveLearner
):
    def __init__(self, initDataSet, config: Config) -> None:
        self.numIters: int = 0
        self.testSetProportion = float(config.getSetting('TestSetProportion'))
        self.initTrainSetProportion = float(config.getSetting('InitTrainSetProportion'))
        self.initDataSet = initDataSet

        self.adjMtxMasks = {
            rel: np.zeros(mtx.shape)
            for rel, mtx in initDataSet.adjacencyMatrices.drugDrugRelationMtxs.items()
        }

        self.possibilities, self.testEdges = self._getPossibilitiesAndTestEdges(
            self.adjMtxMasks
        )

        self.dataSetSize = len(self.possibilities)

    @property
    def _attrNameToIdx(self):
        return {
            'drugDrugRelationMtxs': 0,
            'drugProteinRelationMtx': 1,
            'proteinProteinRelationMtx': 2
        }

    @property
    def _idxToAttrName(self):
        return {v: k for k, v in self._attrNameToIdx.items()}

    def _getPossibilitiesAndTestEdges(self, adjMtxMasks):
        prePossbilitiesResult = []
        testEdgeResult = {}
        for rel, mtx in self.adjMtxMasks.items():
            if not self._isRelationValid(rel):
                continue

            grid = self._getShapePossibleIdxs(mtx.shape)

            posTestEdgeIdxs, negTestEdgeIdxs = self._getTestEdgeLinearIdxs(
                self.initDataSet.adjacencyMatrices.drugDrugRelationMtxs[rel],
                grid
            )

            testEdgeResult[rel] = {
                'positive': grid[posTestEdgeIdxs],
                'negative': grid[negTestEdgeIdxs],
            }

            grid = np.delete(grid, np.hstack([posTestEdgeIdxs, negTestEdgeIdxs]), axis=0)

            graphTypeArr = np.full((grid.shape[0], 1), int(rel))
            prePossbilitiesResult.append(np.hstack((graphTypeArr, grid)))

        possibilities = np.vstack(prePossbilitiesResult) \
                        if len(prePossbilitiesResult) > 0 else np.empty((0, 0, 0))

        possibilities = self._reducePossibilitiesForInit(possibilities)
        return possibilities, testEdgeResult

    def _isRelationValid(self, relation: str) -> bool:
        return True

    def _reducePossibilitiesForInit(self, possibilities) -> np.array:
        numToUnmask = int(
            np.floor(len(possibilities) * self.initTrainSetProportion)
        )

        idxsToUnmask = \
            np.random.choice(len(possibilities), size=numToUnmask, replace=False)

        for idx in possibilities[idxsToUnmask]:
            self.adjMtxMasks[idx[0]][idx[1], idx[2]] = 1

        return np.delete(possibilities, idxsToUnmask, axis=0)

    def _getTestEdgeLinearIdxs(self, mtx, possibilities):
        allPosEdges = self._getPossiblePositiveEdgeIdxs(mtx)
        allNegEdges = self._getPossibleNegativeEdgeIdxs(
            possibilities,
            allPosEdges,
            mtx.shape
        )

        numEdges = max(1, int(allPosEdges.shape[0] * self.testSetProportion)) \
                   if allPosEdges.shape[0] > 0 else 0
        linearizedPosTestEdgeIdxs = self._sampleIndicesLinear(
            allPosEdges,
            numEdges,
            mtx.shape[1]
        )

        linearizedNegTestEdgeIdxs = self._sampleIndicesLinear(
            allNegEdges,
            numEdges,
            mtx.shape[1]
        )

        return linearizedPosTestEdgeIdxs, linearizedNegTestEdgeIdxs

    def _sampleIndicesLinear(self, possibleEdges, numToSample, fullSetColDim):
        edgeSetIdxs = np.random.choice(
            possibleEdges.shape[0],
            size=numToSample,
            replace=False
        )

        edges = possibleEdges[edgeSetIdxs]

        return (edges[:, 0] * fullSetColDim) + edges[:, 1]

    def _getPossiblePositiveEdgeIdxs(self, mtx: sp.csr_matrix) -> np.array:
        nonzeroTpl = mtx.nonzero()
        return np.dstack([nonzeroTpl[0], nonzeroTpl[1]]).reshape(-1, 2)

    def _getPossibleNegativeEdgeIdxs(
        self,
        possibilities: np.array,
        positiveEdges: np.array,
        mtxShape: Tuple[int, int]
    ) -> np.array:
        possibilitiesLinear = (possibilities[:, 0] * mtxShape[1]) + possibilities[:, 1]
        positivesLinear = (positiveEdges[:, 0] * mtxShape[1]) + positiveEdges[:, 1]

        negativesLinear = np.setdiff1d(possibilitiesLinear, positivesLinear)

        return np.dstack(np.unravel_index(negativesLinear, mtxShape)).reshape(-1, 2)

    def _getShapePossibleIdxs(self, shape: Tuple[int, int]) -> np.array:
        xx, yy = np.indices(shape)
        return np.dstack([xx, yy]).reshape(-1, 2)

    def hasUpdate(self, dataset, iterResults) -> bool:
        return 2 ** self.numIters < 100

    def getUpdate(self, dataSet, iterResults) -> DataSet:
        self._updateMask()

        idStr = "%s%sIter%d" % (type(self).__name__, self.initDataSet.id, self.numIters)
        adjMtxs = self._applyMask()

        self.numIters += 1

        return DataSet(
            idStr=idStr,
            nodeLists=dataSet.nodeLists,
            adjacencyMatrices=adjMtxs,
            nodeFeatures=dataSet.nodeFeatures
        )

    def _updateMask(self):
        lastNumerator = 2 ** (self.numIters - 1) if self.numIters > 0 else 0
        thisNumerator = min(2 ** self.numIters, 100)

        multiplier = (thisNumerator - lastNumerator) / 100
        numToUnmask = int(np.floor(self.dataSetSize * multiplier))

        idxsToUnmask = self._getNewSampleIdxs(numToUnmask)
        for idx in self.possibilities[idxsToUnmask]:
            self.adjMtxMasks[idx[0]][idx[1], idx[2]] = 1

        self.possibilities = np.delete(self.possibilities, idxsToUnmask, axis=0)

        return

    def _getNewSampleIdxs(self, numToUnmask: int):
        return np.random.choice(
            len(self.possibilities),
            size=numToUnmask,
            replace=False
        )

    def _applyMask(self):
        drugDrugRelationMtxs = {}
        for relId, maskMtx in self.adjMtxMasks.items():
            drugDrugRelationMtxs[relId] = RelationCsrMatrix(np.multiply(
                maskMtx,
                self.initDataSet.adjacencyMatrices.drugDrugRelationMtxs[relId].toarray()
            ))

        return AdjacencyMatrices(
            drugDrugRelationMtxs,
            self.initDataSet.adjacencyMatrices.drugProteinRelationMtx,
            self.initDataSet.adjacencyMatrices.proteinProteinRelationMtx
        )

