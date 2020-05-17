from ..Utils.Config import Config
from operator import itemgetter
import numpy as np

class RandomMaskingActiveLearner(BaseActiveLearner, functionalityType=None):
    def __init__(self, initDataSet, config: Config) -> None:
        self.numIters: int = 0

        self.initUnmaskedProportion = float(
            config.getSetting('InitialUnmaskedProportion')
        )

        self.proportionUnmaskedPerIter = float(
            config.getSetting('ProportionUnmaskedPerIteration')
        )

        self.currAmountUnmasked = self.initUnmaskedProportion

        self.initDataSet = initDataSet
        self.relIdToIntId = {
            i: relId
            for i, relId in initDataSet.adjacencyMatrices.drugDrugRelationMtxs.keys()
        }

        self.adjMtxMasks = {
            rel: np.zeros(mtx.shape)
            for rel, mtx in initDataSet.adjacencyMatrices.drugDrugRelationMtxs.items()
        }

        self.possibilities = self._getIndices(self.adjMtxMasks)
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

    def _getPossibilities(self, adjMtxMasks):
        preResult = []
        for rel, mtx in self.adjMtxMasks.items():
            xx, yy = np.indices(mtx.shape)
            grid = np.dstack([xx, yy]).reshape(-1, 2)

            graphTypeArr = np.full((grid.shape[0], 1), self.relIdToIntId[rel])

            preResult.append(np.hstack(graphTypeArr, grid))

        return np.vstack(preResult).tolist()

    def hasUpdate(self, dataset, iterResults) -> bool:
        return self.currAmountUnmasked < 1. and not np.isclose(1., self.currAmountUnmasked)

    def getUpdate(self, dataSet, iterResults) -> DataSet:
        self._updateMask()

        return DataSet(
            adjacencyMatrices=self._applyMask(),
            nodeFeatures=dataSet.nodeFeatures
        )

    def _updateMask(self):
        multiplier = self.initUnmaskedProportion if self.numItes == 0 \
                     else self.proportionUnmaskedPerIter

        numToUnmask = int(np.floor(self.dataSetSize * multiplier))

        samples = np.random.choice(0, len(numToUnmask), size=(numToUnmask,))
        idxsToUnmask = itemgetter(*samples)(self.possibilities)

        for idx in idxsToUnmask:
            self.adjMtxMasks[idx[0]][idx[2]][idx[3], idx[4]] = 1

        for sampleIdx in samples:
            self.possibilities.pop(sampleIdx)

        return

    def _applyMask(self):
        newAdjMtxs = AdjacencyMatrices()

        newAdjMtxs.drugProteinRelationMtx = \
            self.adjMtxMasks[1][0] * self.initDataSet.adjacencyMatrices.drugProteinRelationMtx

        newAdjMtxs.drugProteinRelationMtx = \
            self.adjMtxMasks[2][0] * self.initDataSet.adjacencyMatrices.proteinProteinRelationMtx

        newAdjMtxs.drugDrugRelationMtxs = {}
        for relIdInt, maskMtx in self.adjMtxMasks.items():
            relId = self.relIdToIntId[relIdInt]

            newAdjMtxs.drugDrugRelationMtxs[relId] = \
                self.initDataSet.adjacencyMatrices.drugDrugRelationMtxs[relId] * maskMtx

        return newAdjMtxs

