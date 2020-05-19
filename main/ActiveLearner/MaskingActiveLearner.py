from .BaseActiveLearner import BaseActiveLearner
from ..Dtos.AdjacencyMatrices import AdjacencyMatrices
from ..Dtos.DataSet import DataSet
from ..Dtos.Enums.ActiveLearnerType import ActiveLearnerType
from ..Utils.Config import Config
from ..Utils.Sparse import RelationCsrMatrix
from operator import itemgetter
import numpy as np

class RandomMaskingActiveLearner(
    BaseActiveLearner,
    functionalityType=ActiveLearnerType.RandomMaskingActiveLearner
):
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

        self.adjMtxMasks = {
            rel: np.zeros(mtx.shape)
            for rel, mtx in initDataSet.adjacencyMatrices.drugDrugRelationMtxs.items()
        }

        self.possibilities = self._getPossibilities(self.adjMtxMasks)
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

            graphTypeArr = np.full((grid.shape[0], 1), int(rel))

            preResult.append(np.hstack((graphTypeArr, grid)))

        return np.vstack(preResult) if len(preResult) > 0 else np.empty((0, 0))

    def hasUpdate(self, dataset, iterResults) -> bool:
        return len(self.possibilities) > 0

    def getUpdate(self, dataSet, iterResults) -> DataSet:
        self._updateMask()

        idStr = "RandomPolicy%sIter%d" % (self.initDataSet.id, self.numIters)
        adjMtxs = self._applyMask()

        self.numIters += 1

        return DataSet(
            idStr=idStr,
            adjacencyMatrices=adjMtxs,
            nodeFeatures=dataSet.nodeFeatures
        )

    def _updateMask(self):
        multiplier = self.initUnmaskedProportion if self.numIters == 0 \
                     else self.proportionUnmaskedPerIter

        numToUnmask = int(np.floor(self.dataSetSize * multiplier))

        samples = np.random.choice(
            len(self.possibilities),
            size=numToUnmask,
            replace=False
        )

        idxsToUnmask = itemgetter(*samples)(self.possibilities)

        for idx in idxsToUnmask:
            self.adjMtxMasks[idx[0]][idx[1], idx[2]] = 1

        self.possibilities = np.delete(self.possibilities, samples, axis=0)

        return

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

