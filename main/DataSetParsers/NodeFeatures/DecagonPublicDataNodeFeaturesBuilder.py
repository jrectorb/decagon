from .BaseNodeFeaturesBuilder import BaseNodeFeaturesBuilder
from ...Dtos.Enums.DataSetType import DataSetType
from ...Dtos.NodeFeatures import NodeFeatures
from ...Dtos.NodeIds import DrugId, SideEffectId
from ...Dtos.NodeLists import NodeLists
from ...Utils import Config
from ...Utils.Sparse import RelationCooMatrix
from collections import defaultdict
from typing import Dict, List, Iterable, Type
import csv
import numpy as np
import scipy.sparse as sp

DrugFeaturesDict = Dict[DrugId, List[SideEffectId]]

class DecagonPublicDataNodeFeaturesBuilder(
    BaseNodeFeaturesBuilder,
    functionalityType = DataSetType.DecagonPublicData
):
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        self.filename: str = config.getSetting('DecagonNodeFeaturesFilename')
        self.drugNodeList: EdgeList = nodeLists.drugNodeList
        self.numProteins: int = len(nodeLists.proteinNodeList)

    def build(self) -> NodeFeatures:
        return NodeFeatures(
            proteinNodeFeatures=self._getProteinNodeFeatures(),
            drugNodeFeatures=self._getDrugNodeFeatures(),
        )

    def _getProteinNodeFeatures(self) -> Type[sp.coo_matrix]:
        return RelationCooMatrix(sp.identity(self.numProteins, format='coo'))

    def _getDrugNodeFeatures(self) -> Type[sp.coo_matrix]:
        drugFeaturesDict = self._getDrugFeaturesDict()

        drugIdToIdx = self._getDrugIdToIdx()
        sideEffectIdToIdx = self._getSideEffectIdToIdx(drugFeaturesDict.values())

        result = np.zeros((len(drugIdToIdx), len(sideEffectIdToIdx)))
        for drugId, drugSideEffects in drugFeaturesDict.items():
            for sideEffectId in drugSideEffects:
                if drugId not in drugIdToIdx:
                    continue

                drugIdx = drugIdToIdx[drugId]
                sideEffectIdx = sideEffectIdToIdx[sideEffectId]

                result[drugIdx, sideEffectIdx] = 1

        return RelationCooMatrix(result)

    def _getDrugFeaturesDict(self) -> DrugFeaturesDict:
        DRUG_ID_IDX     = 0
        SIDE_EFFECT_IDX = 1

        result = defaultdict(list)
        with open(self.filename) as drugFtrsFile:
            reader = csv.reader(drugFtrsFile)

            # Discard header
            next(reader)
            for row in reader:
                drugId = DrugId.fromDecagonFormat(row[DRUG_ID_IDX])
                sideEffectId = SideEffectId.fromDecagonFormat(row[SIDE_EFFECT_IDX])

                result[drugId].append(sideEffectId)

        return result

    def _getDrugIdToIdx(self) -> Dict[DrugId, int]:
        return {drugId: idx for idx, drugId in enumerate(self.drugNodeList)}

    def _getSideEffectIdToIdx(
        self,
        sideEffects: Iterable[List[SideEffectId]]
    ) -> Dict[SideEffectId, int]:
        uniqueSideEffects = np.unique(np.concatenate(tuple(sideEffects)))
        return {sideEffect: idx for idx, sideEffect in enumerate(uniqueSideEffects)}

