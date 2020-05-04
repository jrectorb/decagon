from ..Utils import Config
from collections import defaultdict
import csv

DrugFeaturesDict = Dict[DrugId, List[SideEffectId]]

class DecagonPublicDataNodeFeaturesBuilder(
    BaseNodeFeaturesBuilder,
    dataSetType = DataSetType.DecagonPublicData
):
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        self.filename: str = config.getSetting('NodeFeaturesFilename')
        self.drugNodeList: EdgeList = nodeLists.drugNodeList
        self.numProteins: int = len(nodeLists.proteinNodeList)

    def build(self) -> NodeFeatures:
        return NodeFeatures(
            proteinNodeFeatures=self._getProteinNodeFeatures(),
            drugNodeFeatures=self._getDrugNodeFeatures(),
        )

    def _getProteinNodeFeatures(self) -> sp.coo_matrix:
        return sp.identity(self.numProteins, format=sp.coo_matrix)

    def _getProteinNodeFeatures(self) -> np.array:
        drugFeaturesDict = self._getDrugFeaturesDict()

        drugIdToIdx = self._getDrugIdToIdx(drugFeaturesDict.Keys())
        sideEffectIdToIdx = self._getSideEffectIdToIdx(drugFeaturesDict.Values())

        result = np.zeros((len(drugIdToIdx), len(sideEffectIdToIdx)))
        for drugId, drugSideEffects in drugFeaturesDict:
            for sideEffectId in drugSideEffects:
                if drugId not in drugIdToIdx:
                    continue

                drugIdx = drugIdToIdx[drugId]
                sideEffectIdx = sideEffectIdToIdx[sideEffectId]

                result[drugIdx, sideEffectIdx] = 1

        return result

    def _getDrugFeaturesDict(self) -> DrugFeaturesDict:
        DRUG_ID_IDX     = 0
        SIDE_EFFECT_IDX = 1

        result = defaultdict(list)
        with open(self.filename) as drugFtrsFile:
            reader = csv.reader(drugFtsFile)
            for row in reader:
                drugId = DrugId.fromDecagonFormat(row[DRUG_ID_IDX])
                sideEffectId = SideEffectId.fromDecagonFormat(row[SIDE_EFFECT_ID])

                result[drugId].append(sideEffectId)

        return result

    def _getDrugIdToIdx(self) -> Dict[DrugId, int]:
        return {drugId: i for i, drugId in enumerate(self.drugNodeList)}

    def _getSideEffectIdToIdx(
        self,
        sideEffects: dict_values
    ) -> Dict[SideEffectId, int]:
        uniqueSideEffects = np.unique(np.concatenate(sideEffects))
        return {sideEffect: i for i, sideEffect in enumerate(uniqueSideEffects)}

