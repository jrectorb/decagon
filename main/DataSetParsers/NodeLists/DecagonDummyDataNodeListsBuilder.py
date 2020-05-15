from .BaseNodeListsBuilder import BaseNodeListsBuilder
from ...Dtos.Enums.DataSetType import DataSetType
from ...Dtos.NodeLists import NodeLists
from ...Dtos.NodeIds import ProteinId, DrugId
from ...Utils import Config

class DecagonDummyDataNodeListsBuilder(
    BaseNodeListsBuilder,
    functionalityType = DataSetType.DecagonDummyData
):
    def __init__(self, config: Config) -> None:
        self.numProteins: int = config.getSetting('NumProteins')
        self.numDrugs: int    = config.getSetting('NumDrugs')

    def build(self) -> NodeLists:
        proteinNodeList = [ProteinId(i) for i in range(self.numProteins)]
        drugNodeList = [DrugId(i) for i in range(self.numDrugs)]

        return NodeLists(proteinNodeList, drugNodeList)

