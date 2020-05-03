from ..Dtos.NodeLists import NodeLists

class DecagonPublicDataAdjacencyMatricesBuilder(
    BaseNodeListsBuilder,
    dataSetType = DataSetType.DecagonDummyData
):
    def __init__(self, config: Config) -> None:
        self.numProteins: int = config.getSetting('NumProteins')
        self.numDrugs: int    = config.getSetting('NumDrugs')

    def build(self) -> AdjacencyMatrices:
        proteinNodeList = [ProteinId(i) for i in range(self.numProteins)]
        drugNodeList = [DrugId(i) for i in range(self.numProteins)]

        return NodeLists(proteinNodeList, drugNodeList)

