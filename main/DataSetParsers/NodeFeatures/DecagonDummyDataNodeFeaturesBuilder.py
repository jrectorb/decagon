from ..Dtos.NodeFeatures import NodeFeatures
import scipy.sparse as sp

class DecagonPublicDataNodeFeaturesBuilder(
    BaseNodeFeaturesBuilder,
    dataSetType = AdjacencyMatricesType.DecagonDummyData
):
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        self.numDrugs: int    = config.getInt('NumDrugs')
        self.numProteins: int = config.getInt('NumProteins')

    def build(self) -> NodeFeatures:
        return NodeFeatures(
            proteinNodeFeatures=sp.identity(self.numProteins).to_coo(),
            drugNodeFeatures=sp.identity(self.numDrugs).to_coo(),
        )

