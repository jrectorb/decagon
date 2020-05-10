from .BaseNodeFeaturesBuilder import BaseNodeFeaturesBuilder
from ...Dtos.Enums.DataSetType import DataSetType
from ...Dtos.NodeFeatures import NodeFeatures
from ...Dtos.NodeLists import NodeLists
from ...Utils import Config
import scipy.sparse as sp

class DecagonDummyDataNodeFeaturesBuilder(
    BaseNodeFeaturesBuilder,
    dataSetType = DataSetType.DecagonDummyData
):
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        self.numDrugs: int    = int(config.getSetting('NumDrugs'))
        self.numProteins: int = int(config.getSetting('NumProteins'))

    def build(self) -> NodeFeatures:
        return NodeFeatures(
            proteinNodeFeatures=sp.identity(self.numProteins).tocoo(),
            drugNodeFeatures=sp.identity(self.numDrugs).tocoo(),
        )

