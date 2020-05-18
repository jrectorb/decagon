from ..Dtos.AdjacencyMatrices import AdjacencyMatrices
from ..Dtos.DataSet import DataSet
from ..Dtos.NodeFeatures import NodeFeatures
from ..Dtos.NodeLists import NodeLists
from ..Dtos.Enums.DataSetType import DataSetType
from .AdjacencyMatrices.BaseAdjacencyMatricesBuilder import BaseAdjacencyMatricesBuilder
from .NodeFeatures.BaseNodeFeaturesBuilder import BaseNodeFeaturesBuilder
from .NodeLists.BaseNodeListsBuilder import BaseNodeListsBuilder
from ..Utils.Config import Config
from ..Utils.ObjectFactory import ObjectFactory

class DataSetBuilder:
    @staticmethod
    def build(config: Config) -> DataSet:
        dataSetType = DataSetType[config.getSetting('DataSetType')]

        idStr = "Base%sDataSet" % dataSetType.name

        nodeLists = DataSetBuilder._getNodeLists(dataSetType, config)

        adjacencyMatrices = DataSetBuilder._getAdjacencyMatrices(
            nodeLists,
            dataSetType,
            config,
        )

        nodeFeatures = DataSetBuilder._getNodeFeatures(
            nodeLists,
            dataSetType,
            config,
        )

        return DataSet(idStr, adjacencyMatrices, nodeFeatures)

    @staticmethod
    def _getNodeLists(dataSetType: DataSetType, config: Config) -> NodeLists:
        nodeListsBuilder = ObjectFactory.build(
            BaseNodeListsBuilder,
            functionalityType=dataSetType,
            config=config
        )

        return nodeListsBuilder.build()

    @staticmethod
    def _getAdjacencyMatrices(
        nodeLists: NodeLists,
        dataSetType: DataSetType,
        config: Config
    ) -> AdjacencyMatrices:
        adjacencyMatricesBuilder = ObjectFactory.build(
            BaseAdjacencyMatricesBuilder,
            functionalityType=dataSetType,
            nodeLists=nodeLists,
            config=config
        )

        return adjacencyMatricesBuilder.build()

    @staticmethod
    def _getNodeFeatures(
        nodeLists: NodeLists,
        dataSetType: DataSetType,
        config: Config
    ) -> AdjacencyMatrices:
        nodeFeaturesBuilder = ObjectFactory.build(
            BaseNodeFeaturesBuilder,
            functionalityType=dataSetType,
            nodeLists=nodeLists,
            config=config
        )

        return nodeFeaturesBuilder.build()

