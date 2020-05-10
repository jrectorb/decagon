from ..Dtos.AdjacencyMatrices import AdjacencyMatrices
from ..Dtos.DataSet import DataSet
from ..Dtos.NodeFeatures import NodeFeatures
from ..Dtos.NodeLists import NodeLists
from ..Dtos.Enums.DataSetType import DataSetType
from ..Utils.Config import Config

class DataSetBuilder:
    @staticmethod
    def build(config: Config) -> DataSet:
        dataSetType = DataSetType(config.getSetting('DataSetType'))

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

        return DataSet(adjacencyMatrices, nodeFeatures)

    @staticmethod
    def _getNodeLists(dataSetType: DataSetType, config: Config) -> NodeLists:
        nodeListsBuilder = ObjectFactory.build(
            BaseNodeListsBuilder,
            dataSetType,
            config
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
            nodeLists,
            dataSetType,
            conf,
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
            nodeLists,
            dataSetType,
            conf,
        )

        return godeFeaturesBuilder.build()

