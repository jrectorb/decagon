class DataSetBuilder:
    @staticmethod
    def build(config: Config) -> DataSet:
        dataSetType = DataSetTypee(config.getSetting('DataSetType'))

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
        nodeListsBuilder = NodeListsBuilderFactory.buildBuilder(dataSetType, config)
        return nodeListsBuilder.build()

    @staticmethod
    def _getAdjacencyMatrices(
        nodeLists: NodeLists,
        dataSetType: DataSetType,
        config: Config
    ) -> AdjacencyMatrices:
        adjacencyMatricesBuilder = AdjacencyMatricesBuilderFactory.buildBuilder(
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
        nodeFeaturesBuilder = NodeFeaturesBuilderFactory.buildBuilder(
            nodeLists,
            dataSetType,
            conf,
        )

        return godeFeaturesBuilder.build()

