from Utils import Config

import sys

def _getConfig() -> Config:
    args = ArgParser().Parse()
    return Config(args)

def _getNodeLists(conf: Config):
    nodeListsBuilder = NodeListsBuilderFactory.buildBuilder(conf)
    return nodeListsBuilder.build()

def _getAdjacencyMatrices(nodeLists: NodeLists, conf: Config) -> AdjacencyMatrices:
    adjacencyMatricesBuilder = AdjacencyMatricesBuilderFactory.buildBuilder(
        nodeLists,
        conf,
    )

    return adjacencyMatricesBuilder.build()

def main() -> int:
    config: Config = _getConfig()
    dataSet: DataSet = _getDataSet(config)

    model = _getModel(dataSet, config)
    optimizer = _getOptimizer(dataSet, config)
    batchIterator = _getBatchIterator(dataSet, config)

    trainer = TrainerFactory.BuildTrainer(
        conf,
        model,
        optimizer,
        batchIterator
    )

    trainedModel = trainer.Train()

    AccuracyFinder.RecordAccuracy(adjMtxInfos, trainedModel)

    return 0

if __name__ == '__main__':
    sys.exit(main())

