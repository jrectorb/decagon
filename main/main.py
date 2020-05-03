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

    nodeLists: NodeLists = _getNodeLists(config)
    adjacencyMatrices: AdjacencyMatrices = _getAdjacencyMatrices(nodeLists, config)

    model = _getModel(adjacencyMatrices, config)
    optimizer = _getOptimizer(adjacencyMatrices, config)
    batchIterator = _getBatchIterator(adjacencyMatrices, config)

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

