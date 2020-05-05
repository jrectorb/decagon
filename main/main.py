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

    trainable: Trainable = _getTrainable(dataSet, config)

    trainer = TrainerFactory.BuildTrainer(trainable, config)
    trainedModel = trainer.Train()

    AccuracyFinder.RecordAccuracy(dataSet, trainedModel)

    return 0

if __name__ == '__main__':
    sys.exit(main())

