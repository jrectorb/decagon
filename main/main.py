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

def _getTrainer(trainable: Trainable, config: Config) -> Type[Trainer]:
    return ObjectFactory.build(BaseTrainer, trainable, config)

def main() -> int:
    config: Config = _getConfig()
    dataSet: Type[DataSet] = _getDataSet(config)


    activeLearner: Type[BaseActiveLearner] = ObjectFactory.build(
        BaseActiveLearner,
        config
    )

    iterResults: Type[IterationResults] = None
    while activeLearner.hasUpdate(dataSet, iterResults):
        dataSet = activeLearner.getUpdate(dataSet, iterResults)

        trainable: Type[Trainable] = _getTrainable(dataSet, config)
        trainer: Type[BaseTrainer] = ObjectFactory.build(BaseTrainer, trainable, config)

        trainer.train()

        iterResults = trainable.getIterationResults()

    return 0

if __name__ == '__main__':
    sys.exit(main())

