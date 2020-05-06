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
    dataSet: DataSet = _getDataSet(config)

    trainable: Trainable = _getTrainable(dataSet, config)

    trainer: Type[BaseTrainer] = ObjectFactory.build(BaseTrainer, trainable, config)
    trainer.train()

    return 0

if __name__ == '__main__':
    sys.exit(main())

