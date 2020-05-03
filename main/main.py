from Utils import Config

import sys

def _getConfig() -> Config:
    args = ArgParser().Parse()
    return Config(args)

def _getAdjacencyMatrices(conf: Config) -> AdjacencyMatrices:
    adjacencyMatricesBuilder = AdjacencyMatricesBuilderFactory.buildBuilder(conf)
    return adjacencyMatricesBuilder.build()

def main() -> int:
    config: Config = _getConfig()
    adjacencyMatrices: AdjacencyMatrices = _getAdjacencyMatrices(config)

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

