import sys

def _getConfig():
    args = ArgParser().Parse()
    return Config(args)

def _getAdjMtxInfos(conf):
    adjMtxInfosBuilder = AdjacencyMatrixInformationBuilderFactory.BuildBuilder(conf)
    return adjMtxInfosBuilder.Build()

def main():
    conf = _getConfig()
    adjMtxInfos = _getAdjMtxInfos(conf)

    model = _getModel(adjMtxInfos, conf)
    optimizer = _getOptimizer(adjMtxInfos, conf)
    trainingSetIterator = _getTrainingSetItertaor(adjMtxInfos, conf)

    trainer = TrainerFactory.BuildTrainer(
        conf,
        model,
        optimizer,
        trainingSetIterator
    )

    trainedModel = trainer.Train()

    AccuracyFinder.RecordAccuracy(adjMtxInfos, trainedModel)

    return 0

if __name__ == '__main__':
    sys.exit(main())
