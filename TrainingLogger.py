import datetime.datetime as datetime

class TrainingIterationRecords:
    @staticmethod
    def GetCSVFileHeader():
        return (
            'Epoch',
            'Iteration',
            'Edge Type',
            'Train Loss',
            'Validation AUROC',
            'Validation AUPRC',
            'Validation AUPK',
            'Iteration Time',
        )

    def __init__(
        self,
        epoch,
        iteration,
        edgeType,
        trainLoss,
        validationAUROC,
        validationAUPRC,
        validationAUPK,
        iterationTime,
    ):
        self.epoch = epoch
        self.iteration = iteration
        self.edgeType = edgeType
        self.trainLoss = trainLoss
        self.validationAUROC = validationAUROC
        self.validationAUPRC = validationAUPRC
        self.validationAUPK = validationAUPK
        self.iterationTime = iterationTime

    def AsTuple(self):
        return (
            self.epoch,
            self.iteration,
            self.edgeType,
            self.trainLoss,
            self.validationAUROC,
            self.validationAUPRC,
            self.validationAUPK,
            self.iterationTime,
        )

    def __str__(self):
        '''
        Epoch: %04d, Iteration: %04d, Edge: %04d, Training Loss: %.5f,
        Validation AUC: %.5f, Validation AUPRC: %.5f, Validation APK: %.5f,
        Time: %.5f
        ''' % (
            self.epoch,
            self.iteration,
            self.edgeType,
            self.trainLoss,
            self.validationAUROC,
            self.validationAUPRC,
            self.validationAUPK,
            self.iterationTime,
        )

class TrainingLogger:
    def __init__(self, iterLogThreshold):
        self._iterLogThreshold

        self._lastLogIter = 0
        self._lastTimestamp = datetime.now()

        self._iterationRecords = []

    def ShouldLog(self, currIter):
        return (
            currIter - self._lastLogIter > self._iterLogThreshold and
            currIter % 4 == 3
        )

    def Log(
        self,
        epoch,
        iteration,
        edgeType,
        trainLoss,
        valAUROC,
        valAUPRC,
        valAUPK
    ):
        iterTime = datetime.now() - self._lastTimestamp
        record = TrainingIterationRecords(
            epoch,
            iteration,
            edgeType,
            trainLoss,
            valAUROC,
            valAUPRC,
            valAUPK,
            iterTime
        )

        self._iterationRecords.append(record)
        self._logToFileSystem(record)
        self._logToStdOut(record)

    def _logToFileSystem(self, record):
        # idk just use csv logger

    def _logToStdOut(self, record):
        print(record)

