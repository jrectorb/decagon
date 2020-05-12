from ..IterationResults import IterationResults

class Trainable:
    '''
    This class is a DTO containing the objects necessary in order to
    correctly train a model.
    '''

    def __init__(self, dataSetIterator, optimizer, model):
        self.dataSetIterator = dataSetIterator
        self.optimizer = optimizer
        self.model = model

    def getIterationResults(self) -> IterationResults:
        pass

