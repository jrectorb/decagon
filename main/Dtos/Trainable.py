class Trainable:
    '''
    This class is a DTO containing the objects necessary in order to
    correctly train a model.
    '''

    def __init__(
        self,
        dataSetIterator: DataSetIterator,
        optimizer: Optimizer,
        model: Model
    ) -> None:
        self.dataSetIterator: DataSetIterator = dataSetIterator
        self.optimizer: Optimizer = optimizer
        self.model: Model = model

