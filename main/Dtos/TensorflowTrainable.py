from .Trainable import Trainable
from ..Utils.ObjectWalker import ObjectWalker
from tensorflow.python.training import checkpointable

class TensorflowTrainable(Trainable):
    def __init__(self, dataSetIterator, optimizer, model):
        super().__init__(dataSetIterator, optimizer, model)
        self._checkpointDir = None

    @property
    def checkpointDict(self):
        if self._checkpointDir is None:
            travObjs = ObjectWalker.walk(
                self,
                filterFxn=TensorflowTrainable._isCheckpointable,
                ignoreStrs=['checkpointDict']
            )

            self._checkpointDir = {
                travObj.name: travObj.obj
                for travObj in travObjs
            }

        return self._checkpointDir

    @staticmethod
    def _isCheckpointable(obj: object):
        return isinstance(obj, checkpointable.CheckpointableBase)

