from .Trainable import Trainable
from tensorflow.python.training import checkpointable

class TensorflowTrainable(Trainable):
    @property
    def checkpointDict(self):
        travObjs = ObjectWalker.walk(
            self,
            filterFxn=TensorflowTrainable._isCheckpointable
        )

        return {
            travObj.name: travObj.obj
            for travObj in travObjs
        }

    @staticmethod
    def _isCheckpointable(obj: object):
        return isinstance(obj, checkpointable.CheckpointableBase)

