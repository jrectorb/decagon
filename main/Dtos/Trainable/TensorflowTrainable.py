from .Trainable import Trainable
import tensorflow.python.eager.def_function as tfDefFxn
import tensorflow.python.training.tracking.base as tfBase

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
        return isinstance(obj, (tfBase.Trackable, tfDefFxn.Function))

