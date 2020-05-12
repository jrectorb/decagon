from .BaseCheckpointer import BaseCheckpointer
from ..Dtos.Trainable.TensorflowTrainable import TensorflowTrainable
from ..Utils.Config import Config
import tensorflow.contrib.eager as tfe

class TensorflowCheckpointer(BaseCheckpointer):
    def __init__(self, trainable: TensorflowTrainable, config: Config):
        super().__init__(config)

        self.checkpoint: tf.train.Checkpoint = \
            tfe.Checkpoint(**trainable.checkpointDict)

        self.ckptDir: str = config.getSetting('CheckpointDirectory')

        # TODO: Implement only keeping k checkpoints
        #self.maxToKeep: int = int(config.getSetting('MaxCheckpointsToKeep'))

        customName = config.getSetting('CustomCheckpointName')
        self.ckptBaseName: str = 'ckpt'
        if customName:
            self.ckptBaseName = self.ckptBaseName + "_%s" % customName

    def _getCkptName(self, isNew: bool) -> str:
        thisFileIdx = self._getCurrFnameIdx()
        if thisFileIdx == -1:
            thisFileIdx = 0
        else:
            # If not the newest file, increment the ckpt idx
            thisFileIdx += 1

        return self.ckptBaseName + '_%d' % thisFileIdx

    def _getCurrFnameIdx(self) -> int:
        existingIndices = [
            self._getFnameIdx(f)
            for f in os.listdir(baseDir)
            if self._isValidFname(baseDir, f)
        ]

        if len(existingIndices) > 0:
            thisFileIdx = max(thisFileIdx)
        else:
            return -1

    def _getFnameIdx(self, fname: str, matchStr: str) -> int:
        fnameFirstPortion = fname.split('.')[0]
        lastUscoreIdx = fnameFirstPortion.rfind('_')

        return int(fnameFirstPortion[lastUscoreIdx + 1:])

    def _isValidFname(self, baseDir: str, fname: str, matchStr: str) -> bool:
        isFile = os.isfile(baseDir, fname)

        # From "thiscoolfile.txt", extract "thiscoolfile"
        fnameFirstPortion = fname.split('.')[0]

        lastUscoreIdx = fnameFirstPortion.rfind('_')
        isGoodPrefix = fnameFirstPortion[:lastUscoreIdx] == matchStr

        return isFile and isGoodPrefix

    def save(self):
        newFname = self._getCkptName(isNew=True)
        self.checkpointManager.save(newFname)

    def restore(self):
        someCkptExists = self._getCurrFnameIdx() != -1
        currFname = self._getCkptName(isNew=False)

        self.checkpoint.restore(self.ckptDir + currFname)

