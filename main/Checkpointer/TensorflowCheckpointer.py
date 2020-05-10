from .BaseCheckpointer import BaseCheckpointer
from ..Dtos.Trainable.TensorflowTrainable import TensorflowTrainable
from ..Utils.Config import Config

class TensorflowCheckpointer(BaseCheckpointer):
    def __init__(self, trainable: TensorflowTrainable, config: Config):
        super().__init__(config)

        self.checkpoint: tf.train.Checkpoint = \
            tf.train.Checkpoint(trainable.checkpointDict)

        ckptDir = config.getSetting('CheckpointDirectory')
        ckptName = self._getCkptName(ckptDir, config)
        maxToKeep = int(config.getSetting('MaxCheckpointsToKeep'))

        self.checkpointManager: tf.train.CheckpointManager = \
            tf.train.CheckpointManager(
                checkpoint=self.checkpoint,
                directory=ckptDir,
                max_to_keep=maxToKeep,
                checkpoint_name=ckptName,
            )

    def _getCkptName(self, ckptDir: str, config: Config) -> str:
        customName = config.getSetting('CustomCheckpointName')

        ckptBaseName = 'ckpt'
        if customName:
            ckptBaseName = ckptBaseName + "_%s" % customName

        existingIndices = [
            self._getFnameIdx(f)
            for f in os.listdir(baseDir)
            if self._isValidFname(baseDir, f)
        ]

        thisFileIdx = 0
        if len(existingIndices) > 0:
            thisFileIdx = max(thisFileIdx)

        return ckptBaseName + '_%d' % thisFileIdx

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
        self.checkpointManager.save()

    def restore(self):
        self.checkpointManager.restore_or_initializer()

