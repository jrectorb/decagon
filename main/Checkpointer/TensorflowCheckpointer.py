from .BaseCheckpointer import BaseCheckpointer
from ..Dtos.TensorflowTrainable import TensorflowTrainable
from ..Utils import StrUtils
from ..Utils.Config import Config
from pathlib import Path
import string
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os

NON_DIGIT_CHARS = set(string.printable).difference(set(string.digits))

class TensorflowCheckpointer(BaseCheckpointer):
    def __init__(
        self,
        trainable: TensorflowTrainable,
        session: tf.Session,
        config: Config
    ) -> None:
        super().__init__(config)

        self.checkpoint: tf.train.Checkpoint = \
            tfe.Checkpoint(**trainable.checkpointDict)

        self.session: tf.Session = session

        self.ckptDir: str = config.getSetting('CheckpointDirectory')
        Path(self.ckptDir).mkdir(parents=True, exist_ok=True)

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
            self._getFnameIdx(f, self.ckptBaseName)
            for f in os.listdir(self.ckptDir)
            if self._isValidFname(self.ckptDir, f, self.ckptBaseName)
        ]

        if len(existingIndices) > 0:
            return max(existingIndices)
        else:
            return -1

    def _getFnameIdx(self, fname: str, matchStr: str) -> int:
        fnameFirstPortion = fname.split('.')[0]
        lastUscoreIdx = fnameFirstPortion.rfind('_')

        strippedFname = fnameFirstPortion[lastUscoreIdx + 1:]

        nonNumIdx = StrUtils.rfindsub(strippedFname, NON_DIGIT_CHARS)
        if nonNumIdx != -1:
            strippedFname = strippedFname[:nonNumIdx]

        return int(strippedFname)

    def _isValidFname(self, baseDir: str, fname: str, matchStr: str) -> bool:
        isFile = os.path.isfile(baseDir + fname)

        # From "thiscoolfile.txt", extract "thiscoolfile"
        fnameFirstPortion = fname.split('.')[0]

        lastUscoreIdx = fnameFirstPortion.rfind('_')
        isGoodPrefix = fnameFirstPortion[:lastUscoreIdx] == matchStr

        return isFile and isGoodPrefix

    def save(self):
        newFname = self.ckptDir + self._getCkptName(isNew=True)
        self.checkpoint.save(newFname, session=self.session)

    def restore(self):
        someCkptExists = self._getCurrFnameIdx() != -1
        currFname = self._getCkptName(isNew=False)

        self.checkpoint.restore(self.ckptDir + currFname)

