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
    def __init__(self, session: tf.Session, config: Config) -> None:
        super().__init__(config)

        self.session: tf.Session = session
        self.shouldEverCheckpoint = bool(config.getSetting('ShouldCheckpoint'))

        maxToKeep: int = int(config.getSetting('MaxCheckpointsToKeep'))
        self.saver = tf.train.Saver = tf.train.Saver(max_to_keep=maxToKeep)

        self.saverBaseName = self._getModelSaverBaseName(config)

    def _getModelSaverBaseName(self, config: Config):
        ckptDir: str = config.getSetting('CheckpointDirectory')
        Path(ckptDir).mkdir(parents=True, exist_ok=True)

        customName = config.getSetting('CustomCheckpointName')
        ckptBaseName: str = 'ckpt'
        if customName:
            ckptBaseName = ckptBaseName + "_%s" % customName

        return ckptDir + ckptBaseName

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
        if not self.shouldEverCheckpoint:
            return

        self.saver.save(self.session, self.saverBaseName)

    def restore(self):
        if not self.shouldEverCheckpoint:
            return

        someCkptExists = self._getCurrFnameIdx() != -1
        currFname = self._getCkptName(isNew=False)

        self.saver.restore(self.session, self.ckptDir + currFname)

