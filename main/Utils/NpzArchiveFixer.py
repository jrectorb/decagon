from ..DataSetParsers.AdjacencyMatrices.DecagonPublicDataAdjacencyMatricesBuilder import DecagonPublicDataAdjacencyMatricesBuilder
from ..DataSetParsers.NodeLists.DecagonPublicDataNodeListsBuilder import DecagonPublicDataNodeListsBuilder
from ..Dtos.NodeIds import SideEffectId
from .Config import Config
import numpy as np

class NpzFixer:
    def __init__(self) -> None:
        conf = Config.getConfig()

        nodeLists = DecagonPublicDataNodeListsBuilder(conf).build()
        mtxBuilder = DecagonPublicDataAdjacencyMatricesBuilder(nodeLists, conf)

        self.edgeSets = mtxBuilder._getValidEdgeSets()

    def fixFile(self, fname: str) -> None:
        arrDict = {}

        allArr = np.load(fname)['arr_0']
        for i, relationId in enumerate(self.edgeSets.keys()):
            arrDict[SideEffectId.toDecagonFormat(relationId)] = allArr[i]

        np.savez(fname + '.repaired', **arrDict)

if __name__ == '__main__':
    fnames = [
        '/Users/jarridr/repos/DecagonPredictorDataSets/NumpyFiles/TrainedWithMasks/EmbeddingImportance.npyz.npz'
    ]

    fixer = NpzFixer()
    for fname in fnames:
        fixer.fixFile(fname)

