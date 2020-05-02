from enum import Enum
from pathlib import Path
import numpy as np

class OrderedListType(Enum):
    DRUG = 1
    PROTEIN = 2

class AdjacencyMatricesWriter:
    def __init__(
        self,
        adjacencyMatrices,
        orderedDrugList,
        orderedProteinList
    ):
        self.adjacencyMatrices = adjacencyMatrices
        self.orderedDrugList = orderedDrugList
        self.orderedProteinList = orderedProteinList

    def writeToFileSystem(self, basePath):
        # Make dir if not exist
        Path(basePath).mkdir(parents=True, exist_ok=True)

        self._writeOrderedLists(basePath)
        self._writeMatrices(basePath)

    def _writeOrderedLists(self, basePath):
        self._writeOrderedList(basePath, OrderedListType.DRUG)
        self._writeOrderedList(basePath, OrderedListType.PROTEIN)

    def _writeOrderedList(self, basePath, listType):
        orderedListArr, compoundTypeStr = self._getOrderedListInfos(listType)
        headerTxt = '''
            Ordered %s names -- the ith drug here corresponds to the ith
            index of all %s related adjacency matrices.
        ''' % compoundTypeStr

        np.savetxt(
            fname='%s/ordered_%s_names.csv' % (basePath, compoundTypeStr),
            X=drugListArr,
            delimiter=',',
            header=headerTxt
        )

    def _getOrderedListInfos(self, listType):
        dataAsArr = None
        compoundTypeStr = ''

        if listType == OrderedListType.DRUG:
            dataAsArr = np.array(self.orderedDrugList)
            compoundTypeStr = 'drug'

        elif listType == OrderedListType.PROTEIN:
            dataAsArr = np.array(self.orderedProteinList)
            compoundTypeStr = 'protein'

        else:
            raise NotImplementedError

        return dataAsArr, compoundTypeStr

    def _writeMatrices(self, basePath):
        self._writeDrugDrugRelationAdjacencyMatrices(basePath)
        self._writeDrugProteinAdjacencyMatrix(basePath)
        self._writeProteinProteinAdjacencyMatrix(basePath)

    def _writeDrugDrugRelationAdjacencyMatrices(self, basePath):
        headerTxt = '''
            Adjacency matrix for drug drug pairs with relation of type %s.
            The ith drug may be identified exactly by the ith drug in the
            ordered-drug-names.csv file.
        '''

        adjMtxKvps = self.adjacencyMatrices.drugDrugRelationMtxs.items()
        for relationID, adjacencyMtx in adjMtxKvps:
            np.savetxt(
                fname='%s/drug-drug-adj-mtx-relation-%s.csv' % (basePath, relationID),
                X=adjacencyMtx,
                delimiter=',',
                header=headerTxt % relationID
            )

    def _writeDrugProteinAdjacencyMatrix(self, basePath):
        headerTxt = '''
            Adjacency matrix for drug protein pairs.
            The ith drug may be identified exactly by the ith drug in the
            ordered-drug-names.csv file. Similarly, the jth protein may be
            identified exactly by the jth drug in the
            ordered-protein-names.csv file.
        '''

        np.savetxt(
            fname='%s/drug-protein-adj-mtx.csv' % basePath,
            X=self.adjacencyMatrrices.drugProteinRelationMtx,
            delimiter=',',
            header=headerTxt
        )

    def _writeProteinProteinAdjacencyMatrix(self, basePath):
        headerTxt = '''
            Adjacency matrix for protein protein pairs.
            The ith protein may be identified exactly by the ith protein in the
            ordered-protein-names.csv file.
        '''

        np.savetxt(
            fname='%s/protein-protein-adj-mtx.csv' % basePath,
            X=self.adjacencyMatrrices.drugProteinRelationMtx,
            delimiter=',',
            header=headerTxt
        )

