from .BaseAdjacencyMatricesBuilder import BaseAdjacencyMatricesBuilder
from ...Dtos.AdjacencyMatrices import AdjacencyMatrices
from ...Dtos.Enums.DataSetType import DataSetType
from ...Dtos.NodeLists import NodeLists
from ...Utils.Config import Config
from ...Utils.Sparse import RelationCsrMatrix
from typing import Dict, Type
from itertools import combinations
import networkx as nx
import numpy as np
import scipy.sparse as sp

RelationIDToGraph = Dict[str, Type[nx.Graph]]
RelationIDToSparseMtx = Dict[str, Type[sp.spmatrix]]

class DecagonDummyDataAdjacencyMatricesBuilder(
    BaseAdjacencyMatricesBuilder,
    functionalityType = DataSetType.DecagonDummyData
):
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        self.numDrugDrugRelationTypes: int = int(config.getSetting('NumDrugDrugRelationTypes'))
        self.numProteins: int              = len(nodeLists.proteinNodeList)
        self.numDrugs: int                 = len(nodeLists.drugNodeList)

    def build(self) -> AdjacencyMatrices:
        drugProteinRelationMtx: Type[sp.csr_matrix] = self._buildDrugProteinRelationMtx()
        drugDrugRelationMtxs: RelationIDToSparseMtx = \
            self._buildDrugDrugRelationMtxs(drugProteinRelationMtx)

        return AdjacencyMatrices(
            drugDrugRelationMtxs=drugDrugRelationMtxs,
            drugProteinRelationMtx=drugProteinRelationMtx,
            proteinProteinRelationMtx=self._buildPpiMtx(),
        )

    def _buildDrugProteinRelationMtx(self) -> Type[sp.csr_matrix]:
        preMtx = 10 * np.random.randn(self.numProteins, self.numDrugs)
        binaryMtx = (preMtx > 15).astype(int)

        return RelationCsrMatrix(binaryMtx)

    def _buildDrugDrugRelationMtxs(
        self,
        drugProteinMtx: Type[sp.csr_matrix]
    ) -> RelationIDToSparseMtx:
        result: RelationIDToSparseMtx = {}

        tmp = np.dot(drugProteinMtx.transpose(copy=True), drugProteinMtx)
        for i in range(self.numDrugDrugRelationTypes):
            mat = np.zeros((self.numDrugs, self.numDrugs))

            for d1, d2 in combinations(list(range(self.numDrugs)), 2):
                if tmp[d1, d2] == i + 4:
                    mat[d1, d2] = mat[d2, d1] = 1.

            result[i] = RelationCsrMatrix(mat)

        return result

    def _buildPpiMtx(self) -> Type[sp.csr_matrix]:
        plantedGraph = nx.planted_partition_graph(
            self.numProteins // 10,
            10,
            0.2,
            0.05,
            seed=42
        )

        return RelationCsrMatrix(nx.adjacency_matrix(plantedGraph))

