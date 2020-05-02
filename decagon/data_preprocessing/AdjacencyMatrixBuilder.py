from collections import defaultdict
from .AdjacencyMatrices import AdjacencyMatrices
import networkx as nx
import numpy as np
import scipy.sparse as sp

class AdjacencyMatrixBuilder:
    def __init__(
        self,
        drugDrugRelationFname,
        drugProteinRelationFname,
        ppiFname
    ):
        self.drugDrugRelationGraph = nx.read_edgelist(
            drugDrugRelationFname,
            delimiter=',',
            create_using=nx.MultiGraph(),
            data=(('relationType', str),)
        )

        self.drugProteinRelationGraph = nx.read_edgelist(
            drugProteinRelationFname,
            delimiter=','
        )

        self.ppiGraph = nx.read_edgelist(
            ppiFname,
            delimiter=','
        )

        # When building adjacency matrices with networkx, we can
        # guarantee ordering of the built matrices.  Thus, we precompute
        # them here so they can be used later in the building of the matrices.
        self.drugNodeList = self._getOrderedDrugNodeList()
        self.proteinNodeList = self._getOrderedProteinNodeList()

    def buildAdjacencyMatrices(self):
        return AdjacencyMatrices(
            drugDrugRelationMtxs=self._buildDrugDrugRelationMtxs(),
            drugProteinRelationMtx=self._buildDrugProteinRelationMtx(),
            ppiMtx=self._buildPpiMtx(),
        )

    def _buildDrugDrugRelationMtxs(self):
        validEdgeSets = self._getValidEdgeSets()
        graphs = self._getDrugDrugRelationGraphs(validEdgeSets)

        adjMtxs = {
            relID: nx.adjacency_matrix(graph) for relID, graph in graphs.items()
        }

        return adjMtxs

    def _getDrugDrugRelationGraphs(self, validEdgeSets):
        graphs = {}
        for relID, validEdgeSet in validEdgeSets.items():
            graph = nx.Graph()

            graph.add_nodes_from(self.drugNodeList)
            graph.add_edges_from(validEdgeSet)

            graphs[relID] = graph

        return graphs

    def _getValidEdgeSets(self):
        RELATION_TYPE_IDX = 2

        preResult = defaultdict(list)
        for edge in self.drugDrugRelationGraph.edges:
            # Remove edge type from edge
            truncatedEdge = edge[:2]
            preResult[edge[RELATION_TYPE_IDX]].append(truncatedEdge)

        result = {}
        # Filter out edge types that don't have 500
        for edgeType, edgeList in preResult.items():
            if self._isEdgeListValid(edgeType, edgeList):
                result[edgeType] = edgeList

        return result

    def _isEdgeListValid(self, edgeType, edgeList):
        return len(edgeList) >= 500

    def _buildDrugProteinRelationMtx(self):
        drugToIdx = {drug: idx for idx, drug in enumerate(self.drugNodeList)}
        proteinToIdx = {protein: idx for idx, protein in enumerate(self.proteinNodeList)}

        drugProteinMtx = np.zeros((len(drugToIdx), len(proteinToIdx)))
        for edge in self.drugProteinRelationGraph.edges:
            drug, protein = self._extractDrugProtein(edge)
            drugProteinMtx[drugToIdx[drug], proteinToIdx[protein]] = 1

        return sp.csr_matrix(drugProteinMtx)

    def _extractDrugProtein(self, edge):
        drugIdx = 0 if edge[0][:3] == 'CID' else 1
        proteinIdx = 1 - drugIdx

        return edge[drugIdx], edge[proteinIdx]

    def _buildPpiMtx(self):
        self.ppiGraph.add_nodes_from(self._getDrugProteinGraphProteins())
        return nx.adjacency_matrix(self.ppiGraph)

    def _getOrderedDrugNodeList(self):
        allDrugs = set(
            self.drugDrugRelationGraph.nodes
        ).union(set(self._getDrugProteinGraphDrugs()))

        return sorted(allDrugs)

    def _getDrugProteinGraphDrugs(self):
        # In preprocessed dataset, all drug identifiers are prefixed with 'CID'
        # while protein identifiers are not
        return filter(
            lambda x: x[:3] == 'CID',
            self.drugProteinRelationGraph.nodes
        )

    def _getOrderedProteinNodeList(self):
        allProteins = set(
            self.ppiGraph.nodes
        ).union(set(self._getDrugProteinGraphProteins()))

        return sorted(allProteins)

    def _getDrugProteinGraphProteins(self):
        # In preprocessed dataset, all drug identifiers are prefixed with 'CID'
        # while protein identifiers are not
        return filter(
            lambda x: x[:3] != 'CID',
            self.drugProteinRelationGraph.nodes
        )

