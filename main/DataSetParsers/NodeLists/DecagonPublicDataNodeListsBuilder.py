from collections import defaultdict
from .AdjacencyMatrices import AdjacencyMatrices
from ..Dtos.TypeShortcuts import EdgeList, RelationIDToEdgeList, RelationIDToGraph, RelationIDToSparseMtx
from ..Dtos.Enums import AdjacencyMatricesType
from ..Utils import Config
import networkx as nx
import numpy as np
import scipy.sparse as sp

class DecagonPublicDataAdjacencyMatricesBuilder(
    BaseAdjacencyMatricesBuilder,
    adjacencyMatricesType = AdjacencyMatricesType.DecagonPublicData
):
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        self.drugDrugRelationGraph: nx.MultiGraph = nx.read_edgelist(
            config.getSetting('DecagonDrugDrugRelationsFilename'),
            delimiter=',',
            create_using=nx.MultiGraph(),
            data=(('relationType', str),)
        )

        self.ppiGraph: nx.Graph = nx.read_edgelist(
            config.getSetting('DecagonProteinProteinRelationsFilename'),
            delimiter=','
        )

    def build(self) -> AdjacencyMatrices:
        proteinNodeList = self._getOrderedProteinNodeList()
        drugNodeList    = self._getOrderedDrugNodeList()

        return NodeLists(proteinNodeList, drugNodeList)

    def _getOrderedDrugNodeList(self) -> EdgeList:
        allDrugs = set(
            self.drugDrugRelationGraph.nodes
        ).union(set(self._getDrugProteinGraphDrugs()))

        return sorted(list(allDrugs))

    def _getDrugProteinGraphDrugs(self) -> Iterator[tuple]:
        # In preprocessed dataset, all drug identifiers are prefixed with 'CID'
        # while protein identifiers are not
        return filter(
            lambda x: x[:3] == 'CID',
            self.drugProteinRelationGraph.nodes
        )

    def _getOrderedProteinNodeList(self) -> EdgeList:
        allProteins = set(
            self.ppiGraph.nodes
        ).union(set(self._getDrugProteinGraphProteins()))

        return sorted(list(allProteins))

    def _getDrugProteinGraphProteins(self) -> Iterator[tuple]:
        # In preprocessed dataset, all drug identifiers are prefixed with 'CID'
        # while protein identifiers are not
        return filter(
            lambda x: x[:3] != 'CID',
            self.drugProteinRelationGraph.nodes
        )

