from typing import List, Dict, Tuple, Type
from .NodeIds import DrugId, ProteinId

import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

_nxGraphType = Type[nx.Graph]
_sparseMtxType = Type[sp.spmatrix]

EdgeType = Tuple[int, int]
EdgeList = List[EdgeType]
RelationIDToEdgeList = Dict[str, EdgeList]
RelationIDToGraph = Dict[str, _nxGraphType]
RelationIDToSparseMtx = Dict[str, _sparseMtxType]

PlaceholdersDict = Dict[str, tf.placeholder]

