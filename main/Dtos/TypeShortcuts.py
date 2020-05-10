from typing import List, Dict, Type
from .NodeIds import DrugId, ProteinId

import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

_nxGraphType = Type[nx.Graph]
_sparseMtxType = Type[sp.spmatrix]

EdgeList = List[tuple]
RelationIDToEdgeList = Dict[str, EdgeList]
RelationIDToGraph = Dict[str, _nxGraphType]
RelationIDToSparseMtx = Dict[str, _sparseMtxType]

PlaceholdersDict = Dict[str, tf.placeholder]

