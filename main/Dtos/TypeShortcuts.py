from typing import List, Dict, Type, NewType
from .DrugId import DrugId
from .ProteinId import ProteinId

import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

EdgeList = List[tuple]
RelationIDToEdgeList = Dict[str, EdgeList]
RelationIDToGraph = Dict[str, Type[nx.Graph]]
RelationIDToSparseMtx = Dict[str, Type[sp.spmatrix]]

PlaceholdersDict = Dict[str, tf.placeholder]

