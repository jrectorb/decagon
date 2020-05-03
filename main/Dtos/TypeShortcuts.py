from typing import List, Dict, Type

import networkx as nx
import scipy.sparse as sp

EdgeList = List[tuple]
RelationIDToEdgeList = Dict[str, EdgeList]
RelationIDToGraph = Dict[str, Type[nx.Graph]]
RelationIDToSparseMtx = Dict[str, Type[sp.spmatrix]]

