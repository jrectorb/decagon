from .AdjacencyMatrices import AdjacencyMatrices
from .NodeFeatures import NodeFeatures
from .NodeLists import NodeLists

class DataSet:
    def __init__(
        self,
        idStr: str,
        nodeLists: NodeLists,
        adjacencyMatrices: AdjacencyMatrices,
        nodeFeatures: NodeFeatures
    ) -> None:
        self.id: str = idStr
        self.nodeLists: NodeLists = nodeLists
        self.adjacencyMatrices: AdjacencyMatrices = adjacencyMatrices
        self.nodeFeatures: NodeFeatures = nodeFeatures

