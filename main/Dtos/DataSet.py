from .AdjacencyMatrices import AdjacencyMatrices
from .NodeFeatures import NodeFeatures

class DataSet:
    def __init__(
        self,
        idStr: str,
        adjacencyMatrices: AdjacencyMatrices,
        nodeFeatures: NodeFeatures
    ) -> None:
        self.id = idStr
        self.adjacencyMatrices: AdjacencyMatrices = adjacencyMatrices
        self.nodeFeatures: NodeFeatures = nodeFeatures

