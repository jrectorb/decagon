from .DecagonPublicDataAdjacencyMatricesBuilder import DecagonPublicDataAdjacencyMatricesBuilder
from ...Dtos.TypeShortcuts import RelationIDToEdgeList

class HyperglycaemiaAdjMtxBuilder(DecagonPublicDataAdjacencyMatricesBuilder, functionalityType=None):
    # Hyperglycaemia STITCH ID is C0020456
    def _filterEdgeSets(
        self,
        allEdgeSets: RelationIDToEdgeList
    ) -> RelationIDToEdgeList:
        return {20456: allEdgeSets[20456]}

