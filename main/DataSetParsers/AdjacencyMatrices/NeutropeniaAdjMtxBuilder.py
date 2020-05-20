from .DecagonPublicDataAdjacencyMatricesBuilder import DecagonPublicDataAdjacencyMatricesBuilder
from ...Dtos.TypeShortcuts import RelationIDToEdgeList

class NeutropeniaAdjMtxBuilder(DecagonPublicDataAdjacencyMatricesBuilder, functionalityType=None):
    # Neutropenia STITCH ID is C0020456
    def _filterEdgeSets(
        self,
        allEdgeSets: RelationIDToEdgeList
    ) -> RelationIDToEdgeList:
        return {27947: allEdgeSets[27947]}

