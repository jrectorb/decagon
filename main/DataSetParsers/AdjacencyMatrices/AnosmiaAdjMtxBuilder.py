from .DecagonPublicDataAdjacencyMatricesBuilder import DecagonPublicDataAdjacencyMatricesBuilder
from ...Dtos.TypeShortcuts import RelationIDToEdgeList

class AnosmiaAdjMtxBuilder(DecagonPublicDataAdjacencyMatricesBuilder, functionalityType=None):
    # Anosmia STITCH ID is C0020456
    def _filterEdgeSets(
        self,
        allEdgeSets: RelationIDToEdgeList
    ) -> RelationIDToEdgeList:
        return {3126: allEdgeSets[3126]}

