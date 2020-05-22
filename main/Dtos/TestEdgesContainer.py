from typing import Dict, List, Tuple

Edge = Tuple[int, int]
EdgeList  = List[Edge]

class TestEdgesContainer:
    def __init__(
        self,
        positiveEdgeSamples: List[EdgeList],
        negativeEdgeSamples: List[EdgeList]
    ) -> None:
        '''
        Positive edge samples are edges whose true labels are all 1.
        Similarly, negative edge samples are edges whose true labels are all 0.
        '''

        self.positiveEdgeSamples: List[EdgeList] = positiveEdgeSamples
        self.negativeEdgeSamples: List[EdgeList] = negativeEdgeSamples

    def retrievePosForGraphRelType(self, relType: int) -> EdgeList:
        return self.positiveEdgeSamples[relType]

    def retrieveNegForGraphRelType(self, relType: int) -> EdgeList:
        return self.negativeEdgeSamples[relType]

