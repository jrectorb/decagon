class NodeLists:
    def __init__(
        self,
        proteinNodeList: List[ProteinId],
        drugNodeList: List[DrugId]
    ) -> None:
        self.proteinNodeList: List[ProteinId] = proteinNodeList
        self.drugNodeList: List[DrugId] = drugNodeList

