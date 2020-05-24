from .RandomMaskingActiveLearner import RandomMaskingActiveLearner
from ..Dtos.Enums.ActiveLearnerType import ActiveLearnerType
from ..Utils.Config import Config
from typing import Set

class RelationFullMaskingLearner(
    RandomMaskingActiveLearner,
    functionalityType=ActiveLearnerType.RelationFullMaskingLearner
):
    def __init__(self, initDataSet, config: Config) -> None:
        super().__init__(initDataSet, config)

        self.invalidRelationIds: Set[str] = set(
            config.getSetting('InvalidRelationIds')
        )

    def _isRelationValid(self, relation: str) -> bool:
        return relation not in self.invalidRelationIds

