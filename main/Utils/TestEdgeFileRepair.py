from .Config import Config
from ..DataSetParsers.NodeLists.DecagonPublicDataNodeListsBuilder import DecagonPublicDataNodeListsBuilder
from ..Dtos.NodeLists import NodeLists
from ..Dtos.NodeIds import BaseNodeId, ProteinId, DrugId
from typing import Type

import csv
import _io

FROM_NODE_IDX = 0
TO_NODE_IDX   = 1
RELATION_IDX  = 2
LABEL_IDX     = 3

class TestEdgeFileRepairer:
    def __init__(self):
        self.config: Config = Config.getConfig()
        self.preFname: str = self.config.getSetting('EdgeFileToRepair')
        self.postFname: str = self._getPostFname(self.preFname)
        self.nodeLists: NodeLists = self._getNodeLists()

        # Just a utility to make code cleaner later
        self.nodeListDecoders = {
            DrugId: self.nodeLists.drugNodeList,
            ProteinId: self.nodeLists.proteinNodeList,
        }

    def _getNodeLists(self) -> NodeLists:
        nodeListBuilder = DecagonPublicDataNodeListsBuilder(self.config)
        return nodeListBuilder.build()

    def _getPostFname(self, preFname: str) -> str:
        csvIdx = preFname.find('.csv')

        return preFname[:csvIdx] + '-repaired.csv'

    def repair(self) -> None:
        with open(self.preFname, 'rt') as preF:
            with open(self.postFname, 'w') as postF:
                self._repair(preF, postF)

    def _repair(
        self,
        preFile: _io.TextIOWrapper,
        postFile: _io.TextIOWrapper
    ) -> None:
        reader = self._getReader(preFile)
        writer = self._getWriter(postFile)

        for row in reader:
            writer.writerow({
                'FromNode': self._processNode(row['FromNode']),
                'ToNode': self._processNode(row['FromNode']),
                'RelationId': row['RelationId'],
                'Label': row['Label'],
            })

        return

    def _processNode(self, preStr: str) -> str:
        if preStr[:3] != 'CID':
            return preStr

        preprocessed = self._preprocessNode(preStr)
        return self._postprocessNode(preprocessed)

    def _getReader(self, f: _io.TextIOWrapper) -> csv.DictReader:
        return csv.DictReader(f)

    def _getWriter(self, f: _io.TextIOWrapper) -> csv.DictWriter:
        fieldnames = [
            'FromNode',
            'ToNode',
            'RelationId',
            'Label'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        return writer

    def _preprocessNode(self, nodeStr: str) -> Type[BaseNodeId]:
        if nodeStr[:3] == 'CID':
            return DrugId.fromDecagonFormat(nodeStr)
        else:
            return ProteinId.fromDecagonFormat(nodeStr)

    def _postprocessNode(self, node: Type[BaseNodeId]) -> str:
        try:
            nodeType = type(node)
            trueValue = self.nodeListDecoders[nodeType][node]

            return nodeType.toDecagonFormat(trueValue)
        except IndexError:
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    repairer = TestEdgeFileRepairer()
    repairer.repair()

