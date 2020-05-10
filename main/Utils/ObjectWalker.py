from typing import List, Iterator, Type, Set, Iterable
from ..Dtos.TraversedObject import TraversedObject

MAX_DEPTH = 30

class ObjectWalker:
    @staticmethod
    def walk(
        obj: object,
        filterFxn: Type['function'] = lambda x: True,
        ignoreStrs: List[str] = []
    ) -> Iterator[TraversedObject]:
        yield from ObjectWalker._walkInternal(
            obj,
            filterFxn,
            ignoreStrs,
            seenObjs=set(),
            pathStr='',
            depth=0
        )

    @staticmethod
    def _walkInternal(
        obj: object,
        filterFxn: Type['function'],
        ignoreStrs: Set[str],
        seenObjs: set,
        pathStr: str,
        depth: int
    ) -> Iterator[TraversedObject]:
        if depth >= MAX_DEPTH or id(obj) in seenObjs:
            return
        else:
            seenObjs.add(id(obj))

        if filterFxn(obj):
            yield TraversedObject(pathStr, obj)
            return

        genFxn = None
        if isinstance(obj, list):
            genFxn = ObjectWalker._walkList
        elif isinstance(obj, dict):
            genFxn = ObjectWalker._walkDict
        else:
            genFxn = ObjectWalker._walkDir

        yield from genFxn(obj, filterFxn, ignoreStrs, seenObjs, pathStr, depth + 1)

    @staticmethod
    def _walkList(
        listObj: List,
        filterFxn: Type['function'],
        ignoreStrs: Set[str],
        seenObjs: set,
        pathStr: str,
        depth: int
    ) -> Iterator[TraversedObject]:
        for i, innerObj in enumerate(listObj):
            newPathStr = pathStr + '_%d|' % i

            yield from ObjectWalker._walkInternal(
                innerObj,
                filterFxn,
                ignoreStrs,
                seenObjs,
                newPathStr,
                depth
            )

    @staticmethod
    def _walkDict(
        dictObj: List,
        filterFxn: Type['function'],
        ignoreStrs: Set[str],
        seenObjs: set,
        pathStr: str,
        depth: int
    ) -> Iterator[TraversedObject]:
        for key, innerObj in dictObj.items():
            if key in ignoreStrs:
                continue

            newPthStr = pathStr + '|%s' % key

            yield from ObjectWalker._walkInternal(
                innerObj,
                filterFxn,
                ignoreStrs,
                seenObjs,
                newPathStr,
                depth
            )

    @staticmethod
    def _walkDir(
        obj: object,
        filterFxn: Type['function'],
        ignoreStrs: Set[str],
        seenObjs: set,
        pathStr: str
    ) -> Iterator[TraversedObject]:
        for attrName in dir(obj):
            if attrName in ignoreStrs:
                continue

            newPathStr = pathStr + '|%s' % attrName

            yield from ObjectWalker._walkInternal(
                getattr(attrName, obj),
                filterFxn,
                ignoreStrs,
                seenObjs,
                newPathStr,
                depth
            )

    @staticmethod
    def _getAttributes(obj: object) -> Iterable[TraversedObject]:
        objs = map(lambda x: TraversedObject(y, getattr(obj, y)), dir(obj))
        return filter(lambda y: ObjectWalker._isValid(y), objs)

    @staticmethod
    def _isValid(travObj: TraversedObject) -> bool:
        nameValid = travObj.name[:2] != '__'
        isCallable = callable(travObj.obj)

        return nameValid and not isCallable

