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
        seenObjs: Set[int],
        pathStr: str,
        depth: int
    ) -> Iterator[TraversedObject]:
        if not ObjectWalker._isValidObj(obj, depth, seenObjs):
            return
        else:
            seenObjs.add(id(obj))

        if filterFxn(obj):
            print(pathStr)
            yield TraversedObject(pathStr, obj)
            return

        # Only walk objects defined in our code
        if isinstance(obj, (dict, list)) or ObjectWalker._modulesMatch(obj):
            genFxn = None
            if isinstance(obj, list):
                genFxn = ObjectWalker._walkList
            elif isinstance(obj, dict):
                genFxn = ObjectWalker._walkDict
            else:
                genFxn = ObjectWalker._walkDir

            yield from genFxn(obj, filterFxn, ignoreStrs, seenObjs, pathStr, depth + 1)

        else:
            return

    @staticmethod
    def _isValidObj(obj: object, depth: int, seenObjs: Set[int]):
        cond1 = obj is not None
        cond2 = not callable(obj)
        cond3 = depth < MAX_DEPTH
        cond4 = id(obj) not in seenObjs

        return cond1 and cond2 and cond3 and cond4

    @staticmethod
    def _modulesMatch(obj):
        thisModulePrefix = __name__.split('.')[0]
        objModulePrefix  = obj.__module__.split('.')[0]

        return thisModulePrefix == objModulePrefix

    @staticmethod
    def _walkList(
        listObj: List,
        filterFxn: Type['function'],
        ignoreStrs: Set[str],
        seenObjs: Set[int],
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
        seenObjs: Set[int],
        pathStr: str,
        depth: int
    ) -> Iterator[TraversedObject]:
        for key, innerObj in dictObj.items():
            if key in ignoreStrs:
                continue

            newPathStr = pathStr + '|%s' % str(key)

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
        seenObjs: Set[int],
        pathStr: str,
        depth: int
    ) -> Iterator[TraversedObject]:
        print(obj)
        for attrName in dir(obj):
            if attrName in ignoreStrs or attrName[:2] == '__':
                continue

            newPathStr = pathStr + '|%s' % attrName

            try:
                yield from ObjectWalker._walkInternal(
                    getattr(obj, attrName),
                    filterFxn,
                    ignoreStrs,
                    seenObjs,
                    newPathStr,
                    depth
                )

            except AttributeError:
                continue

    @staticmethod
    def _getAttributes(obj: object) -> Iterable[TraversedObject]:
        objs = map(lambda x: TraversedObject(y, getattr(obj, y)), dir(obj))
        return filter(lambda y: ObjectWalker._isValidTraversedObj(y), objs)

    @staticmethod
    def _isValidTraversedObj(travObj: TraversedObject) -> bool:
        nameValid = travObj.name[:2] != '__'
        isCallable = callable(travObj.obj)

        return nameValid and not isCallable

