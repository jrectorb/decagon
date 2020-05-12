from typing import Set

# heh, meta...
IDX_IDX  = 0
CHAR_IDX = 1

def rfindsub(val: str, chars: Set[str]) -> int:
    reversedVal = reversed(list(enumerate(val)))
    filterIter  = filter(lambda x: x[CHAR_IDX] in chars, reversedVal)

    # Get the first item from reversed string
    return next(map(lambda x: x[IDX_IDX], filterIter), -1)

