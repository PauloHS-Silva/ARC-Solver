from typing import Union, Tuple, Any, Container, Callable, FrozenSet
import numpy as np
from numba import njit, prange
from functools import lru_cache

Boolean = bool
Integer = int
IntegerTuple = Tuple[Integer, Integer]
Numerical = Union[Integer, IntegerTuple]
IntegerSet = FrozenSet[Integer]
Grid = Tuple[Tuple[Integer]]
Cell = Tuple[Integer, IntegerTuple]
Object = FrozenSet[Cell]
Objects = FrozenSet[Object]
Indices = FrozenSet[IntegerTuple]
IndicesSet = FrozenSet[Indices]
Patch = Union[Object, Indices]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]
TupleTuple = Tuple[Tuple]
ContainerContainer = Container[Container]


def identity(x: Any) -> Any:
    """identity function"""
    return x


def add(a: Numerical, b: Numerical) -> Numerical:
    """addition"""
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)


def subtract(a: Numerical, b: Numerical) -> Numerical:
    """subtraction"""
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)


def multiply(a: Numerical, b: Numerical) -> Numerical:
    """multiplication"""
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)


def divide(a: Numerical, b: Numerical) -> Numerical:
    """floor division"""
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)


def invert(n: Numerical) -> Numerical:
    """inversion with respect to addition"""
    return -n if isinstance(n, int) else (-n[0], -n[1])


def even(n: Integer) -> Boolean:
    """evenness"""
    return n % 2 == 0


def double(n: Numerical) -> Numerical:
    """scaling by two"""
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)


def halve(n: Numerical) -> Numerical:
    """scaling by one half"""
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)


def flip(b: Boolean) -> Boolean:
    """logical not"""
    return not b


def equality(a: Any, b: Any) -> Boolean:
    """equality"""
    return a == b


def contained(value: Any, container: Container) -> Boolean:
    """element of"""
    return value in container


def combine(a: Container, b: Container) -> Container:
    """union"""
    return type(a)((*a, *b))


def intersection(a: FrozenSet, b: FrozenSet) -> FrozenSet:
    """returns the intersection of two containers"""
    return a & b


def difference(a: FrozenSet, b: FrozenSet) -> FrozenSet:
    """set difference"""
    return type(a)(e for e in a if e not in b)


def dedupe(tup: Tuple) -> Tuple:
    """remove duplicates"""
    return tuple(e for i, e in enumerate(tup) if tup.index(e) == i)


def order(container: Container, compfunc: Callable) -> Tuple:
    """order container by custom key"""
    return tuple(sorted(container, key=compfunc))


def repeat(item: Any, num: Integer) -> Tuple:
    """repetition of item within vector"""
    return tuple(item for i in range(num))


def greater(a: Integer, b: Integer) -> Boolean:
    """greater"""
    return a > b


def size(container: Container) -> Integer:
    """cardinality"""
    return len(container)


def merge(containers: ContainerContainer) -> Container:
    """merging"""
    return type(containers)(e for c in containers for e in c)


def maximum(container: IntegerSet) -> Integer:
    """maximum"""
    return max(container, default=0)


def minimum(container: IntegerSet) -> Integer:
    """minimum"""
    return min(container, default=0)


def valmax(container: Container, compfunc: Callable) -> Integer:
    """maximum by custom function"""
    return compfunc(max(container, key=compfunc, default=0))


def valmin(container: Container, compfunc: Callable) -> Integer:
    """minimum by custom function"""
    return compfunc(min(container, key=compfunc, default=0))


def argmax(container: Container, compfunc: Callable) -> Any:
    """largest item by custom order"""
    return max(container, key=compfunc)


def argmin(container: Container, compfunc: Callable) -> Any:
    """smallest item by custom order"""
    return min(container, key=compfunc)


def mostcommon(container: Container) -> Any:
    """most common item"""
    return max(set(container), key=container.count)


def leastcommon(container: Container) -> Any:
    """least common item"""
    return min(set(container), key=container.count)


def initset(value: Any) -> FrozenSet:
    """initialize container"""
    return frozenset({value})


def both(a: Boolean, b: Boolean) -> Boolean:
    """logical and"""
    return a and b


def either(a: Boolean, b: Boolean) -> Boolean:
    """logical or"""
    return a or b


def increment(x: Numerical) -> Numerical:
    """incrementing"""
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)


def decrement(x: Numerical) -> Numerical:
    """decrementing"""
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)


def crement(x: Numerical) -> Numerical:
    """incrementing positive and decrementing negative"""
    if isinstance(x, int):
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)
    return (
        0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
        0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1),
    )


def sign(x: Numerical) -> Numerical:
    """sign"""
    if isinstance(x, int):
        return 0 if x == 0 else (1 if x > 0 else -1)
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1),
    )


def positive(x: Integer) -> Boolean:
    """positive"""
    return x > 0


def toivec(i: Integer) -> IntegerTuple:
    """vector pointing vertically"""
    return (i, 0)


def tojvec(j: Integer) -> IntegerTuple:
    """vector pointing horizontally"""
    return (0, j)


def sfilter(container: Container, condition: Callable) -> Container:
    """keep elements in container that satisfy condition"""
    return type(container)(e for e in container if condition(e))


def mfilter(container: Container, function: Callable) -> FrozenSet:
    """filter and merge"""
    return merge(sfilter(container, function))


def extract(container: Container, condition: Callable) -> Any:
    """first element of container that satisfies condition"""
    return next(e for e in container if condition(e))


def totuple(container: FrozenSet) -> Tuple:
    """conversion to tuple"""
    return tuple(container)


def first(container: Container) -> Any:
    """first item of container"""
    return next(iter(container))


def last(container: Container) -> Any:
    """last item of container"""
    return max(enumerate(container))[1]


def insert(value: Any, container: FrozenSet) -> FrozenSet:
    """insert item into container"""
    return container.union(frozenset({value}))


def remove(value: Any, container: Container) -> Container:
    """remove item from container"""
    return type(container)(e for e in container if e != value)


def other(container: Container, value: Any) -> Any:
    """other value in the container"""
    return first(remove(value, container))


def interval(start: Integer, stop: Integer, step: Integer) -> Tuple:
    """range"""
    return tuple(range(start, stop, step))


def astuple(a: Integer, b: Integer) -> IntegerTuple:
    """constructs a tuple"""
    return (a, b)


def product(a: Container, b: Container) -> FrozenSet:
    """cartesian product"""
    return frozenset((i, j) for j in b for i in a)


def pair(a: Tuple, b: Tuple) -> TupleTuple:
    """zipping of two tuples"""
    return tuple(zip(a, b))


def branch(condition: Boolean, a: Any, b: Any) -> Any:
    """if else branching"""
    return a if condition else b


def compose(outer: Callable, inner: Callable) -> Callable:
    """function composition"""
    return lambda x: outer(inner(x))


def chain(
    h: Callable,
    g: Callable,
    f: Callable,
) -> Callable:
    """function composition with three functions"""
    return lambda x: h(g(f(x)))


def matcher(function: Callable, target: Any) -> Callable:
    """construction of equality function"""
    return lambda x: function(x) == target


def rbind(function: Callable, fixed: Any) -> Callable:
    """fix the rightmost argument"""
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)


def lbind(function: Callable, fixed: Any) -> Callable:
    """fix the leftmost argument"""
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)


def power(function: Callable, n: Integer) -> Callable:
    """power of function"""
    if n == 1:
        return function
    return compose(function, power(function, n - 1))


def fork(outer: Callable, a: Callable, b: Callable) -> Callable:
    """creates a wrapper function"""
    return lambda x: outer(a(x), b(x))


def apply(function: Callable, container: Container) -> Container:
    """apply function to each item in container"""
    return type(container)(function(e) for e in container)


def rapply(functions: Container, value: Any) -> Container:
    """apply each function in container to value"""
    return type(functions)(function(value) for function in functions)


def mapply(function: Callable, container: ContainerContainer) -> FrozenSet:
    """apply and merge"""
    return merge(apply(function, container))


def papply(function: Callable, a: Tuple, b: Tuple) -> Tuple:
    """apply function on two vectors"""
    return tuple(function(i, j) for i, j in zip(a, b))


def mpapply(function: Callable, a: Tuple, b: Tuple) -> Tuple:
    """apply function on two vectors and merge"""
    return merge(papply(function, a, b))


def prapply(function, a: Container, b: Container) -> FrozenSet:
    """apply function on cartesian product"""
    return frozenset(function(i, j) for j in b for i in a)


def mostcolor(element) -> int:
    """most common color"""
    if isinstance(element, tuple):
        arr = np.asarray(element, dtype=np.int64)
        flat = arr.ravel()
    else:
        n = len(element)
        if n == 0:
            return 0
        flat = np.fromiter((v for v, _ in element), dtype=np.int64, count=n)

    if flat.size == 0:
        return 0
    counts = np.bincount(flat)
    return int(counts.argmax())


def leastcolor(element) -> int:
    """least common color"""
    if isinstance(element, tuple):
        arr = np.asarray(element, dtype=np.int64)
        flat = arr.ravel()
    else:
        n = len(element)
        if n == 0:
            return 0
        flat = np.fromiter((v for v, _ in element), dtype=np.int64, count=n)
    if flat.size == 0:
        return 0
    # Count occurrences only for present values
    counts = np.bincount(flat)
    uniques = np.unique(flat)
    # Select the unique with the smallest count
    idx_min = np.argmin(counts[uniques])
    return int(uniques[idx_min])


def height(piece: Piece) -> Integer:
    """height of grid or patch"""
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1


def width(piece: Piece) -> Integer:
    """width of grid or patch"""
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1


def shape(piece: Piece) -> IntegerTuple:
    """height and width of grid or patch"""
    return (height(piece), width(piece))


def portrait(piece: Piece) -> Boolean:
    """whether height is greater than width"""
    return height(piece) > width(piece)


@njit(cache=True)
def _colorcount_patch(vals: np.ndarray, target: np.int64) -> int:
    cnt = 0
    for k in range(vals.shape[0]):
        if vals[k] == target:
            cnt += 1
    return cnt


def colorcount(element, value: int) -> int:
    """number of cells with color"""
    if isinstance(element, tuple):
        arr = np.array(element, dtype=np.int64)
        return int((arr == value).sum())
    vals = np.fromiter((v for v, _ in element), dtype=np.int64, count=len(element))
    return int(_colorcount_patch(vals, np.int64(value)))


def colorfilter(objs: Objects, value: Integer) -> Objects:
    """filter objects by color"""
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)


def sizefilter(container: Container, n: Integer) -> FrozenSet:
    """filter items by size"""
    return frozenset(item for item in container if len(item) == n)


@lru_cache(maxsize=None)
def _asindices_cached(shape: Tuple[int, int]) -> FrozenSet[Tuple[int, int]]:
    h, w = shape
    return frozenset((i, j) for i in range(h) for j in range(w))


def asindices(grid: Grid) -> Indices:
    """indices of all grid cells (cached by grid shape)"""
    return _asindices_cached((len(grid), len(grid[0])))


def ofcolor(grid: Grid, value: Integer) -> Indices:
    """indices of all grid cells with value"""
    return frozenset(
        (i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
    )


def ulcorner(patch: Patch) -> IntegerTuple:
    """index of upper-left corner"""
    it = iter(patch)
    first = next(it)

    if isinstance(first[1], tuple):
        _, (min_i, min_j) = first
        for _, (i, j) in it:
            if i < min_i:
                min_i = i
            if j < min_j:
                min_j = j
    else:
        min_i, min_j = first
        for i, j in it:
            if i < min_i:
                min_i = i
            if j < min_j:
                min_j = j

    return (min_i, min_j)


def urcorner(patch: Patch) -> IntegerTuple:
    """index of upper right corner"""
    return tuple(
        map(
            lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))
        )
    )


def llcorner(patch: Patch) -> IntegerTuple:
    """index of lower left corner"""
    return tuple(
        map(
            lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))
        )
    )


def lrcorner(patch: Patch) -> IntegerTuple:
    """index of lower right corner"""
    return tuple(map(max, zip(*toindices(patch))))


def crop(grid: Grid, start: IntegerTuple, dims: IntegerTuple) -> Grid:
    """subgrid specified by start and dimension"""
    return tuple(
        r[start[1] : start[1] + dims[1]] for r in grid[start[0] : start[0] + dims[0]]
    )


@lru_cache(maxsize=None)
def _toindices_cached(patch: FrozenSet) -> FrozenSet:
    first = next(iter(patch))
    if isinstance(first[1], tuple):
        return frozenset(idx for _, idx in patch)
    else:
        return patch


def toindices(patch: Patch) -> Indices:
    """indices of object cells"""
    if not patch:
        return frozenset()
    return _toindices_cached(patch)


def recolor(value: Integer, patch: Patch) -> Object:
    """recolor patch"""
    return frozenset((value, index) for index in toindices(patch))


@njit
def _shift_core(i_arr, j_arr, di, dj):
    for k in range(i_arr.shape[0]):
        i_arr[k] += di
        j_arr[k] += dj


def shift(patch: Patch, directions: IntegerTuple) -> Patch:
    if not patch:
        return patch

    di, dj = directions
    it = iter(patch)
    sample = next(it)
    is_obj = isinstance(sample[1], tuple)

    if is_obj:
        vals, coords = zip(*patch)
        coords = np.array(coords, np.int64)
        i_arr = coords[:, 0].copy()
        j_arr = coords[:, 1].copy()
    else:
        coords = np.array(list(patch), np.int64)
        i_arr = coords[:, 0].copy()
        j_arr = coords[:, 1].copy()

    _shift_core(i_arr, j_arr, di, dj)

    if is_obj:
        return frozenset(zip(vals, zip(i_arr.tolist(), j_arr.tolist())))
    else:
        return frozenset(zip(i_arr.tolist(), j_arr.tolist()))


def normalize(patch: Patch) -> Patch:
    """moves upper left corner to origin"""
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))


def dneighbors(loc: IntegerTuple) -> Indices:
    """directly adjacent indices"""
    return frozenset(
        {
            (loc[0] - 1, loc[1]),
            (loc[0] + 1, loc[1]),
            (loc[0], loc[1] - 1),
            (loc[0], loc[1] + 1),
        }
    )


def ineighbors(loc: IntegerTuple) -> Indices:
    """diagonally adjacent indices"""
    return frozenset(
        {
            (loc[0] - 1, loc[1] - 1),
            (loc[0] - 1, loc[1] + 1),
            (loc[0] + 1, loc[1] - 1),
            (loc[0] + 1, loc[1] + 1),
        }
    )


def neighbors(loc: IntegerTuple) -> Indices:
    """adjacent indices"""
    return dneighbors(loc) | ineighbors(loc)


@njit(cache=True)
def _objects_label_kernel(
    grid: np.ndarray, univalued: bool, diagonal: bool, without_bg: bool
):
    h, w = grid.shape
    # Compute bg if needed
    bg = -1
    if without_bg:
        # Find most common color by simple histogram
        maxv = grid.max() + 1
        cnt = np.zeros(maxv, np.int64)
        for i in range(h):
            for j in range(w):
                cnt[grid[i, j]] += 1
        # background = argmax
        bi = 0
        for v in range(1, maxv):
            if cnt[v] > cnt[bi]:
                bi = v
        bg = bi

    # Visited mask and label image
    visited = np.zeros((h, w), np.uint8)
    labels = -1 * np.ones((h, w), np.int64)

    # Neighbor offsets
    if diagonal:
        di_arr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
        dj_arr = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)
    else:
        di_arr = np.array([-1, 0, 0, 1], np.int64)
        dj_arr = np.array([0, -1, 1, 0], np.int64)

    next_label = 0

    for si in range(h):
        for sj in range(w):
            if visited[si, sj]:
                continue
            val0 = grid[si, sj]
            if without_bg and val0 == bg:
                visited[si, sj] = 1
                continue

            # Start a new component
            head = 0
            tail = 1
            # Simple queue in flat arrays
            q_i = np.empty(h * w, np.int64)
            q_j = np.empty(h * w, np.int64)
            q_i[0], q_j[0] = si, sj
            visited[si, sj] = 1
            labels[si, sj] = next_label

            while head < tail:
                ci = q_i[head]
                cj = q_j[head]
                head += 1
                # Explore neighbors
                for k in range(di_arr.shape[0]):
                    ni = ci + di_arr[k]
                    nj = cj + dj_arr[k]
                    if ni < 0 or ni >= h or nj < 0 or nj >= w:
                        continue
                    if visited[ni, nj]:
                        continue
                    v = grid[ni, nj]
                    if univalued:
                        if v != val0:
                            continue
                    else:
                        if without_bg and v == bg:
                            continue
                    # Accept into this component
                    visited[ni, nj] = 1
                    labels[ni, nj] = next_label
                    q_i[tail] = ni
                    q_j[tail] = nj
                    tail += 1

            next_label += 1

    return labels, next_label


def objects(
    grid: Grid, univalued: Boolean, diagonal: Boolean, without_bg: Boolean
) -> Objects:
    """objects occurring on the grid"""
    arr = np.array(grid, dtype=np.int64)
    # Run the fast C flood-fill
    labels, ncomp = _objects_label_kernel(arr, univalued, diagonal, without_bg)
    # Build the frozensets of (value,(i,j))
    out = []
    for lbl in range(ncomp):
        # Find all cells with this label
        coords = np.argwhere(labels == lbl)
        vals = arr[labels == lbl]
        # Build a single component
        comp = frozenset(
            (int(vals[k]), (int(coords[k, 0]), int(coords[k, 1])))
            for k in range(vals.shape[0])
        )
        out.append(comp)
    return frozenset(out)


def partition(grid: Grid) -> Objects:
    """each cell with the same value part of the same object"""
    return frozenset(
        frozenset(
            (v, (i, j))
            for i, r in enumerate(grid)
            for j, v in enumerate(r)
            if v == value
        )
        for value in palette(grid)
    )


def fgpartition(grid: Grid) -> Objects:
    """each cell with the same value part of the same object without background"""
    return frozenset(
        frozenset(
            (v, (i, j))
            for i, r in enumerate(grid)
            for j, v in enumerate(r)
            if v == value
        )
        for value in palette(grid) - {mostcolor(grid)}
    )


def uppermost(patch: Patch) -> Integer:
    """row index of uppermost occupied cell"""
    return min(i for i, j in toindices(patch))


def lowermost(patch: Patch) -> Integer:
    """row index of lowermost occupied cell"""
    return max(i for i, j in toindices(patch))


def leftmost(patch: Patch) -> Integer:
    """column index of leftmost occupied cell"""
    return min(j for i, j in toindices(patch))


def rightmost(patch: Patch) -> Integer:
    """column index of rightmost occupied cell"""
    return max(j for i, j in toindices(patch))


def square(piece: Piece) -> Boolean:
    """whether the piece forms a square"""
    return (
        len(piece) == len(piece[0])
        if isinstance(piece, tuple)
        else height(piece) * width(piece) == len(piece)
        and height(piece) == width(piece)
    )


def vline(patch: Patch) -> Boolean:
    """whether the piece forms a vertical line"""
    return height(patch) == len(patch) and width(patch) == 1


def hline(patch: Patch) -> Boolean:
    """whether the piece forms a horizontal line"""
    return width(patch) == len(patch) and height(patch) == 1


def hmatching(a: Patch, b: Patch) -> Boolean:
    """whether there exists a row for which both patches have cells"""
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0


def vmatching(a: Patch, b: Patch) -> Boolean:
    """whether there exists a column for which both patches have cells"""
    return len(set(j for i, j in toindices(a)) & set(j for i, j in toindices(b))) > 0


def manhattan(a: Patch, b: Patch) -> Integer:
    """closest manhattan distance between two patches"""
    return min(
        abs(ai - bi) + abs(aj - bj)
        for ai, aj in toindices(a)
        for bi, bj in toindices(b)
    )


def adjacent(a: Patch, b: Patch) -> Boolean:
    """whether two patches are adjacent"""
    return manhattan(a, b) == 1


def bordering(patch: Patch, grid: Grid) -> Boolean:
    """whether a patch is adjacent to a grid border"""
    return (
        uppermost(patch) == 0
        or leftmost(patch) == 0
        or lowermost(patch) == len(grid) - 1
        or rightmost(patch) == len(grid[0]) - 1
    )


def centerofmass(patch: Patch) -> IntegerTuple:
    """center of mass"""
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindices(patch))))


def palette(element: Element) -> IntegerSet:
    """colors occurring in object or grid"""
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})


def numcolors(element: Element) -> IntegerSet:
    """number of colors occurring in object or grid"""
    return len(palette(element))


def color(obj: Object) -> Integer:
    """color of object"""
    return next(iter(obj))[0]


def toobject(patch: Patch, grid: Grid) -> Object:
    """object from patch and grid"""
    h, w = len(grid), len(grid[0])
    return frozenset(
        (grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w
    )


def asobject(grid: Grid) -> Object:
    """conversion of grid to object"""
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))


def rot90(grid: Grid) -> Grid:
    """quarter clockwise rotation"""
    return tuple(row for row in zip(*grid[::-1]))


def rot180(grid: Grid) -> Grid:
    """half rotation"""
    return tuple(tuple(row[::-1]) for row in grid[::-1])


def rot270(grid: Grid) -> Grid:
    """quarter anticlockwise rotation"""
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]


def hmirror(piece: Piece) -> Piece:
    """mirroring along horizontal"""
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)


def vmirror(piece: Piece) -> Piece:
    """mirroring along vertical"""
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)


def dmirror(piece: Piece) -> Piece:
    """mirroring along diagonal"""
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)


def cmirror(piece: Piece) -> Piece:
    """mirroring along counterdiagonal"""
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))


@njit(parallel=True, cache=True)
def _fill_kernel(
    arr: np.ndarray, rows: np.ndarray, cols: np.ndarray, value: np.int64
) -> np.ndarray:
    h, w = arr.shape
    n = rows.shape[0]
    for k in prange(n):
        i = rows[k]
        j = cols[k]
        if 0 <= i < h and 0 <= j < w:
            arr[i, j] = value
    return arr


def fill(grid, value, patch):
    """fill value at indices"""
    arr = np.array(grid, dtype=np.int64)
    idxs = toindices(patch)
    n = len(idxs)
    rows = np.empty(n, dtype=np.int64)
    cols = np.empty(n, dtype=np.int64)
    for k, (i, j) in enumerate(idxs):
        rows[k] = i
        cols[k] = j
    arr_filled = _fill_kernel(arr, rows, cols, np.int64(value))
    return tuple(map(tuple, arr_filled))


@njit(parallel=True)
def _paint_kernel(
    arr: np.ndarray, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
) -> np.ndarray:
    h, w = arr.shape
    n = vals.shape[0]
    for k in prange(n):
        i = rows[k]
        j = cols[k]
        if 0 <= i < h and 0 <= j < w:
            arr[i, j] = vals[k]
    return arr


def paint(grid: Grid, obj: Object) -> Grid:
    """paint object to grid"""
    arr = np.array(grid, dtype=np.int64)
    n = len(obj)
    if n:
        rows = np.empty(n, np.int64)
        cols = np.empty(n, np.int64)
        vals = np.empty(n, np.int64)
        for k, (v, (i, j)) in enumerate(obj):
            vals[k] = v
            rows[k] = i
            cols[k] = j
        arr = _paint_kernel(arr, rows, cols, vals)
    return tuple(map(tuple, arr))


@njit(parallel=True, cache=True)
def _underfill_kernel(
    arr: np.ndarray, rows: np.ndarray, cols: np.ndarray, value: np.int64, bg: np.int64
) -> np.ndarray:
    h, w = arr.shape
    for k in prange(rows.shape[0]):
        i = rows[k]
        j = cols[k]
        if 0 <= i < h and 0 <= j < w and arr[i, j] == bg:
            arr[i, j] = value
    return arr


def underfill(grid: Grid, value: Integer, patch: Patch) -> Grid:
    """fill value at indices that are background"""
    arr = np.array(grid, dtype=np.int64)
    # Find background color via numpy.unique
    vals, counts = np.unique(arr, return_counts=True)
    bg = vals[counts.argmax()]
    # Pack patch indices
    idxs = toindices(patch)
    n = len(idxs)
    rows = np.empty(n, dtype=np.int64)
    cols = np.empty(n, dtype=np.int64)
    for k, (i, j) in enumerate(idxs):
        rows[k] = i
        cols[k] = j

    arr_filled = _underfill_kernel(arr, rows, cols, np.int64(value), np.int64(bg))
    return tuple(map(tuple, arr_filled))


@njit(parallel=True, cache=True)
def _underpaint_kernel(
    arr: np.ndarray, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, bg: np.int64
) -> np.ndarray:
    h, w = arr.shape
    n = rows.shape[0]
    for k in prange(n):
        i = rows[k]
        j = cols[k]
        if 0 <= i < h and 0 <= j < w and arr[i, j] == bg:
            arr[i, j] = vals[k]
    return arr


def underpaint(grid, obj):
    """paint object to grid where there is background"""
    arr = np.array(grid, dtype=np.int64)
    # Find background color
    unique_vals, counts = np.unique(arr, return_counts=True)
    bg = unique_vals[counts.argmax()]
    # Pack object
    n = len(obj)
    rows = np.empty(n, dtype=np.int64)
    cols = np.empty(n, dtype=np.int64)
    vals = np.empty(n, dtype=np.int64)
    for k, (v, (i, j)) in enumerate(obj):
        rows[k] = i
        cols[k] = j
        vals[k] = v
    arr_painted = _underpaint_kernel(arr, rows, cols, vals, np.int64(bg))
    return tuple(map(tuple, arr_painted))


def hupscale(grid: Grid, factor: int) -> Grid:
    """upscale grid horizontally"""
    if not grid or not grid[0]:
        return grid
    if len(grid[0]) * factor > 30:
        raise ValueError

    arr = np.array(grid, dtype=np.int64)
    # repeat each column 'factor' times
    hr = arr.repeat(factor, axis=1)
    return tuple(map(tuple, hr))


def vupscale(grid: Grid, factor: int) -> Grid:
    """upscale grid vertically"""
    if not grid:
        return grid
    if len(grid) * factor > 30:
        raise ValueError("Upscaled grid height cannot exceed 30")
    
    arr = np.array(grid, dtype=np.int64)
    # Repeat each row 'factor' times downwards
    vr = arr.repeat(factor, axis=0)
    return tuple(map(tuple, vr))


def upscale(element: Element, factor: Integer) -> Element:
    """upscale object or grid"""
    if isinstance(element, tuple):
        if not element or not element[0]:
            return element
        h = len(element)
        w = len(element[0])
        if (h * factor > 30) or (w * factor > 30):
            raise ValueError
        g = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            g = g + tuple(upscaled_row for num in range(factor))
        return g
    else:
        if not element:
            return frozenset()
        indices = toindices(element)
        h = height(indices)
        w = width(indices)
        if (h * factor > 30) or (w * factor > 30):
            raise ValueError
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        o = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    o.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(o), (di_inv, dj_inv))


def downscale(grid: Grid, factor: int) -> Grid:
    """downscale grid"""
    arr = np.array(grid, dtype=np.int64)
    # Take every factor-th row and column
    ds = arr[::factor, ::factor]
    return tuple(map(tuple, ds))


def hconcat(a: Grid, b: Grid) -> Grid:
    """concatenate two grids horizontally"""
    if not a:
        return b
    if not b:
        return a
    h_a, w_a = len(a), len(a[0])
    h_b, w_b = len(b), len(b[0])
    if h_a != h_b:
        raise ValueError
    if (w_a + w_b) > 30:
        raise ValueError
    return tuple(i + j for i, j in zip(a, b))


def vconcat(a: Grid, b: Grid) -> Grid:
    """concatenate two grids vertically"""
    if not a:
        return b
    if not b:
        return a
    h_a, w_a = len(a), len(a[0])
    h_b, w_b = len(b), len(b[0])
    if w_a != w_b:
        raise ValueError
    if (h_a + h_b) > 30:
        raise ValueError
    return a + b


def subgrid(patch: Patch, grid: Grid) -> Grid:
    """smallest subgrid containing object"""
    return crop(grid, ulcorner(patch), shape(patch))


def hsplit(grid: Grid, n: Integer) -> Tuple:
    """split grid horizontally"""
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))


def vsplit(grid: Grid, n: Integer) -> Tuple:
    """split grid vertically"""
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))


def cellwise(a: Grid, b: Grid, fallback: int) -> Grid:
    """cellwise match of two grids"""
    arr_a = np.array(a, dtype=np.int64)
    arr_b = np.array(b, dtype=np.int64)
    # Vectorized comparison and selection
    res = np.where(arr_a == arr_b, arr_a, fallback)
    return tuple(map(tuple, res))


def replace(grid: Grid, replacee: Integer, replacer: Integer) -> Grid:
    """color substitution"""
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)


def switch(grid: Grid, a: Integer, b: Integer) -> Grid:
    """color switching"""
    return tuple(
        tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid
    )


def center(patch: Patch) -> IntegerTuple:
    """center of the patch"""
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)


def position(a: Patch, b: Patch) -> IntegerTuple:
    """relative position between two patches"""
    ia, ja = center(toindices(a))
    ib, jb = center(toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)


def index(grid: Grid, loc: IntegerTuple) -> Integer:
    """color at location"""
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]]


def canvas(value: Integer, dimensions: IntegerTuple) -> Grid:
    """grid construction"""
    return tuple(
        tuple(value for j in range(dimensions[1])) for i in range(dimensions[0])
    )


def corners(patch: Patch) -> Indices:
    """indices of corners"""
    return frozenset(
        {ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)}
    )


def connect(a: IntegerTuple, b: IntegerTuple) -> Indices:
    """line between two points"""
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset(
            (i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1))
        )
    return frozenset()


def cover(grid: Grid, patch: Patch) -> Grid:
    """remove object from grid"""
    return fill(grid, mostcolor(grid), toindices(patch))


def trim(grid: Grid) -> Grid:
    """trim border of grid"""
    return tuple(r[1:-1] for r in grid[1:-1])


def move(grid: Grid, obj: Object, offset: IntegerTuple) -> Grid:
    """move object on grid"""
    return paint(cover(grid, obj), shift(obj, offset))


def tophalf(grid: Grid) -> Grid:
    """upper half of grid"""
    return grid[: len(grid) // 2]


def bottomhalf(grid: Grid) -> Grid:
    """lower half of grid"""
    return grid[len(grid) // 2 + len(grid) % 2 :]


def lefthalf(grid: Grid) -> Grid:
    """left half of grid"""
    return rot270(tophalf(rot90(grid)))


def righthalf(grid: Grid) -> Grid:
    """right half of grid"""
    return rot270(bottomhalf(rot90(grid)))


def vfrontier(location: IntegerTuple) -> Indices:
    """vertical frontier"""
    return frozenset((i, location[1]) for i in range(30))


def hfrontier(location: IntegerTuple) -> Indices:
    """horizontal frontier"""
    return frozenset((location[0], j) for j in range(30))


def backdrop(patch: Patch) -> Indices:
    """indices in bounding box of patch"""
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))


def delta(patch: Patch) -> Indices:
    """indices in bounding box but not part of patch"""
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)


def gravitate(source: Patch, destination: Patch) -> IntegerTuple:
    """direction to move source until adjacent to destination"""
    si, sj = center(source)
    di, dj = center(destination)
    i, j = 0, 0
    if vmatching(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacent(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shift(source, (i, j))
    return (gi - i, gj - j)


def inbox(patch: Patch) -> Indices:
    """inbox for patch"""
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def outbox(patch: Patch) -> Indices:
    """outbox for patch"""
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def box(patch: Patch) -> Indices:
    """outline of patch"""
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def shoot(start: IntegerTuple, direction: IntegerTuple) -> Indices:
    """line from starting point and direction"""
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))


@njit(parallel=True, cache=True)
def _occurrences_mask(
    grid: np.ndarray, vals: np.ndarray, coords: np.ndarray, h2: int, w2: int
) -> np.ndarray:
    """
    Build a boolean mask of shape (h2, w2) where mask[i,j] == 1 iff
    placing the pattern defined by (vals, coords) at (i,j) matches grid.
    """
    mask = np.zeros((h2, w2), np.uint8)
    n = vals.shape[0]
    for i in prange(h2):
        for j in range(w2):
            ok = True
            for k in range(n):
                gi = coords[k, 0] + i
                gj = coords[k, 1] + j
                if grid[gi, gj] != vals[k]:
                    ok = False
                    break
            if ok:
                mask[i, j] = 1
    return mask


def occurrences(grid: Grid, obj: Object) -> Indices:
    """locations of occurrences of object in grid"""
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape

    # Unpack obj into parallel arrays
    m = len(obj)
    vals = np.empty(m, dtype=np.int64)
    coords = np.empty((m, 2), dtype=np.int64)
    for idx, (v, (i, j)) in enumerate(obj):
        vals[idx] = v
        coords[idx, 0] = i
        coords[idx, 1] = j

    # Normalize to upper-left = (0,0)
    min_i, min_j = ulcorner(obj)
    coords[:, 0] -= min_i
    coords[:, 1] -= min_j

    # Compute object shape
    oh = int(coords[:, 0].max()) + 1
    ow = int(coords[:, 1].max()) + 1

    # Sliding window size
    h2 = h - oh + 1
    w2 = w - ow + 1
    if h2 <= 0 or w2 <= 0:
        return frozenset()

    # Mask of matches
    mask = _occurrences_mask(arr, vals, coords, h2, w2)

    # Collect positions
    occs = {(int(i), int(j)) for i in range(h2) for j in range(w2) if mask[i, j]}
    return frozenset(occs)


def frontiers(grid: Grid) -> Objects:
    """set of frontiers"""
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset(
        {frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices}
    )
    vfrontiers = frozenset(
        {frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices}
    )
    return hfrontiers | vfrontiers


def compress(grid: Grid) -> Grid:
    """removes frontiers from grid"""
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(
        tuple(v for j, v in enumerate(r) if j not in ci)
        for i, r in enumerate(grid)
        if i not in ri
    )


@njit(cache=True)
def _hperiod_core(arr: np.ndarray, coords: np.ndarray, w: int) -> int:
    """
    arr: 2D array of shape (h, w) with arr[i,j] = color or sentinel at holes
    coords: shape (n,2) array of normalized (i,j) locations of each object cell
    w: width of arr
    """
    n = coords.shape[0]
    for p in range(1, w):
        ok = True
        for k in range(n):
            i = coords[k, 0]
            j = coords[k, 1]
            if j >= p:
                # Require that the cell shifted left by p matches its own color
                if arr[i, j - p] != arr[i, j]:
                    ok = False
                    break
        if ok:
            return p
    return w


def hperiod(obj: Object) -> int:
    """horizontal periodicity"""
    # Normalize so upper-left corner is at (0,0)
    norm = normalize(obj)
    min_i, min_j = ulcorner(norm)

    # Pack into parallel arrays
    n = len(norm)
    vals = np.empty(n, dtype=np.int64)
    coords = np.empty((n, 2), dtype=np.int64)
    for k, (v, (i, j)) in enumerate(norm):
        vals[k] = v
        coords[k, 0] = i - min_i
        coords[k, 1] = j - min_j

    # Build a small dense array of shape (h, w)
    h = coords[:, 0].max() + 1
    w = coords[:, 1].max() + 1
    # Pick a sentinel that can’t clash with any real color
    sentinel = vals.min() - 1
    arr = np.full((h, w), sentinel, np.int64)
    for k in range(n):
        arr[coords[k, 0], coords[k, 1]] = vals[k]

    return int(_hperiod_core(arr, coords, w))


@njit(cache=True)
def _vperiod_kernel(arr: np.ndarray, h: int, w: int) -> int:
    for p in range(1, h):
        ok = True
        for i in range(h - p):
            for j in range(w):
                if arr[i, j] != arr[i + p, j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return p
    return h


def vperiod(obj: Object) -> int:
    """vertical periodicity"""
    if not obj:
        return 0
    norm = normalize(obj)
    if not norm:
        return 0

    h = height(norm)
    w = width(norm)
    if h <= 1:
        return h

    # Pack values and coords
    m = len(norm)
    vals = np.empty(m, dtype=np.int64)
    coords = np.empty((m, 2), dtype=np.int64)
    for k, (v, (i, j)) in enumerate(norm):
        vals[k] = v
        coords[k, 0] = i
        coords[k, 1] = j

    # Build dense array with sentinel for empty cells
    sentinel = vals.min() - 1
    arr = np.full((h, w), sentinel, np.int64)
    for k in range(m):
        arr[coords[k, 0], coords[k, 1]] = vals[k]

    return int(_vperiod_kernel(arr, h, w))
