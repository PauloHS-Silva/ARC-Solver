from __future__ import annotations

from typing import Any, Callable, Tuple, FrozenSet, Container
from abc import ABC, abstractmethod
from dataclasses import dataclass
from geneticengine.grammar import extract_grammar
from geneticengine.prelude import abstract
from geneticengine.grammar.decorators import weight
import dsl


@abstract
class Expression(ABC):
    """Base class for all expressions in the DSL."""

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        pass

    def __str__(self) -> str:
        args = ", ".join([str(v) for v in self.__dict__.values()])
        return f"{self.__class__.__name__}({args})"


@abstract
class NumericalExpression(Expression):
    def evaluate(self, *args, **kwargs) -> dsl.Numerical: ...


@abstract
class BooleanExpression(Expression):
    def evaluate(self, *args, **kwargs) -> bool: ...


@abstract
class GridExpression(Expression):
    def evaluate(self, *args, **kwargs) -> dsl.Grid: ...


@abstract
class ObjectExpression(Expression):
    def evaluate(self, *args, **kwargs) -> dsl.Object: ...


@abstract
class ObjectsExpression(Expression):
    def evaluate(self, *args, **kwargs) -> dsl.Objects: ...


@abstract
class IndicesExpression(Expression):
    def evaluate(self, *args, **kwargs) -> dsl.Indices: ...


@abstract
class IntegerExpression(NumericalExpression):
    def evaluate(self, *args, **kwargs) -> int: ...


@abstract
class ContainerExpression(Expression):
    """An expression that evaluates to a Container (e.g., Tuple, FrozenSet)."""

    def evaluate(self, *args, **kwargs) -> Container: ...


@abstract
class TupleExpression(ContainerExpression):
    """An expression that evaluates to a Tuple."""

    def evaluate(self, *args, **kwargs) -> Tuple: ...


@abstract
class FrozenSetExpression(ContainerExpression):
    """An expression that evaluates to a FrozenSet."""

    def evaluate(self, *args, **kwargs) -> FrozenSet: ...


@abstract
class CallableExpression(Expression):
    """An expression that evaluates to a callable function."""

    def evaluate(self, *args, **kwargs) -> dsl.Callable: ...


@abstract
class PatchExpression(Expression):
    """An expression that evaluates to a Patch (Object or Indices)."""

    def evaluate(self, *args, **kwargs) -> dsl.Patch: ...


@abstract
class ElementExpression(Expression):
    """An expression that evaluates to an Element (Object or Grid)."""

    def evaluate(self, *args, **kwargs) -> dsl.Element: ...


@abstract
class PieceExpression(Expression):
    """An expression that evaluates to a Piece (Grid or Patch)."""

    def evaluate(self, *args, **kwargs) -> dsl.Piece: ...


# ======================================================================================
# Terminal Expressions
# ======================================================================================


@dataclass(unsafe_hash=True)
class Constant(IntegerExpression):
    value: int

    def evaluate(self, *args, **kwargs) -> int:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(unsafe_hash=True)
@weight(970)
class InputGrid(GridExpression):
    """A typed input node specifically for the ARC task grid."""

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return kwargs.get("input_grid", tuple())

    def __str__(self) -> str:
        return "input_grid"


# ======================================================================================
# DSL Functions as Classes
# ======================================================================================


@dataclass(unsafe_hash=True)
@weight(72)
class Identity(Expression):
    x: Expression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.identity(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(51)
class AddInteger(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.add(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(51)
class AddTuple(TupleExpression):
    left: TupleExpression
    right: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.add(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(59)
class SubtractInteger(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.subtract(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(59)
class SubtractTuple(TupleExpression):
    left: TupleExpression
    right: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.subtract(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(49)
class MultiplyInteger(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.multiply(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(49)
class MultiplyTuple(TupleExpression):
    left: TupleExpression
    right: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.multiply(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(7)
class DivideInteger(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.divide(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(7)
class DivideTuple(TupleExpression):
    left: TupleExpression
    right: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.divide(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(20)
class InvertInteger(IntegerExpression):
    n: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.invert(self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(20)
class InvertTuple(TupleExpression):
    n: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.invert(self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(26)
class Even(BooleanExpression):
    n: IntegerExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.even(self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(18)
class DoubleInteger(IntegerExpression):
    n: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.double(self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(18)
class DoubleTuple(TupleExpression):
    n: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.double(self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(9)
class HalveInteger(IntegerExpression):
    n: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.halve(self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(9)
class HalveTuple(TupleExpression):
    n: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.halve(self.n.evaluate(**kwargs))


@weight(25)
@dataclass(unsafe_hash=True)
class Flip(BooleanExpression):
    b: BooleanExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.flip(self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(44)
class Equality(BooleanExpression):
    left: Expression
    right: Expression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.equality(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(26)
class Contained(BooleanExpression):
    """Checks if a value is in a container."""

    value: Expression
    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.contained(
            self.value.evaluate(**kwargs), self.container.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(60)
class Combine(ContainerExpression):
    """Combines two containers."""

    a: ContainerExpression
    b: ContainerExpression

    def evaluate(self, *args, **kwargs) -> Container:
        return dsl.combine(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(35)
class Intersection(FrozenSetExpression):
    """Calculates the intersection of two frozensets."""

    a: FrozenSetExpression
    b: FrozenSetExpression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.intersection(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(59)
class Difference(FrozenSetExpression):
    """Calculates the difference between two frozensets."""

    a: FrozenSetExpression
    b: FrozenSetExpression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.difference(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(8)
class Dedupe(TupleExpression):
    """Removes duplicates from a tuple."""

    tup: TupleExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.dedupe(self.tup.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(37)
class Order(TupleExpression):
    """Orders a container using a key function."""

    container: ContainerExpression
    compfunc: CallableExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.order(
            self.container.evaluate(**kwargs), self.compfunc.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(9)
class Repeat(TupleExpression):
    """Repeats an item N times to create a tuple."""

    item: Expression
    num: IntegerExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.repeat(self.item.evaluate(**kwargs), self.num.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(31)
class Greater(BooleanExpression):
    """Checks if a > b."""

    a: IntegerExpression
    b: IntegerExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.greater(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(141)
class Size(IntegerExpression):
    """Returns the size (length) of a container."""

    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.size(self.container.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(78)
class Merge(ContainerExpression):
    """Merges a container of containers into a single container."""

    containers: ContainerExpression

    def evaluate(self, *args, **kwargs) -> Container:
        return dsl.merge(self.containers.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(11)
class Maximum(IntegerExpression):
    """Finds the maximum value in a set of integers."""

    container: FrozenSetExpression  # Specifically IntegerSet

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.maximum(self.container.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(6)
class Minimum(IntegerExpression):
    """Finds the minimum value in a set of integers."""

    container: FrozenSetExpression  # Specifically IntegerSet

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.minimum(self.container.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(8)
class Valmax(IntegerExpression):
    """Finds the maximum value of a container according to a key function."""

    container: ContainerExpression
    compfunc: CallableExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.valmax(
            self.container.evaluate(**kwargs), self.compfunc.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(4)
class Valmin(IntegerExpression):
    """Finds the minimum value of a container according to a key function."""

    container: ContainerExpression
    compfunc: CallableExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.valmin(
            self.container.evaluate(**kwargs), self.compfunc.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(64)
class Argmax(Expression):
    """Finds the item in a container that maximizes a key function."""

    container: ContainerExpression
    compfunc: CallableExpression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.argmax(
            self.container.evaluate(**kwargs), self.compfunc.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(32)
class Argmin(Expression):
    """Finds the item in a container that minimizes a key function."""

    container: ContainerExpression
    compfunc: CallableExpression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.argmin(
            self.container.evaluate(**kwargs), self.compfunc.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(4)
class MostCommon(Expression):
    """Finds the most common item in a container."""

    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.mostcommon(self.container.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(3)
class LeastCommon(Expression):
    """Finds the least common item in a container."""

    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.leastcommon(self.container.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(44)
class Initset(FrozenSetExpression):
    """Initializes a frozenset with a single value."""

    value: Expression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.initset(self.value.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(9)
class Both(BooleanExpression):
    """Logical AND for two boolean expressions."""

    a: BooleanExpression
    b: BooleanExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.both(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(13)
class Either(BooleanExpression):
    """Logical OR for two boolean expressions."""

    a: BooleanExpression
    b: BooleanExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.either(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(18)
class IncrementInteger(IntegerExpression):
    """Increments an integer value by 1."""

    x: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.increment(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(18)
class IncrementTuple(TupleExpression):
    x: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.increment(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(39)
class DecrementInteger(IntegerExpression):
    """Decrements an integer value by 1."""

    x: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.decrement(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(39)
class DecrementTuple(TupleExpression):
    x: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.decrement(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(63)
class CrementInteger(IntegerExpression):
    """Increments positive values, decrements negative values."""

    x: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.crement(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(63)
class CrementTuple(TupleExpression):
    x: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.crement(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(3)
class SignInteger(IntegerExpression):
    """Returns the sign of an integer value."""

    x: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.sign(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(9)
class SignTuple(TupleExpression):
    x: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.sign(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(9)
class Positive(BooleanExpression):
    """Checks if an integer is positive."""

    x: IntegerExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.positive(self.x.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(20)
class Toivec(TupleExpression):
    """Creates a vertical vector (i, 0)."""

    i: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.toivec(self.i.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(24)
class Tojvec(TupleExpression):
    """Creates a horizontal vector (0, j)."""

    j: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.tojvec(self.j.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(101)
class Sfilter(ContainerExpression):
    """Filters a container based on a condition."""

    container: ContainerExpression
    condition: CallableExpression

    def evaluate(self, *args, **kwargs) -> Container:
        return dsl.sfilter(
            self.container.evaluate(**kwargs), self.condition.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(40)
class Mfilter(FrozenSetExpression):
    """Filters a container of containers and merges the result."""

    container: ContainerExpression
    function: CallableExpression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.mfilter(
            self.container.evaluate(**kwargs), self.function.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(31)
class Extract(Expression):
    """Extracts the first element that satisfies a condition."""

    container: ContainerExpression
    condition: CallableExpression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.extract(
            self.container.evaluate(**kwargs), self.condition.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(11)
class Totuple(TupleExpression):
    """Converts a frozenset to a tuple."""

    container: FrozenSetExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.totuple(self.container.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(168)
class First(Expression):
    """Gets the first item of a container."""

    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.first(self.container.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(82)
class Last(Expression):
    """Gets the last item of a container."""

    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.last(self.container.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(35)
class Insert(FrozenSetExpression):
    """Inserts an item into a frozenset."""

    value: Expression
    container: FrozenSetExpression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.insert(
            self.value.evaluate(**kwargs), self.container.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(35)
class Remove(ContainerExpression):
    """Removes an item from a container."""

    value: Expression
    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> Container:
        return dsl.remove(
            self.value.evaluate(**kwargs), self.container.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(17)
class Other(Expression):
    """Gets the other value in a two-element container."""

    container: ContainerExpression
    value: Expression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.other(
            self.container.evaluate(**kwargs), self.value.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(40)
class Interval(TupleExpression):
    """Creates a tuple representing a range."""

    start: IntegerExpression
    stop: IntegerExpression
    step: IntegerExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.interval(
            self.start.evaluate(**kwargs),
            self.stop.evaluate(**kwargs),
            self.step.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(128)
class Astuple(TupleExpression):
    """Constructs a tuple from two integers."""

    a: IntegerExpression
    b: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.astuple(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(20)
class Product(FrozenSetExpression):
    """Creates the Cartesian product of two containers."""

    a: ContainerExpression
    b: ContainerExpression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.product(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(23)
class Pair(TupleExpression):
    """Zips two tuples together."""

    a: TupleExpression
    b: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.TupleTuple:
        return dsl.pair(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(80)
class Branch(Expression):
    """If-else branching."""

    condition: BooleanExpression
    a: Expression
    b: Expression

    def evaluate(self, *args, **kwargs) -> Any:
        return dsl.branch(
            self.condition.evaluate(**kwargs),
            self.a.evaluate(**kwargs),
            self.b.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(293)
class Compose(CallableExpression):
    """Function composition: outer(inner(x))."""

    outer: CallableExpression
    inner: CallableExpression

    def evaluate(self, *args, **kwargs) -> Callable:
        return dsl.compose(self.outer.evaluate(**kwargs), self.inner.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(129)
class Chain(CallableExpression):
    """Three-function composition: h(g(f(x)))."""

    h: CallableExpression
    g: CallableExpression
    f: CallableExpression

    def evaluate(self, *args, **kwargs) -> Callable:
        return dsl.chain(
            self.h.evaluate(**kwargs),
            self.g.evaluate(**kwargs),
            self.f.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(55)
class Matcher(CallableExpression):
    """Creates an equality checking function: lambda x: function(x) == target."""

    function: CallableExpression
    target: Expression

    def evaluate(self, *args, **kwargs) -> Callable:
        return dsl.matcher(
            self.function.evaluate(**kwargs), self.target.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(231)
class Rbind(CallableExpression):
    """Fixes the rightmost argument of a function."""

    function: CallableExpression
    fixed: Expression

    def evaluate(self, *args, **kwargs) -> Callable:
        return dsl.rbind(
            self.function.evaluate(**kwargs), self.fixed.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(263)
class Lbind(CallableExpression):
    """Fixes the leftmost argument of a function."""

    function: CallableExpression
    fixed: Expression

    def evaluate(self, *args, **kwargs) -> Callable:
        return dsl.lbind(
            self.function.evaluate(**kwargs), self.fixed.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(21)
class Power(CallableExpression):
    """Applies a function N times."""

    function: CallableExpression
    n: IntegerExpression

    def evaluate(self, *args, **kwargs) -> Callable:
        return dsl.power(self.function.evaluate(**kwargs), self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(279)
class Fork(CallableExpression):
    """Creates a wrapper: lambda x: outer(a(x), b(x))."""

    outer: CallableExpression
    a: CallableExpression
    b: CallableExpression

    def evaluate(self, *args, **kwargs) -> Callable:
        return dsl.fork(
            self.outer.evaluate(**kwargs),
            self.a.evaluate(**kwargs),
            self.b.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(447)
class Apply(ContainerExpression):
    """Applies a function to each item in a container."""

    function: CallableExpression
    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> Container:
        return dsl.apply(
            self.function.evaluate(**kwargs), self.container.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(19)
class Rapply(ContainerExpression):
    """Applies each function in a container to a single value."""

    functions: ContainerExpression
    value: Expression

    def evaluate(self, *args, **kwargs) -> Container:
        return dsl.rapply(
            self.functions.evaluate(**kwargs), self.value.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(208)
class Mapply(FrozenSetExpression):
    """Applies a function to a container of containers and merges the result."""

    function: CallableExpression
    container: ContainerExpression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.mapply(
            self.function.evaluate(**kwargs), self.container.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(27)
class Papply(TupleExpression):
    """Applies a function element-wise to two tuples."""

    function: CallableExpression
    a: TupleExpression
    b: TupleExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.papply(
            self.function.evaluate(**kwargs),
            self.a.evaluate(**kwargs),
            self.b.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(208)
class Mpapply(TupleExpression):
    """Applies a function element-wise to two tuples and merges the results."""

    function: CallableExpression
    a: TupleExpression
    b: TupleExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.mpapply(
            self.function.evaluate(**kwargs),
            self.a.evaluate(**kwargs),
            self.b.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(12)
class Prapply(FrozenSetExpression):
    """Applies a function to the Cartesian product of two containers."""

    function: CallableExpression
    a: ContainerExpression
    b: ContainerExpression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.prapply(
            self.function.evaluate(**kwargs),
            self.a.evaluate(**kwargs),
            self.b.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(23)
class Mostcolor(IntegerExpression):
    """Finds the most common color in an Element."""

    element: ElementExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.mostcolor(self.element.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(45)
class Leastcolor(IntegerExpression):
    """Finds the least common color in an Element."""

    element: ElementExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.leastcolor(self.element.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(43)
class Height(IntegerExpression):
    """Calculates the height of a Piece."""

    piece: PieceExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.height(self.piece.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(50)
class Width(IntegerExpression):
    """Calculates the width of a Piece."""

    piece: PieceExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.width(self.piece.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(30)
class Shape(TupleExpression):
    """Gets the (height, width) of a Piece."""

    piece: PieceExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.shape(self.piece.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(12)
class Portrait(BooleanExpression):
    """Checks if a Piece is taller than it is wide."""

    piece: PieceExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.portrait(self.piece.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(22)
class Colorcount(IntegerExpression):
    """Counts cells of a specific color in an Element."""

    element: ElementExpression
    value: IntegerExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.colorcount(
            self.element.evaluate(**kwargs), self.value.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(63)
class Colorfilter(ObjectsExpression):
    """Filters objects by color."""

    objs: ObjectsExpression
    value: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Objects:
        return dsl.colorfilter(
            self.objs.evaluate(**kwargs), self.value.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(29)
class Sizefilter(FrozenSetExpression):
    """Filters items in a container by their size."""

    container: ContainerExpression
    n: IntegerExpression

    def evaluate(self, *args, **kwargs) -> FrozenSet:
        return dsl.sizefilter(
            self.container.evaluate(**kwargs), self.n.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(21)
class Asindices(IndicesExpression):
    """Gets the indices of all cells in a grid."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.asindices(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(183)
class Ofcolor(IndicesExpression):
    """Gets indices of cells with a specific color in a grid."""

    grid: GridExpression
    value: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.ofcolor(self.grid.evaluate(**kwargs), self.value.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(58)
class Ulcorner(TupleExpression):
    """Gets the upper-left corner index of a Patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.ulcorner(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(16)
class Urcorner(TupleExpression):
    """Gets the upper-right corner index of a Patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.urcorner(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(9)
class Llcorner(TupleExpression):
    """Gets the lower-left corner index of a Patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.llcorner(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(15)
class Lrcorner(TupleExpression):
    """Gets the lower-right corner index of a Patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.lrcorner(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(42)
class Crop(GridExpression):
    """Crops a subgrid from a grid."""

    grid: GridExpression
    start: TupleExpression
    dims: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.crop(
            self.grid.evaluate(**kwargs),
            self.start.evaluate(**kwargs),
            self.dims.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(36)
class Toindices(IndicesExpression):
    """Converts a Patch to a set of indices."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.toindices(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(61)
class Recolor(ObjectExpression):
    """Recolors a patch to a new uniform color."""

    value: IntegerExpression
    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Object:
        return dsl.recolor(self.value.evaluate(**kwargs), self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(153)
class Shift(PatchExpression):
    """Shifts a patch by a given vector."""

    patch: PatchExpression
    directions: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Patch:
        return dsl.shift(
            self.patch.evaluate(**kwargs), self.directions.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(48)
class Normalize(PatchExpression):
    """Moves a patch's upper-left corner to the origin (0,0)."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Patch:
        return dsl.normalize(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(15)
class Dneighbors(IndicesExpression):
    """Gets the four directly adjacent (cardinal) neighbors of a location."""

    loc: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.dneighbors(self.loc.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(5)
class Ineighbors(IndicesExpression):
    """Gets the four diagonally adjacent neighbors of a location."""

    loc: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.ineighbors(self.loc.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(53)
class Neighbors(IndicesExpression):
    """Gets all eight adjacent neighbors of a location."""

    loc: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.neighbors(self.loc.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(237)
class ObjectsFromGrid(ObjectsExpression):
    """Extracts objects from a grid based on connectivity rules."""

    grid: GridExpression
    univalued: BooleanExpression
    diagonal: BooleanExpression
    without_bg: BooleanExpression

    def evaluate(self, *args, **kwargs) -> dsl.Objects:
        return dsl.objects(
            self.grid.evaluate(**kwargs),
            self.univalued.evaluate(**kwargs),
            self.diagonal.evaluate(**kwargs),
            self.without_bg.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(26)
class Partition(ObjectsExpression):
    """Partitions a grid into objects based on color, including background."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Objects:
        return dsl.partition(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(16)
class Fgpartition(ObjectsExpression):
    """Partitions a grid into objects based on color, excluding background."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Objects:
        return dsl.fgpartition(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(23)
class Uppermost(IntegerExpression):
    """Gets the minimum row index of a patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.uppermost(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(7)
class Lowermost(IntegerExpression):
    """Gets the maximum row index of a patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.lowermost(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(18)
class Leftmost(IntegerExpression):
    """Gets the minimum column index of a patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.leftmost(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(6)
class Rightmost(IntegerExpression):
    """Gets the maximum column index of a patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.rightmost(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(5)
class Square(BooleanExpression):
    """Checks if a piece is a square."""

    piece: PieceExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.square(self.piece.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(18)
class Vline(BooleanExpression):
    """Checks if a patch is a vertical line."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.vline(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(16)
class Hline(BooleanExpression):
    """Checks if a patch is a horizontal line."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.hline(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(4)
class Hmatching(BooleanExpression):
    """Checks if two patches share any row indices."""

    a: PatchExpression
    b: PatchExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.hmatching(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(7)
class Vmatching(BooleanExpression):
    """Checks if two patches share any column indices."""

    a: PatchExpression
    b: PatchExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.vmatching(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(11)
class Manhattan(IntegerExpression):
    """Calculates the closest Manhattan distance between two patches."""

    a: PatchExpression
    b: PatchExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.manhattan(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(10)
class Adjacent(BooleanExpression):
    """Checks if two patches are cardinally adjacent."""

    a: PatchExpression
    b: PatchExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.adjacent(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(10)
class Bordering(BooleanExpression):
    """Checks if a patch touches the border of a grid."""

    patch: PatchExpression
    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> bool:
        return dsl.bordering(
            self.patch.evaluate(**kwargs), self.grid.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(3)
class Centerofmass(TupleExpression):
    """Calculates the center of mass of a patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.centerofmass(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(32)
class Palette(FrozenSetExpression):
    """Gets the set of colors in an Element."""

    element: ElementExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.palette(self.element.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(23)
class Numcolors(IntegerExpression):
    """Counts the number of unique colors in an Element."""

    element: ElementExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.numcolors(self.element.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(495)
class Color(IntegerExpression):
    """Gets the color of a univalued object."""

    obj: ObjectExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.color(self.obj.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(24)
class Toobject(ObjectExpression):
    """Creates an object from a patch using colors from a grid."""

    patch: PatchExpression
    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Object:
        return dsl.toobject(self.patch.evaluate(**kwargs), self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(31)
class Asobject(ObjectExpression):
    """Converts an entire grid into a single object."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Object:
        return dsl.asobject(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(24)
class Rot90(GridExpression):
    """Rotates a grid 90 degrees clockwise."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.rot90(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(14)
class Rot180(GridExpression):
    """Rotates a grid 180 degrees."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.rot180(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(15)
class Rot270(GridExpression):
    """Rotates a grid 270 degrees clockwise."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.rot270(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(39)
class HmirrorGrid(GridExpression):
    """Mirrors a grid horizontally."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.hmirror(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(39)
class HmirrorPatch(PatchExpression):
    """Mirrors a patch horizontally."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Patch:
        return dsl.hmirror(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(47)
class VmirrorGrid(GridExpression):
    """Mirrors a grid vertically."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.vmirror(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(47)
class VmirrorPatch(PatchExpression):
    """Mirrors a patch vertically."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Patch:
        return dsl.vmirror(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(35)
class DmirrorGrid(GridExpression):
    """Mirrors a grid along the main diagonal."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.dmirror(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(35)
class DmirrorPatch(PatchExpression):
    """Mirrors a patch along the main diagonal."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Patch:
        return dsl.dmirror(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(13)
class CmirrorGrid(GridExpression):
    """Mirrors a grid along the anti-diagonal."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.cmirror(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(13)
class CmirrorPatch(PatchExpression):
    """Mirrors a patch along the anti-diagonal."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Patch:
        return dsl.cmirror(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(287)
class Fill(GridExpression):
    """Fills a patch of a grid with a uniform color."""

    grid: GridExpression
    value: IntegerExpression
    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.fill(
            self.grid.evaluate(**kwargs),
            self.value.evaluate(**kwargs),
            self.patch.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(148)
class Paint(GridExpression):
    """Paints an object onto a grid."""

    grid: GridExpression
    obj: ObjectExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.paint(self.grid.evaluate(**kwargs), self.obj.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(35)
class Underfill(GridExpression):
    """Fills a patch of a grid with a color, but only on background cells."""

    grid: GridExpression
    value: IntegerExpression
    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.underfill(
            self.grid.evaluate(**kwargs),
            self.value.evaluate(**kwargs),
            self.patch.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(3)
class Underpaint(GridExpression):
    """Paints an object onto a grid, but only on background cells."""

    grid: GridExpression
    obj: ObjectExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.underpaint(self.grid.evaluate(**kwargs), self.obj.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(8)
class Hupscale(GridExpression):
    """Upscales a grid horizontally by a given factor."""

    grid: GridExpression
    factor: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.hupscale(
            self.grid.evaluate(**kwargs), self.factor.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(5)
class Vupscale(GridExpression):
    """Upscales a grid vertically by a given factor."""

    grid: GridExpression
    factor: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.vupscale(
            self.grid.evaluate(**kwargs), self.factor.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(42)
class UpscaleGrid(GridExpression):
    """Upscales a grid by a given factor."""

    grid: GridExpression
    factor: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.upscale(self.grid.evaluate(**kwargs), self.factor.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(42)
class UpscaleObject(ObjectExpression):
    """Upscales an object by a given factor."""

    object: ObjectExpression
    factor: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Object:
        return dsl.upscale(
            self.object.evaluate(**kwargs), self.factor.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(9)
class Downscale(GridExpression):
    """Downscales a grid by a given factor."""

    grid: GridExpression
    factor: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.downscale(
            self.grid.evaluate(**kwargs), self.factor.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(38)
class Hconcat(GridExpression):
    """Concatenates two grids horizontally."""

    a: GridExpression
    b: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.hconcat(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(47)
class Vconcat(GridExpression):
    """Concatenates two grids vertically."""

    a: GridExpression
    b: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.vconcat(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(66)
class Subgrid(GridExpression):
    """Extracts the smallest subgrid containing a patch."""

    patch: PatchExpression
    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.subgrid(self.patch.evaluate(**kwargs), self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(13)
class Hsplit(TupleExpression):
    """Splits a grid into N horizontal subgrids."""

    grid: GridExpression
    n: IntegerExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.hsplit(self.grid.evaluate(**kwargs), self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(10)
class Vsplit(TupleExpression):
    """Splits a grid into N vertical subgrids."""

    grid: GridExpression
    n: IntegerExpression

    def evaluate(self, *args, **kwargs) -> Tuple:
        return dsl.vsplit(self.grid.evaluate(**kwargs), self.n.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(6)
class Cellwise(GridExpression):
    """Compares two grids cell by cell, keeping matched values."""

    a: GridExpression
    b: GridExpression
    fallback: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.cellwise(
            self.a.evaluate(**kwargs),
            self.b.evaluate(**kwargs),
            self.fallback.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(64)
class Replace(GridExpression):
    """Replaces all occurrences of one color with another."""

    grid: GridExpression
    replacee: IntegerExpression
    replacer: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.replace(
            self.grid.evaluate(**kwargs),
            self.replacee.evaluate(**kwargs),
            self.replacer.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(12)
class Switch(GridExpression):
    """Swaps two colors in a grid."""

    grid: GridExpression
    a: IntegerExpression
    b: IntegerExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.switch(
            self.grid.evaluate(**kwargs),
            self.a.evaluate(**kwargs),
            self.b.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(47)
class Center(TupleExpression):
    """Calculates the geometric center of a patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.center(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(7)
class Position(TupleExpression):
    """Calculates the relative position vector between two patches."""

    a: PatchExpression
    b: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.position(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(14)
class Index(IntegerExpression):
    """Gets the color at a specific location in a grid."""

    grid: GridExpression
    loc: TupleExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.index(self.grid.evaluate(**kwargs), self.loc.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(64)
class Canvas(GridExpression):
    """Creates a new grid of a given size and color."""

    value: IntegerExpression
    dimensions: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.canvas(
            self.value.evaluate(**kwargs), self.dimensions.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(9)
class Corners(IndicesExpression):
    """Gets the four corner indices of a patch's bounding box."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.corners(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(42)
class Connect(IndicesExpression):
    """Draws a line between two points."""

    a: TupleExpression
    b: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.connect(self.a.evaluate(**kwargs), self.b.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(23)
class Cover(GridExpression):
    """Removes a patch from a grid by filling it with the background color."""

    grid: GridExpression
    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.cover(self.grid.evaluate(**kwargs), self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(7)
class Trim(GridExpression):
    """Trims the outermost border of a grid."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.trim(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(46)
class Move(GridExpression):
    """Moves an object on a grid by a given offset."""

    grid: GridExpression
    obj: ObjectExpression
    offset: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.move(
            self.grid.evaluate(**kwargs),
            self.obj.evaluate(**kwargs),
            self.offset.evaluate(**kwargs),
        )


@dataclass(unsafe_hash=True)
@weight(19)
class Tophalf(GridExpression):
    """Gets the top half of a grid."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.tophalf(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(16)
class Bottomhalf(GridExpression):
    """Gets the bottom half of a grid."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.bottomhalf(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(16)
class Lefthalf(GridExpression):
    """Gets the left half of a grid."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.lefthalf(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(12)
class Righthalf(GridExpression):
    """Gets the right half of a grid."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.righthalf(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(18)
class Vfrontier(IndicesExpression):
    """Gets the vertical line passing through a location."""

    location: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.vfrontier(self.location.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(17)
class Hfrontier(IndicesExpression):
    """Gets the horizontal line passing through a location."""

    location: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.hfrontier(self.location.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(17)
class Backdrop(IndicesExpression):
    """Gets all indices within the bounding box of a patch."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.backdrop(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(14)
class Delta(IndicesExpression):
    """Gets indices in the bounding box but not in the patch itself."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.delta(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(11)
class Gravitate(TupleExpression):
    """Calculates the vector to move a source patch until adjacent to a destination."""

    source: PatchExpression
    destination: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return dsl.gravitate(
            self.source.evaluate(**kwargs), self.destination.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(12)
class Inbox(IndicesExpression):
    """Gets the inner box one step inside a patch's bounding box."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.inbox(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(26)
class Outbox(IndicesExpression):
    """Gets the outer box one step outside a patch's bounding box."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.outbox(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(53)
class Box(IndicesExpression):
    """Gets the outline of a patch's bounding box."""

    patch: PatchExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.box(self.patch.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(59)
class Shoot(IndicesExpression):
    """Projects a line from a start point in a given direction."""

    start: TupleExpression
    direction: TupleExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.shoot(
            self.start.evaluate(**kwargs), self.direction.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(21)
class Occurrences(IndicesExpression):
    """Finds all locations of an object within a grid."""

    grid: GridExpression
    obj: ObjectExpression

    def evaluate(self, *args, **kwargs) -> dsl.Indices:
        return dsl.occurrences(
            self.grid.evaluate(**kwargs), self.obj.evaluate(**kwargs)
        )


@dataclass(unsafe_hash=True)
@weight(4)
class Frontiers(ObjectsExpression):
    """Finds all single-colored horizontal and vertical lines in a grid."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Objects:
        return dsl.frontiers(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(6)
class Compress(GridExpression):
    """Removes all single-colored rows and columns (frontiers) from a grid."""

    grid: GridExpression

    def evaluate(self, *args, **kwargs) -> dsl.Grid:
        return dsl.compress(self.grid.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(7)
class Hperiod(IntegerExpression):
    """Calculates the horizontal periodicity of an object."""

    obj: ObjectExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.hperiod(self.obj.evaluate(**kwargs))


@dataclass(unsafe_hash=True)
@weight(6)
class Vperiod(IntegerExpression):
    """Calculates the vertical periodicity of an object."""

    obj: ObjectExpression

    def evaluate(self, *args, **kwargs) -> int:
        return dsl.vperiod(self.obj.evaluate(**kwargs))


# ======================================================================================
# Constant Classes
# ======================================================================================


@dataclass(unsafe_hash=True)
@weight(257)
class FalseConstant(BooleanExpression):
    def evaluate(self, *args, **kwargs) -> bool:
        return False

    def __str__(self):
        return "False"


@dataclass(unsafe_hash=True)
@weight(454)
class TrueConstant(BooleanExpression):
    def evaluate(self, *args, **kwargs) -> bool:
        return True

    def __str__(self):
        return "True"


@dataclass(unsafe_hash=True)
@weight(230)
class ZeroConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 0

    def __str__(self):
        return "0"


@dataclass(unsafe_hash=True)
@weight(186)
class OneConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 1

    def __str__(self):
        return "1"


@dataclass(unsafe_hash=True)
@weight(196)
class TwoConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 2

    def __str__(self):
        return "2"


@dataclass(unsafe_hash=True)
@weight(133)
class ThreeConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 3

    def __str__(self):
        return "3"


@dataclass(unsafe_hash=True)
@weight(82)
class FourConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 4

    def __str__(self):
        return "4"


@dataclass(unsafe_hash=True)
@weight(83)
class FiveConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 5

    def __str__(self):
        return "5"


@dataclass(unsafe_hash=True)
@weight(28)
class SixConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 6

    def __str__(self):
        return "6"


@dataclass(unsafe_hash=True)
@weight(17)
class SevenConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 7

    def __str__(self):
        return "7"


@dataclass(unsafe_hash=True)
@weight(78)
class EightConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 8

    def __str__(self):
        return "8"


@dataclass(unsafe_hash=True)
@weight(31)
class NineConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 9

    def __str__(self):
        return "9"


@dataclass(unsafe_hash=True)
@weight(14)
class TenConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return 10

    def __str__(self):
        return "10"


@dataclass(unsafe_hash=True)
@weight(10)
class NegOneConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return -1

    def __str__(self):
        return "-1"


@dataclass(unsafe_hash=True)
@weight(8)
class NegTwoConstant(IntegerExpression):
    def evaluate(self, *args, **kwargs) -> int:
        return -2

    def __str__(self):
        return "-2"


@dataclass(unsafe_hash=True)
@weight(24)
class DownConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (1, 0)

    def __str__(self):
        return "(1,0)"


@dataclass(unsafe_hash=True)
@weight(13)
class RightConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (0, 1)

    def __str__(self):
        return "(0,1)"


@dataclass(unsafe_hash=True)
@weight(8)
class UpConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (-1, 0)

    def __str__(self):
        return "(-1,0)"


@dataclass(unsafe_hash=True)
@weight(6)
class LeftConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (0, -1)

    def __str__(self):
        return "(0,-1)"


@dataclass(unsafe_hash=True)
@weight(47)
class OriginConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (0, 0)

    def __str__(self):
        return "(0,0)"


@dataclass(unsafe_hash=True)
@weight(43)
class UnityConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (1, 1)

    def __str__(self):
        return "(1,1)"


@dataclass(unsafe_hash=True)
@weight(15)
class NegUnityConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (-1, -1)

    def __str__(self):
        return "(-1,-1)"


@dataclass(unsafe_hash=True)
@weight(13)
class UpRightConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (-1, 1)

    def __str__(self):
        return "(-1,1)"


@dataclass(unsafe_hash=True)
@weight(14)
class DownLeftConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (1, -1)

    def __str__(self):
        return "(1,-1)"


@dataclass(unsafe_hash=True)
@weight(9)
class ZeroByTwoConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (0, 2)

    def __str__(self):
        return "(0,2)"


@dataclass(unsafe_hash=True)
@weight(13)
class TwoByZeroConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (2, 0)

    def __str__(self):
        return "(2,0)"


@dataclass(unsafe_hash=True)
@weight(11)
class TwoByTwoConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (2, 2)

    def __str__(self):
        return "(2,2)"


@dataclass(unsafe_hash=True)
@weight(18)
class ThreeByThreeConstant(TupleExpression):
    def evaluate(self, *args, **kwargs) -> dsl.IntegerTuple:
        return (3, 3)

    def __str__(self):
        return "(3,3)"


# Create and export grammar

grammar = extract_grammar(
    [
        Expression,
        Identity,
        AddInteger,
        AddTuple,
        SubtractInteger,
        SubtractTuple,
        MultiplyInteger,
        MultiplyTuple,
        DivideInteger,
        DivideTuple,
        InvertInteger,
        InvertTuple,
        Even,
        DoubleInteger,
        DoubleTuple,
        HalveInteger,
        HalveTuple,
        Flip,
        Equality,
        Contained,
        Combine,
        Intersection,
        Difference,
        Dedupe,
        Order,
        Repeat,
        Greater,
        Size,
        Merge,
        Maximum,
        Minimum,
        Valmax,
        Valmin,
        Argmax,
        Argmin,
        MostCommon,
        LeastCommon,
        Initset,
        Both,
        Either,
        IncrementInteger,
        IncrementTuple,
        DecrementInteger,
        DecrementTuple,
        CrementInteger,
        CrementTuple,
        SignInteger,
        SignTuple,
        Positive,
        Toivec,
        Tojvec,
        Sfilter,
        Mfilter,
        Extract,
        Totuple,
        First,
        Last,
        Insert,
        Remove,
        Other,
        Interval,
        Astuple,
        Product,
        Pair,
        Branch,
        Compose,
        Chain,
        Matcher,
        Rbind,
        Lbind,
        Power,
        Fork,
        Apply,
        Rapply,
        Mapply,
        Papply,
        Mpapply,
        Prapply,
        Mostcolor,
        Leastcolor,
        Height,
        Width,
        Shape,
        Portrait,
        Colorcount,
        Colorfilter,
        Sizefilter,
        Asindices,
        Ofcolor,
        Ulcorner,
        Urcorner,
        Llcorner,
        Lrcorner,
        Crop,
        Toindices,
        Recolor,
        Shift,
        Normalize,
        Dneighbors,
        Ineighbors,
        Neighbors,
        ObjectsFromGrid,
        Partition,
        Fgpartition,
        Uppermost,
        Lowermost,
        Leftmost,
        Rightmost,
        Square,
        Vline,
        Hline,
        Hmatching,
        Vmatching,
        Manhattan,
        Adjacent,
        Bordering,
        Centerofmass,
        Palette,
        Numcolors,
        Color,
        Toobject,
        Asobject,
        Rot90,
        Rot180,
        Rot270,
        HmirrorGrid,
        VmirrorGrid,
        HmirrorPatch,
        VmirrorPatch,
        CmirrorGrid,
        CmirrorPatch,
        DmirrorGrid,
        DmirrorPatch,
        Fill,
        Paint,
        Underfill,
        Underpaint,
        Hupscale,
        Vupscale,
        # UpscaleGrid,
        # UpscaleObject,
        Downscale,
        Hconcat,
        Vconcat,
        Subgrid,
        Hsplit,
        Vsplit,
        Cellwise,
        Replace,
        Switch,
        Center,
        Position,
        Index,
        Canvas,
        Corners,
        Connect,
        Cover,
        Trim,
        Move,
        Tophalf,
        Bottomhalf,
        Lefthalf,
        Righthalf,
        Vfrontier,
        Hfrontier,
        Backdrop,
        Delta,
        Gravitate,
        Inbox,
        Outbox,
        Box,
        Shoot,
        Occurrences,
        Frontiers,
        Compress,
        Hperiod,
        Vperiod,
        FalseConstant,
        TrueConstant,
        ZeroConstant,
        OneConstant,
        TwoConstant,
        ThreeConstant,
        FourConstant,
        FiveConstant,
        SixConstant,
        SevenConstant,
        EightConstant,
        NineConstant,
        TenConstant,
        NegOneConstant,
        NegTwoConstant,
        DownConstant,
        RightConstant,
        UpConstant,
        LeftConstant,
        OriginConstant,
        UnityConstant,
        NegUnityConstant,
        UpRightConstant,
        DownLeftConstant,
        ZeroByTwoConstant,
        TwoByZeroConstant,
        TwoByTwoConstant,
        ThreeByThreeConstant,
        InputGrid,
    ],
    GridExpression,
)
