import dsl


A = ((1, 0), (0, 1), (1, 0))
B = ((2, 1), (0, 1), (2, 1))
C = ((3, 4), (5, 5))
D = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
E = ((1, 2), (4, 5))
F = ((5, 6), (8, 0))
G = (
    (1, 0, 0, 0, 3),
    (0, 1, 1, 0, 0),
    (0, 1, 1, 2, 0),
    (0, 0, 2, 2, 0),
    (0, 2, 0, 0, 0),
)
H = (
    (0, 0, 0, 0, 0),
    (0, 2, 0, 2, 0),
    (2, 0, 0, 2, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 2, 0, 0),
)
II = (
    (0, 0, 2, 0, 0),
    (0, 2, 0, 2, 0),
    (2, 0, 0, 2, 0),
    (0, 2, 0, 2, 0),
    (0, 0, 2, 0, 0),
)
J = (
    (0, 0, 2, 0, 0),
    (0, 2, 0, 2, 0),
    (0, 0, 2, 2, 0),
    (0, 2, 0, 2, 0),
    (0, 0, 2, 0, 0),
)
K = (
    (0, 0, 1, 0, 0, 1, 0, 0),
    (0, 0, 1, 0, 0, 1, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 1, 0, 0, 1, 0, 0),
    (0, 0, 1, 0, 0, 1, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 1, 0, 0, 1, 0, 0),
    (0, 0, 1, 0, 0, 1, 0, 0),
)


def test_identity():
    assert dsl.identity(1) == 1


def test_add():
    assert dsl.add(1, 2) == 3
    assert dsl.add(4, 6) == 10


def test_subtract():
    assert dsl.subtract(1, 2) == -1
    assert dsl.subtract(4, 6) == -2


def test_multiply():
    assert dsl.multiply(2, 3) == 6
    assert dsl.multiply(4, 3) == 12


def test_divide():
    assert dsl.divide(4, 2) == 2
    assert dsl.divide(9, 2) == 4


def test_invert():
    assert dsl.invert(1) == -1
    assert dsl.invert(-4) == 4


def test_even():
    assert not dsl.even(1)
    assert dsl.even(2)


def test_double():
    assert dsl.double(1) == 2


def test_halve():
    assert dsl.halve(2) == 1
    assert dsl.halve(5) == 2


def test_flip():
    assert dsl.flip(False)
    assert not dsl.flip(True)


def test_equality():
    assert dsl.equality(A, A)
    assert not dsl.equality(A, B)


def test_contained():
    assert dsl.contained(1, (1, 3))
    assert not dsl.contained(2, {3, 4})


def test_combine():
    assert dsl.combine(frozenset({1, 2}), frozenset({3, 4})) == frozenset({1, 2, 3, 4})
    assert dsl.combine((1, 2), (3, 4)) == (1, 2, 3, 4)


def test_intersection():
    assert dsl.intersection(frozenset({1, 2}), frozenset({2, 3})) == frozenset({2})


def test_difference():
    assert dsl.difference(frozenset({1, 2, 3}), frozenset({1, 2})) == frozenset({3})


def test_dedupe():
    assert dsl.dedupe((1, 2, 3, 3, 2, 4, 1)) == (1, 2, 3, 4)


def test_order():
    assert dsl.order(((1,), (1, 2, 3), (1, 2)), len) == ((1,), (1, 2), (1, 2, 3))
    assert dsl.order((1, 4, -3), abs) == (1, -3, 4)


def test_repeat():
    assert dsl.repeat(C, 3) == (C, C, C)


def test_greater():
    assert dsl.greater(2, 1)
    assert not dsl.greater(4, 10)


def test_size():
    assert dsl.size((1, 2, 3)) == 3
    assert dsl.size(frozenset({2, 5})) == 2


def test_merge():
    assert dsl.merge(
        frozenset({frozenset({(1, (0, 0))}), frozenset({(1, (1, 1)), (1, (0, 1))})})
    ) == frozenset({(1, (0, 0)), (1, (1, 1)), (1, (0, 1))})
    assert dsl.merge(((1, 2), (3, 4, 5))) == (1, 2, 3, 4, 5)
    assert dsl.merge(((4, 5), (7,))) == (4, 5, 7)


def test_maximum():
    assert dsl.maximum({1, 2, 5, 3}) == 5
    assert dsl.maximum((4, 2, 6)) == 6


def test_minimum():
    assert dsl.minimum({1, 2, 5, 3}) == 1
    assert dsl.minimum((4, 2, 6)) == 2


def test_valmax():
    assert dsl.valmax(((1,), (1, 2)), len) == 2


def test_valmin():
    assert dsl.valmin(((1,), (1, 2)), len) == 1


def test_argmax():
    assert dsl.argmax(((1,), (1, 2)), len) == (1, 2)


def test_argmin():
    assert dsl.argmin(((1,), (1, 2)), len) == (1,)


def test_mostcommon():
    assert dsl.mostcommon((1, 2, 2, 3, 3, 3)) == 3


def test_leastcommon():
    assert dsl.leastcommon((1, 2, 3, 4, 2, 3, 4)) == 1


def test_initset():
    assert dsl.initset(2) == frozenset({2})


def test_both():
    assert not dsl.both(True, False)
    assert dsl.both(True, True)
    assert not dsl.both(False, False)


def test_either():
    assert dsl.either(True, False)
    assert dsl.either(True, True)
    assert not dsl.either(False, False)


def test_increment():
    assert dsl.increment(1) == 2


def test_decrement():
    assert dsl.decrement(1) == 0


def test_crement():
    assert dsl.crement(1) == 2
    assert dsl.crement(-2) == -3


def test_sign():
    assert dsl.sign(2) == 1
    assert dsl.sign(0) == 0
    assert dsl.sign(-1) == -1


def test_positive():
    assert dsl.positive(1)
    assert not dsl.positive(-2)


def test_toivec():
    assert dsl.toivec(2) == (2, 0)


def test_tojvec():
    assert dsl.tojvec(3) == (0, 3)


def test_sfilter():
    assert dsl.sfilter((1, 2, 3), lambda x: x > 1) == (2, 3)
    assert dsl.sfilter(frozenset({2, 3, 4}), lambda x: x % 2 == 0) == frozenset({2, 4})


def test_mfilter():
    assert dsl.mfilter(
        frozenset(
            {
                frozenset({(2, (3, 3))}),
                frozenset({(1, (0, 0))}),
                frozenset({(1, (1, 1)), (1, (0, 1))}),
            }
        ),
        lambda x: len(x) == 1,
    ) == frozenset({(1, (0, 0)), (2, (3, 3))})


def test_extract():
    assert dsl.extract((1, 2, 3), lambda x: x > 2) == 3
    assert dsl.extract(frozenset({2, 3, 4}), lambda x: x % 4 == 0) == 4


def test_totuple():
    assert dsl.totuple({1}) == (1,)


def test_first():
    assert dsl.first((2, 3)) == 2


def test_last():
    assert dsl.last((2, 3)) == 3


def test_insert():
    assert dsl.insert(1, frozenset({2})) == frozenset({1, 2})


def test_remove():
    assert dsl.remove(1, frozenset({1, 2})) == frozenset({2})


def test_other():
    assert dsl.other({1, 2}, 1) == 2


def test_interval():
    assert dsl.interval(1, 4, 1) == (1, 2, 3)
    assert dsl.interval(5, 2, -1) == (5, 4, 3)


def test_astuple():
    assert dsl.astuple(3, 4) == (3, 4)


def test_product():
    assert dsl.product({1, 2}, {2, 3}) == frozenset({(1, 2), (1, 3), (2, 2), (2, 3)})


def test_pair():
    assert dsl.pair((1, 2), (4, 3)) == ((1, 4), (2, 3))


def test_branch():
    assert dsl.branch(True, 1, 3) == 1
    assert dsl.branch(False, 4, 2) == 2


def test_compose():
    assert dsl.compose(lambda x: x**2, lambda x: x + 1)(2) == 9
    assert dsl.compose(lambda x: x + 1, lambda x: x**2)(2) == 5


def test_chain():
    assert dsl.chain(lambda x: x + 3, lambda x: x**2, lambda x: x + 1)(2) == 12


def test_matcher():
    assert dsl.matcher(lambda x: x + 1, 3)(2)
    assert not dsl.matcher(lambda x: x - 1, 3)(2)


def test_rbind():
    assert dsl.rbind(lambda a, b: a + b, 2)(3) == 5
    assert dsl.rbind(lambda a, b: a == b, 2)(2)


def test_lbind():
    assert dsl.lbind(lambda a, b: a + b, 2)(3) == 5
    assert dsl.lbind(lambda a, b: a == b, 2)(2)


def test_power():
    assert dsl.power(lambda x: x + 1, 3)(4) == 7


def test_fork():
    assert dsl.fork(lambda x, y: x * y, lambda x: x + 1, lambda x: x + 2)(2) == 12


def test_apply():
    assert dsl.apply(lambda x: x**2, (1, 2, 3)) == (1, 4, 9)
    assert dsl.apply(lambda x: x % 2, frozenset({1, 2})) == frozenset({0, 1})


def test_rapply():
    assert dsl.rapply(frozenset({lambda x: x + 1, lambda x: x - 1}), 1) == {0, 2}


def test_mapply():
    assert dsl.mapply(
        lambda x: frozenset({(v + 1, (i, j)) for v, (i, j) in x}),
        frozenset({frozenset({(1, (0, 0))}), frozenset({(1, (1, 1)), (1, (0, 1))})}),
    ) == frozenset({(2, (0, 0)), (2, (1, 1)), (2, (0, 1))})


def test_papply():
    assert dsl.papply(lambda x, y: x + y, (1, 2), (3, 4)) == (4, 6)


def test_mpapply():
    assert dsl.mpapply(
        lambda x, y: frozenset({(x, (i, j)) for _, (i, j) in y}),
        (3, 4),
        frozenset({frozenset({(1, (0, 0))}), frozenset({(1, (1, 1)), (1, (0, 1))})}),
    ) == ((3, (0, 0)), (4, (1, 1)), (4, (0, 1)))


def test_prapply():
    assert dsl.prapply(lambda x, y: x + y, {1, 2}, {2, 3}) == frozenset({3, 4, 5})


def test_mostcolor():
    assert dsl.mostcolor(B) == 1
    assert dsl.mostcolor(C) == 5


def test_leastcolor():
    assert dsl.leastcolor(B) == 0


def test_height():
    assert dsl.height(A) == 3
    assert dsl.height(C) == 2
    assert dsl.height(frozenset({(0, 4)})) == 1
    assert (
        dsl.height(
            frozenset({(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))})
        )
        == 3
    )


def test_width():
    assert dsl.width(A) == 2
    assert dsl.width(C) == 2
    assert dsl.width(frozenset({(0, 4)})) == 1
    assert (
        dsl.width(
            frozenset({(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))})
        )
        == 3
    )


def test_shape():
    assert dsl.shape(A) == (3, 2)
    assert dsl.shape(C) == (2, 2)
    assert dsl.shape(frozenset({(0, 4)})) == (1, 1)
    assert dsl.shape(
        frozenset({(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))})
    ) == (3, 3)


def test_portrait():
    assert dsl.portrait(A)
    assert not dsl.portrait(C)


def test_colorcount():
    assert dsl.colorcount(A, 1) == 3
    assert dsl.colorcount(C, 5) == 2
    assert dsl.colorcount(frozenset({(1, (0, 0)), (2, (1, 0)), (2, (0, 1))}), 2) == 2
    assert dsl.colorcount(frozenset({(1, (0, 0)), (2, (1, 0)), (2, (0, 1))}), 1) == 1


def test_colorfilter():
    assert dsl.colorfilter(
        frozenset(
            {
                frozenset({(3, (0, 4))}),
                frozenset({(1, (0, 0))}),
                frozenset({(2, (4, 1))}),
                frozenset({(1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))}),
                frozenset({(2, (3, 2)), (2, (2, 3)), (2, (3, 3))}),
            }
        ),
        2,
    ) == frozenset(
        {frozenset({(2, (4, 1))}), frozenset({(2, (3, 2)), (2, (2, 3)), (2, (3, 3))})}
    )


def test_sizefilter():
    assert dsl.sizefilter(
        frozenset(
            {
                frozenset({(3, (0, 4))}),
                frozenset({(1, (0, 0))}),
                frozenset({(2, (4, 1))}),
                frozenset({(1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))}),
                frozenset({(2, (3, 2)), (2, (2, 3)), (2, (3, 3))}),
            }
        ),
        1,
    ) == frozenset(
        {frozenset({(3, (0, 4))}), frozenset({(1, (0, 0))}), frozenset({(2, (4, 1))})}
    )


def test_asindices():
    assert dsl.asindices(A) == frozenset(
        {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}
    )
    assert dsl.asindices(C) == frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})


def test_ofcolor():
    assert dsl.ofcolor(A, 0) == frozenset({(0, 1), (1, 0), (2, 1)})
    assert dsl.ofcolor(B, 2) == frozenset({(0, 0), (2, 0)})
    assert dsl.ofcolor(C, 1) == frozenset()


def test_ulcorner():
    assert dsl.ulcorner(frozenset({(1, 2), (0, 3), (4, 0)})) == (0, 0)
    assert dsl.ulcorner(frozenset({(1, 2), (0, 0), (4, 3)})) == (0, 0)


def test_urcorner():
    assert dsl.urcorner(frozenset({(1, 2), (0, 3), (4, 0)})) == (0, 3)
    assert dsl.urcorner(frozenset({(1, 2), (0, 0), (4, 3)})) == (0, 3)


def test_llcorner():
    assert dsl.llcorner(frozenset({(1, 2), (0, 3), (4, 0)})) == (4, 0)
    assert dsl.llcorner(frozenset({(1, 5), (0, 0), (2, 3)})) == (2, 0)


def test_lrcorner():
    assert dsl.lrcorner(frozenset({(1, 2), (0, 3), (4, 0)})) == (4, 3)
    assert dsl.lrcorner(frozenset({(1, 5), (0, 0), (2, 3)})) == (2, 5)


def test_crop():
    assert dsl.crop(A, (0, 0), (2, 2)) == ((1, 0), (0, 1))
    assert dsl.crop(C, (0, 1), (1, 1)) == ((4,),)
    assert dsl.crop(D, (1, 2), (2, 1)) == ((6,), (0,))


def test_toindices():
    assert dsl.toindices(frozenset({(1, (1, 1)), (1, (1, 0))})) == frozenset(
        {(1, 1), (1, 0)}
    )
    assert dsl.toindices(frozenset({(1, 1), (0, 1)})) == frozenset({(1, 1), (0, 1)})


def test_recolor():
    assert dsl.recolor(
        3, frozenset({(2, (0, 0)), (1, (0, 1)), (5, (1, 0))})
    ) == frozenset({(3, (0, 0)), (3, (0, 1)), (3, (1, 0))})
    assert dsl.recolor(2, frozenset({(2, (2, 5)), (2, (1, 1))})) == frozenset(
        {(2, (2, 5)), (2, (1, 1))}
    )


def test_shift():
    assert dsl.shift(
        frozenset({(2, (1, 1)), (4, (1, 2)), (1, (2, 3))}), (1, 2)
    ) == frozenset({(2, (2, 3)), (4, (2, 4)), (1, (3, 5))})
    assert dsl.shift(frozenset({(1, 3), (0, 2), (3, 4)}), (0, -1)) == frozenset(
        {(1, 2), (0, 1), (3, 3)}
    )


def test_normalize():
    assert dsl.normalize(
        frozenset({(2, (1, 1)), (4, (1, 2)), (1, (2, 3))})
    ) == frozenset({(2, (0, 0)), (4, (0, 1)), (1, (1, 2))})
    assert dsl.normalize(frozenset({(1, 0), (0, 2), (3, 4)})) == frozenset(
        {(1, 0), (0, 2), (3, 4)}
    )


def test_dneighbors():
    assert dsl.dneighbors((1, 1)) == frozenset({(0, 1), (1, 0), (2, 1), (1, 2)})
    assert dsl.dneighbors((0, 0)) == frozenset({(0, 1), (1, 0), (-1, 0), (0, -1)})
    assert dsl.dneighbors((0, 1)) == frozenset({(0, 0), (1, 1), (-1, 1), (0, 2)})
    assert dsl.dneighbors((1, 0)) == frozenset({(0, 0), (1, 1), (1, -1), (2, 0)})


def test_ineighbors():
    assert dsl.ineighbors((1, 1)) == frozenset({(0, 0), (0, 2), (2, 0), (2, 2)})
    assert dsl.ineighbors((0, 0)) == frozenset({(1, 1), (-1, -1), (1, -1), (-1, 1)})
    assert dsl.ineighbors((0, 1)) == frozenset({(1, 0), (1, 2), (-1, 0), (-1, 2)})
    assert dsl.ineighbors((1, 0)) == frozenset({(0, 1), (2, -1), (2, 1), (0, -1)})


def test_neighbors():
    assert dsl.neighbors((1, 1)) == frozenset(
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    )
    assert dsl.neighbors((0, 0)) == frozenset(
        {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}
    )


def test_objects():
    assert dsl.objects(G, True, False, True) == frozenset(
        {
            frozenset({(3, (0, 4))}),
            frozenset({(1, (0, 0))}),
            frozenset({(2, (4, 1))}),
            frozenset({(1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))}),
            frozenset({(2, (3, 2)), (2, (2, 3)), (2, (3, 3))}),
        }
    )
    assert dsl.objects(G, True, True, True) == frozenset(
        {
            frozenset({(3, (0, 4))}),
            frozenset(
                {(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))}
            ),
            frozenset({(2, (4, 1)), (2, (3, 2)), (2, (2, 3)), (2, (3, 3))}),
        }
    )
    assert dsl.objects(G, False, False, True) == frozenset(
        {
            frozenset({(3, (0, 4))}),
            frozenset({(1, (0, 0))}),
            frozenset({(2, (4, 1))}),
            frozenset(
                {
                    (1, (1, 1)),
                    (1, (1, 2)),
                    (1, (2, 1)),
                    (1, (2, 2)),
                    (2, (3, 2)),
                    (2, (2, 3)),
                    (2, (3, 3)),
                }
            ),
        }
    )
    assert dsl.objects(G, False, True, True) == frozenset(
        {
            frozenset({(3, (0, 4))}),
            frozenset(
                {
                    (1, (0, 0)),
                    (1, (1, 1)),
                    (1, (1, 2)),
                    (1, (2, 1)),
                    (1, (2, 2)),
                    (2, (4, 1)),
                    (2, (3, 2)),
                    (2, (2, 3)),
                    (2, (3, 3)),
                }
            ),
        }
    )
    assert dsl.objects(G, True, False, False) == frozenset(
        {
            frozenset({(3, (0, 4))}),
            frozenset({(1, (0, 0))}),
            frozenset({(2, (4, 1))}),
            frozenset({(1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))}),
            frozenset({(2, (3, 2)), (2, (2, 3)), (2, (3, 3))}),
            frozenset(
                {(0, (1, 0)), (0, (2, 0)), (0, (3, 0)), (0, (4, 0)), (0, (3, 1))}
            ),
            frozenset(
                {
                    (0, (0, 1)),
                    (0, (0, 2)),
                    (0, (0, 3)),
                    (0, (1, 3)),
                    (0, (1, 4)),
                    (0, (2, 4)),
                    (0, (3, 4)),
                    (0, (4, 4)),
                    (0, (4, 3)),
                    (0, (4, 2)),
                }
            ),
        }
    )


def test_partition():
    assert dsl.partition(B) == frozenset(
        {
            frozenset({(0, (1, 0))}),
            frozenset({(2, (0, 0)), (2, (2, 0))}),
            frozenset({(1, (0, 1)), (1, (1, 1)), (1, (2, 1))}),
        }
    )
    assert dsl.partition(G) == frozenset(
        {
            frozenset(
                {(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))}
            ),
            frozenset({(2, (4, 1)), (2, (3, 2)), (2, (2, 3)), (2, (3, 3))}),
            frozenset({(3, (0, 4))}),
            frozenset(
                {
                    (0, (0, 1)),
                    (0, (0, 2)),
                    (0, (0, 3)),
                    (0, (1, 0)),
                    (0, (1, 3)),
                    (0, (1, 4)),
                    (0, (2, 0)),
                    (0, (2, 4)),
                    (0, (3, 0)),
                    (0, (3, 1)),
                    (0, (3, 4)),
                    (0, (4, 0)),
                    (0, (4, 2)),
                    (0, (4, 3)),
                    (0, (4, 4)),
                }
            ),
        }
    )


def test_fgpartition():
    assert dsl.fgpartition(B) == frozenset(
        {frozenset({(0, (1, 0))}), frozenset({(2, (0, 0)), (2, (2, 0))})}
    )
    assert dsl.fgpartition(G) == frozenset(
        {
            frozenset(
                {(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))}
            ),
            frozenset({(2, (4, 1)), (2, (3, 2)), (2, (2, 3)), (2, (3, 3))}),
            frozenset({(3, (0, 4))}),
        }
    )


def test_uppermost():
    assert dsl.uppermost(frozenset({(0, 4)})) == 0
    assert (
        dsl.uppermost(
            frozenset({(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))})
        )
        == 0
    )


def test_lowermost():
    assert dsl.lowermost(frozenset({(0, 4)})) == 0
    assert (
        dsl.lowermost(
            frozenset({(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))})
        )
        == 2
    )


def test_leftmost():
    assert dsl.leftmost(frozenset({(0, 4)})) == 4
    assert (
        dsl.leftmost(
            frozenset({(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))})
        )
        == 0
    )


def test_rightmost():
    assert dsl.rightmost(frozenset({(0, 4)})) == 4
    assert (
        dsl.rightmost(
            frozenset({(1, (0, 0)), (1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2))})
        )
        == 2
    )


def test_square():
    assert dsl.square(C)
    assert dsl.square(D)
    assert not dsl.square(A)
    assert not dsl.square(B)
    assert not dsl.square(frozenset({(1, 1), (1, 0)}))
    assert dsl.square(frozenset({(1, 1), (0, 0), (1, 0), (0, 1)}))
    assert not dsl.square(frozenset({(0, 0), (1, 0), (0, 1)}))
    assert dsl.square(frozenset({(1, (1, 1)), (2, (0, 0)), (2, (1, 0)), (3, (0, 1))}))


def test_vline():
    assert dsl.vline(frozenset({(1, (1, 1)), (1, (0, 1))}))
    assert not dsl.vline(frozenset({(1, 1), (1, 0)}))


def test_hline():
    assert dsl.hline(frozenset({(1, (1, 1)), (1, (1, 0))}))
    assert not dsl.hline(frozenset({(1, 1), (0, 1)}))


def test_hmatching():
    assert dsl.hmatching(
        frozenset({(1, (1, 1)), (2, (0, 0)), (2, (1, 0)), (3, (0, 1))}),
        frozenset({(1, (1, 3)), (2, (1, 4))}),
    )
    assert not dsl.hmatching(
        frozenset({(1, (1, 1)), (2, (0, 0)), (2, (1, 0)), (3, (0, 1))}),
        frozenset({(1, (2, 3)), (2, (2, 4))}),
    )


def test_vmatching():
    assert dsl.vmatching(
        frozenset({(1, (1, 1)), (2, (0, 0)), (2, (1, 0)), (3, (0, 1))}),
        frozenset({(1, (3, 1)), (2, (4, 1))}),
    )
    assert not dsl.vmatching(
        frozenset({(1, (1, 1)), (2, (0, 0)), (2, (1, 0)), (3, (0, 1))}),
        frozenset({(1, (3, 2)), (2, (4, 2))}),
    )


def test_manhattan():
    assert dsl.manhattan(frozenset({(0, 0), (1, 1)}), frozenset({(1, 2), (2, 3)})) == 1
    assert dsl.manhattan(frozenset({(1, 1)}), frozenset({(2, 3)})) == 3


def test_adjacent():
    assert dsl.adjacent(frozenset({(0, 0)}), frozenset({(0, 1), (1, 0)}))
    assert not dsl.adjacent(frozenset({(0, 0)}), frozenset({(1, 1)}))


def test_bordering():
    assert dsl.bordering(frozenset({(0, 0)}), D)
    assert dsl.bordering(frozenset({(0, 2)}), D)
    assert dsl.bordering(frozenset({(2, 0)}), D)
    assert dsl.bordering(frozenset({(2, 2)}), D)
    assert not dsl.bordering(frozenset({(1, 1)}), D)


def test_centerofmass():
    assert dsl.centerofmass(frozenset({(0, 0), (1, 1), (1, 2)})) == (0, 1)
    assert dsl.centerofmass(frozenset({(0, 0), (1, 1), (2, 2)})) == (1, 1)
    assert dsl.centerofmass(frozenset({(0, 0), (1, 1), (0, 1)})) == (0, 0)


def test_palette():
    assert dsl.palette(
        frozenset({(1, (1, 1)), (2, (0, 0)), (2, (1, 0)), (3, (0, 1))})
    ) == frozenset({1, 2, 3})
    assert dsl.palette(
        frozenset({(1, (1, 1)), (1, (0, 0)), (1, (1, 0)), (1, (0, 1))})
    ) == frozenset({1})


def test_numcolors():
    assert (
        dsl.numcolors(frozenset({(1, (1, 1)), (2, (0, 0)), (2, (1, 0)), (3, (0, 1))}))
        == 3
    )
    assert (
        dsl.numcolors(frozenset({(1, (1, 1)), (1, (0, 0)), (1, (1, 0)), (1, (0, 1))}))
        == 1
    )


def test_color():
    assert (
        dsl.color(frozenset({(1, (1, 1)), (1, (0, 0)), (1, (1, 0)), (1, (0, 1))})) == 1
    )
    assert dsl.color(frozenset({(2, (3, 1))})) == 2


def test_toobject():
    assert dsl.toobject(frozenset({(0, 0), (0, 2)}), G) == frozenset(
        {(1, (0, 0)), (0, (0, 2))}
    )
    assert dsl.toobject(frozenset({(0, 4)}), G) == frozenset({(3, (0, 4))})


def test_asobject():
    assert dsl.asobject(A) == frozenset(
        {(0, (0, 1)), (0, (1, 0)), (0, (2, 1)), (1, (0, 0)), (1, (1, 1)), (1, (2, 0))}
    )


def test_rot90():
    assert dsl.rot90(B) == ((2, 0, 2), (1, 1, 1))
    assert dsl.rot90(C) == ((5, 3), (5, 4))


def test_rot180():
    assert dsl.rot180(B) == ((1, 2), (1, 0), (1, 2))
    assert dsl.rot180(C) == ((5, 5), (4, 3))


def test_rot270():
    assert dsl.rot270(B) == ((1, 1, 1), (2, 0, 2))
    assert dsl.rot270(C) == ((4, 5), (3, 5))


def test_hmirror():
    assert dsl.hmirror(B) == ((2, 1), (0, 1), (2, 1))
    assert dsl.hmirror(C) == ((5, 5), (3, 4))
    assert dsl.hmirror(frozenset({(0, 0), (1, 1)})) == frozenset({(1, 0), (0, 1)})
    assert dsl.hmirror(frozenset({(0, 0), (1, 0), (1, 1)})) == frozenset(
        {(1, 0), (0, 1), (0, 0)}
    )
    assert dsl.hmirror(frozenset({(0, 1), (1, 2)})) == frozenset({(0, 2), (1, 1)})


def test_vmirror():
    assert dsl.vmirror(B) == ((1, 2), (1, 0), (1, 2))
    assert dsl.vmirror(C) == ((4, 3), (5, 5))
    assert dsl.vmirror(frozenset({(0, 0), (1, 1)})) == frozenset({(1, 0), (0, 1)})
    assert dsl.vmirror(frozenset({(0, 0), (1, 0), (1, 1)})) == frozenset(
        {(1, 0), (1, 1), (0, 1)}
    )
    assert dsl.vmirror(frozenset({(0, 1), (1, 2)})) == frozenset({(0, 2), (1, 1)})


def test_dmirror():
    assert dsl.dmirror(B) == ((2, 0, 2), (1, 1, 1))
    assert dsl.dmirror(C) == ((3, 5), (4, 5))
    assert dsl.dmirror(frozenset({(0, 0), (1, 1)})) == frozenset({(0, 0), (1, 1)})
    assert dsl.dmirror(frozenset({(0, 0), (1, 0), (1, 1)})) == frozenset(
        {(0, 1), (1, 1), (0, 0)}
    )
    assert dsl.dmirror(frozenset({(0, 1), (1, 2)})) == frozenset({(0, 1), (1, 2)})


def test_cmirror():
    assert dsl.cmirror(B) == ((1, 1, 1), (2, 0, 2))
    assert dsl.cmirror(C) == ((5, 4), (5, 3))
    assert dsl.cmirror(frozenset({(0, 0), (1, 1)})) == frozenset({(0, 0), (1, 1)})
    assert dsl.cmirror(frozenset({(0, 0), (1, 0), (1, 1)})) == frozenset(
        {(0, 0), (1, 0), (1, 1)}
    )
    assert dsl.cmirror(frozenset({(0, 1), (1, 2)})) == frozenset({(0, 1), (1, 2)})


def test_fill():
    assert dsl.fill(B, 3, frozenset({(0, 0), (1, 1)})) == ((3, 1), (0, 3), (2, 1))
    assert dsl.fill(C, 1, frozenset({(1, 0)})) == ((3, 4), (1, 5))


def test_paint():
    assert dsl.paint(B, frozenset({(1, (0, 0)), (2, (1, 1))})) == (
        (1, 1),
        (0, 2),
        (2, 1),
    )
    assert dsl.paint(C, frozenset({(6, (1, 0))})) == ((3, 4), (6, 5))


def test_underfill():
    assert dsl.underfill(C, 1, frozenset({(0, 0), (1, 0)})) == ((3, 4), (1, 5))


def test_underpaint():
    assert dsl.underpaint(B, frozenset({(3, (0, 0)), (3, (1, 1))})) == (
        (2, 1),
        (0, 3),
        (2, 1),
    )
    assert dsl.underpaint(C, frozenset({(3, (1, 1))})) == ((3, 4), (5, 3))


def test_hupscale():
    assert dsl.hupscale(B, 1) == B
    assert dsl.hupscale(C, 1) == C
    assert dsl.hupscale(B, 2) == ((2, 2, 1, 1), (0, 0, 1, 1), (2, 2, 1, 1))
    assert dsl.hupscale(C, 2) == ((3, 3, 4, 4), (5, 5, 5, 5))


def test_vupscale():
    assert dsl.vupscale(B, 1) == B
    assert dsl.vupscale(C, 1) == C
    assert dsl.vupscale(B, 2) == ((2, 1), (2, 1), (0, 1), (0, 1), (2, 1), (2, 1))
    assert dsl.vupscale(C, 2) == ((3, 4), (3, 4), (5, 5), (5, 5))


def test_upscale():
    assert dsl.upscale(B, 1) == B
    assert dsl.upscale(C, 1) == C
    assert dsl.upscale(B, 2) == (
        (2, 2, 1, 1),
        (2, 2, 1, 1),
        (0, 0, 1, 1),
        (0, 0, 1, 1),
        (2, 2, 1, 1),
        (2, 2, 1, 1),
    )
    assert dsl.upscale(C, 2) == ((3, 3, 4, 4), (3, 3, 4, 4), (5, 5, 5, 5), (5, 5, 5, 5))
    assert dsl.upscale(
        frozenset({(3, (0, 1)), (4, (1, 0)), (5, (1, 1))}), 2
    ) == frozenset(
        {
            (3, (0, 2)),
            (3, (0, 3)),
            (3, (1, 2)),
            (3, (1, 3)),
            (4, (2, 0)),
            (4, (3, 0)),
            (4, (2, 1)),
            (4, (3, 1)),
            (5, (2, 2)),
            (5, (3, 2)),
            (5, (2, 3)),
            (5, (3, 3)),
        }
    )
    assert dsl.upscale(frozenset({(3, (0, 0))}), 2) == frozenset(
        {(3, (0, 0)), (3, (1, 0)), (3, (0, 1)), (3, (1, 1))}
    )


def test_downscale():
    assert dsl.downscale(B, 1) == B
    assert dsl.downscale(C, 1) == C
    assert (
        dsl.downscale(
            (
                (2, 2, 1, 1),
                (2, 2, 1, 1),
                (0, 0, 1, 1),
                (0, 0, 1, 1),
                (2, 2, 1, 1),
                (2, 2, 1, 1),
            ),
            2,
        )
        == B
    )
    assert (
        dsl.downscale(((3, 3, 4, 4), (3, 3, 4, 4), (5, 5, 5, 5), (5, 5, 5, 5)), 2) == C
    )


def test_hconcat():
    assert dsl.hconcat(A, B) == ((1, 0, 2, 1), (0, 1, 0, 1), (1, 0, 2, 1))
    assert dsl.hconcat(B, A) == ((2, 1, 1, 0), (0, 1, 0, 1), (2, 1, 1, 0))


def test_vconcat():
    assert dsl.vconcat(A, B) == ((1, 0), (0, 1), (1, 0), (2, 1), (0, 1), (2, 1))
    assert dsl.vconcat(B, A) == ((2, 1), (0, 1), (2, 1), (1, 0), (0, 1), (1, 0))
    assert dsl.vconcat(B, C) == ((2, 1), (0, 1), (2, 1), (3, 4), (5, 5))


def test_subgrid():
    assert dsl.subgrid(frozenset({(3, (0, 0))}), C) == ((3,),)
    assert dsl.subgrid(frozenset({(5, (1, 0)), (5, (1, 1))}), C) == ((5, 5),)
    assert dsl.subgrid(frozenset({(2, (0, 1)), (4, (1, 0))}), D) == ((1, 2), (4, 5))
    assert dsl.subgrid(frozenset({(1, (0, 0)), (0, (2, 2))}), D) == D


def test_hsplit():
    assert dsl.hsplit(B, 1) == (B,)
    assert dsl.hsplit(B, 2) == (((2,), (0,), (2,)), ((1,), (1,), (1,)))
    assert dsl.hsplit(C, 1) == (C,)
    assert dsl.hsplit(C, 2) == (((3,), (5,)), ((4,), (5,)))


def test_vsplit():
    assert dsl.vsplit(B, 1) == (B,)
    assert dsl.vsplit(B, 3) == (((2, 1),), ((0, 1),), ((2, 1),))
    assert dsl.vsplit(C, 1) == (C,)
    assert dsl.vsplit(C, 2) == (((3, 4),), ((5, 5),))


def test_cellwise():
    assert dsl.cellwise(A, B, 0) == ((0, 0), (0, 1), (0, 0))
    assert dsl.cellwise(C, E, 0) == ((0, 0), (0, 5))


def test_replace():
    assert dsl.replace(B, 2, 3) == ((3, 1), (0, 1), (3, 1))
    assert dsl.replace(C, 5, 0) == ((3, 4), (0, 0))


def test_switch():
    assert dsl.switch(C, 3, 4) == ((4, 3), (5, 5))


def test_center():
    assert dsl.center(frozenset({(1, (0, 0))})) == (0, 0)
    assert dsl.center(frozenset({(1, (0, 0)), (1, (0, 2))})) == (0, 1)
    assert dsl.center(
        frozenset({(1, (0, 0)), (1, (0, 2)), (1, (2, 0)), (1, (2, 2))})
    ) == (
        1,
        1,
    )


def test_position():
    assert dsl.position(frozenset({(0, (1, 1))}), frozenset({(0, (2, 2))})) == (1, 1)
    assert dsl.position(frozenset({(0, (2, 2))}), frozenset({(0, (1, 2))})) == (-1, 0)
    assert dsl.position(frozenset({(0, (3, 3))}), frozenset({(0, (3, 4))})) == (0, 1)


def test_index():
    assert dsl.index(C, (0, 0)) == 3
    assert dsl.index(D, (1, 2)) == 6


def test_canvas():
    assert dsl.canvas(3, (1, 2)) == ((3, 3),)
    assert dsl.canvas(2, (3, 1)) == ((2,), (2,), (2,))


def test_corners():
    assert dsl.corners(frozenset({(1, 2), (0, 3), (4, 0)})) == frozenset(
        {(0, 0), (0, 3), (4, 0), (4, 3)}
    )
    assert dsl.corners(frozenset({(1, 2), (0, 0), (4, 3)})) == frozenset(
        {(0, 0), (0, 3), (4, 0), (4, 3)}
    )


def test_connect():
    assert dsl.connect((1, 1), (2, 2)) == frozenset({(1, 1), (2, 2)})
    assert dsl.connect((1, 1), (1, 4)) == frozenset({(1, 1), (1, 2), (1, 3), (1, 4)})


def test_cover():
    assert dsl.cover(C, frozenset({(0, 0)})) == ((5, 4), (5, 5))


def test_trim():
    assert dsl.trim(D) == ((5,),)


def test_move():
    assert dsl.move(C, frozenset({(3, (0, 0))}), (1, 1)) == ((5, 4), (5, 3))


def test_tophalf():
    assert dsl.tophalf(C) == ((3, 4),)
    assert dsl.tophalf(D) == ((1, 2, 3),)


def test_bottomhalf():
    assert dsl.bottomhalf(C) == ((5, 5),)
    assert dsl.bottomhalf(D) == ((7, 8, 0),)


def test_lefthalf():
    assert dsl.lefthalf(C) == ((3,), (5,))
    assert dsl.lefthalf(D) == ((1,), (4,), (7,))


def test_righthalf():
    assert dsl.righthalf(C) == ((4,), (5,))
    assert dsl.righthalf(D) == ((3,), (6,), (0,))


def test_vfrontier():
    assert dsl.vfrontier((3, 4)) == frozenset({(i, 4) for i in range(30)})


def test_hfrontier():
    assert dsl.hfrontier((3, 4)) == frozenset({(3, i) for i in range(30)})


def test_backdrop():
    assert dsl.backdrop(frozenset({(2, 3), (3, 2), (3, 3), (4, 1)})) == frozenset(
        {
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 1),
            (4, 2),
            (4, 3),
        }
    )


def test_delta():
    assert dsl.delta(frozenset({(2, 3), (3, 2), (3, 3), (4, 1)})) == frozenset(
        {(2, 1), (2, 2), (3, 1), (4, 2), (4, 3)}
    )


def test_gravitate():
    assert dsl.gravitate(frozenset({(0, 0)}), frozenset({(0, 1)})) == (0, 0)
    assert dsl.gravitate(frozenset({(0, 0)}), frozenset({(0, 4)})) == (0, 3)


def test_inbox():
    assert dsl.inbox(frozenset({(0, 0), (2, 2)})) == frozenset({(1, 1)})


def test_outbox():
    assert dsl.outbox(frozenset({(1, 1)})) == frozenset(
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    )


def test_box():
    assert dsl.box(frozenset({(0, 0), (1, 1)})) == frozenset(
        {(0, 0), (0, 1), (1, 0), (1, 1)}
    )


def test_shoot():
    assert dsl.shoot((0, 0), (1, 1)) == frozenset({(i, i) for i in range(43)})


def test_occurrences():
    assert dsl.occurrences(G, frozenset({(1, (0, 0)), (1, (0, 1))})) == frozenset(
        {(1, 1), (2, 1)}
    )


def test_frontiers():
    assert dsl.frontiers(C) == frozenset({frozenset({(5, (1, 0)), (5, (1, 1))})})


def test_compress():
    assert dsl.compress(K) == (
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    )


def test_hperiod():
    assert (
        dsl.hperiod(
            frozenset(
                {
                    (8, (2, 1)),
                    (8, (1, 3)),
                    (2, (2, 4)),
                    (8, (2, 3)),
                    (2, (2, 2)),
                    (2, (1, 2)),
                    (8, (1, 1)),
                    (8, (1, 5)),
                    (2, (1, 4)),
                    (8, (2, 5)),
                    (2, (2, 0)),
                    (2, (1, 0)),
                }
            )
        )
        == 2
    )
    assert (
        dsl.hperiod(
            frozenset(
                {
                    (2, (2, 6)),
                    (2, (2, 0)),
                    (3, (2, 4)),
                    (3, (2, 2)),
                    (3, (2, 5)),
                    (2, (2, 3)),
                    (3, (2, 1)),
                }
            )
        )
        == 3
    )


def test_vperiod():
    assert (
        dsl.vperiod(
            frozenset(
                {
                    (2, (2, 6)),
                    (2, (2, 0)),
                    (3, (2, 4)),
                    (3, (2, 2)),
                    (3, (2, 5)),
                    (2, (2, 3)),
                    (3, (2, 1)),
                }
            )
        )
        == 1
    )
    assert (
        dsl.vperiod(
            frozenset(
                {
                    (1, (2, 6)),
                    (2, (3, 5)),
                    (2, (3, 0)),
                    (2, (2, 2)),
                    (2, (2, 7)),
                    (1, (3, 4)),
                    (2, (2, 1)),
                    (1, (2, 3)),
                    (2, (2, 5)),
                    (2, (2, 4)),
                    (1, (3, 7)),
                    (1, (2, 0)),
                    (2, (3, 6)),
                    (2, (3, 2)),
                    (2, (3, 3)),
                    (1, (3, 1)),
                }
            )
        )
        == 2
    )
