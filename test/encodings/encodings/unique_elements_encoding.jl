using Test
using ComplexityMeasures
using StateSpaceSets

x = ['a', 2, 5, 2, 5, 'a']; e = UniqueElementsEncoding(x)
@test encode.(Ref(e), x) == [1, 2, 3, 2, 3, 1]

y = ["a", "b", "c", "b", "a"]; ey = UniqueElementsEncoding(y)
@test encode.(Ref(ey), y) == [1, 2, 3, 2, 1]

z = vec(StateSpaceSet(y)); ez = UniqueElementsEncoding(z)
@test encode.(Ref(ez), z)  == [1, 2, 3, 2, 1]

# TODO: this should really work (but broadcasting fails). The error is not in this package,
# but is due to lacking broadcasting implementation in StateSpaceSets.jl
# w = StateSpaceSet(y); ew = UniqueElementsEncoding(w)
# @test encode.(Ref(ew), w)  == [1, 2, 3, 2, 1]
