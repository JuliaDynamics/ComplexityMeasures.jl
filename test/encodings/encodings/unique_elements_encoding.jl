using Test
using ComplexityMeasures
using StateSpaceSets

x = ['a', 2, 5, 2, 5, 'a']
e = UniqueElementsEncoding(x)
@test encode.(Ref(e), x) == [1, 2, 3, 2, 3, 1]

y = StateSpaceSet(["a", "b", "c", "b", "a"])
ey = UniqueElementsEncoding(y)
encode.(Ref(ey), y)
