using Test
using ComplexityMeasures

x = ['a', 2, 5, 2, 5, 'a']
e = UniqueElementsEncoding(x)
@test encode.(Ref(e), x) == [1, 2, 3, 2, 3, 1]
