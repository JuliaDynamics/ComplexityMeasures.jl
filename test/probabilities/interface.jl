using ComplexityMeasures, Test

x = ones(3)
p = probabilities(x)
@test p isa Probabilities
@test p == [1]
