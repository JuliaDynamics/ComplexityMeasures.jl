using Test, ComplexityMeasures
using Random; rng = MersenneTwister(1234)

x = rand(rng, 1000)
est = BubbleEntropy(m = 5)
@test complexity(est, x) isa Real
@test 0.0 <= complexity_normalized(est, x) <= 1.0