using Test
using ComplexityMeasures
using Random
using Distributions: Uniform

rng = Xoshiro(1234)
# With three symbols, the unit interval is symbolized into three different symbols [1, 2 3],
# and the average symbol is 2.
minval, maxval = 0, 1
𝒰 = Uniform(minval, maxval)
n = 3
symbols3 = [encode(RelativeFirstDifferenceEncoding(0, 1; n), rand(rng, 𝒰, 5)) for i = 1:1000000]
@test all(1 .<= symbols3 .<= n)

# Zero first differences should give symbol 1
@test encode(RelativeFirstDifferenceEncoding(0, 1; n = 3), [1, 1, 1]) == 1

# Need at least one interval
@test_throws ArgumentError RelativeFirstDifferenceEncoding(0, 1, n = 0)

# minval/maxval must be ordered correctly
@test_throws ArgumentError RelativeFirstDifferenceEncoding(1, 0, n = 2)

# `n` must be positive and nonzero
@test_throws ArgumentError RelativeMeanEncoding(0, 1, n = 0)
