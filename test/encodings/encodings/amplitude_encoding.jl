using Test
using ComplexityMeasures
using Random
using Distributions: Uniform
using Statistics: mean

rng = Xoshiro(1234)
# With three symbols, the unit interval is symbolized into three different symbols [1, 2 3],
# and the average symbol is 2.
minval, maxval = 0, 1
𝒰 = Uniform(minval, maxval)
n = 3
symbols3 = [encode(AmplitudeEncoding(0, 1; n), rand(rng, 𝒰, 5)) for i = 1:1000000]
@test 1.99 ≤ mean(symbols3) ≤ 2.01
@test all(1 .<= symbols3 .<= n)
