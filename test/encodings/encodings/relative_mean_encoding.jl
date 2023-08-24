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
symbols3 = [encode(RelativeMeanEncoding(0, 1; n = 3), rand(rng, 𝒰, 3)) for i = 1:1000000]
@test 1.99 ≤ mean(symbols3) ≤ 2.01
@test all(1 .<= symbols3 .<= 3)

# Need at least one interval
@test_throws ArgumentError RelativeMeanEncoding(0, 1, n = 0)

# minval/maxval must be ordered correctly
@test_throws ArgumentError RelativeMeanEncoding(1, 0, n = 2)

# `n` must be positive and nonzero
@test_throws ArgumentError RelativeFirstDifferenceEncoding(0, 1, n = 0)


# ----------------------------------------------------------------
# Analytical tests
# ----------------------------------------------------------------
encoding = RelativeMeanEncoding(0, 1, n = 3)

x = [0.1, 0.1, 0.1] # mean = 0.1, so should fall in first bin with n = 3
s1 = encode(encoding, x)
@test s1 == 1

x = [0.0, 0.25, 0.5, 0.75, 1.0] # mean = 0.5, so should fall in second bin with n = 3
s2 = encode(encoding, x)
@test s2 == 2

x = [0.8, 0.9, 1.0] # mean = 0.9, so should fall in third bin with n = 3
s3 = encode(encoding, x)
@test s3 == 3

@test first(decode(encoding, s1)) ≈ 0.0
@test first(decode(encoding, s2)) ≈ 1/3
@test first(decode(encoding, s3)) ≈ 2/3
