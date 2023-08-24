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


# ----------------------------------------------------------------
# Analytical tests
# ----------------------------------------------------------------
encoding = RelativeFirstDifferenceEncoding(0, 1, n = 3)

x = [0.2, 0.1, 0.0] # mean(abs.(diff(x)) => 1st bin with n = 3
s1 = encode(encoding, x)
@test s1 == 1

x = [0.5, 0.2, 0.5, 0.1, 1.0] # mean(abs.(diff(x)) = 0.45 => 2nd bin with n = 3
s2 = encode(encoding, x)
@test s2 == 2

x = [0.8, 0.1, 1.0] # mean(abs.(diff(x)) = 0.8 => 3rd with n = 3
s3 = encode(encoding, x)
@test s3 == 3

@test first(decode(encoding, s1)) ≈ 0.0
@test first(decode(encoding, s2)) ≈ 1/3
@test first(decode(encoding, s3)) ≈ 2/3
