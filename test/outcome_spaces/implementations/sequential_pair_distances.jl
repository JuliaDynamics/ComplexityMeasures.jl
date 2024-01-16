using ComplexityMeasures, Test
using DelayEmbeddings
using Distances
using Random; rng = MersenneTwister(1234)
using DelayEmbeddings

# ----------------------------------------------------------------
# Analytical example
# ----------------------------------------------------------------
m = Chebyshev()
y = [1.0, 0.5, 0.25, 0.6]

# This should give five bins with left adges at [0.25], [0.30], [0.35], [0.40] and [0.45]
o = SequentialPairDistances(y; m = 1, τ = 1, n = 5)
@test total_outcomes(o) == 5
@test outcome_space(o) == collect(1:5)

cts, outs = allcounts_and_outcomes(o, y)
@test outs == collect(1:5)
@test cts[1] == 1
@test cts[2] == 1
@test cts[3] == 0
@test cts[4] == 0
@test cts[5] == 1

cts, outs = counts_and_outcomes(o, y)
@test outs == [1, 2, 5]

x = rand(rng, 100)
m = 2
o = SequentialPairDistances(x; m, n = 5)
encoded_pts = codify(o, x)
@test all(1 .<= encoded_pts .<= 5)
@test length(encoded_pts) == length(x) - (m - 1) - 1

# Initialization with pre-embedded points
z = embed(y, 2, 1)
@test SequentialPairDistances(z) isa SequentialPairDistances
@test SequentialPairDistances(z).m == 2

# ----------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------
s = repr(SequentialPairDistances(x))
@test !occursin("dists = ", s)

# If input is a state space set, then `τ` is not displayed
s = repr(SequentialPairDistances(z))
@test !occursin("τ = ", s)