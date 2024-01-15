using ComplexityMeasures, Test
using DelayEmbeddings
using Distances

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
