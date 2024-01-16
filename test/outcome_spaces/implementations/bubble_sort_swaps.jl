using ComplexityMeasures, Test
using Random; rng = MersenneTwister(1234)

# Constructor
@test BubbleSortSwaps(; m = 3, τ = 1) isa BubbleSortSwaps
@test BubbleSortSwaps(; m = 3, τ = 1) isa ComplexityMeasures.CountBasedOutcomeSpace

# Codify
x = rand(rng, 100000) # enough points to cover the outcome space for small `m`
m = 3
o = BubbleSortSwaps(; m = m, τ = 1)
observed_outs = codify(o, x)
@test length(observed_outs) == length(x) - (m - 1)

# Outcomes
o = BubbleSortSwaps(; m = 3, τ = 1)
cts, outs = counts_and_outcomes(o, x)
@test total_outcomes(o) ==  (m * (m - 1) / 2) + 1
@test total_outcomes(o, x) ==  total_outcomes(o)
@test outcome_space(o) == collect(0:(total_outcomes(o) - 1)) # 0 included, so 1 less
@test outs == outcome_space(o) # should be enough points in `x` to be true
