using Test, ComplexityMeasures
using Random; rng = MersenneTwister(1234)

stencil = [1 1; 1 1];
x = rand(10, 10);
@test SpatialBubbleSortSwaps(stencil, x) isa SpatialBubbleSortSwaps

# m = 4 for the given stencil, so there should be ((4 * (4 - 1)) ÷ 2) + 1 = 7 outcomes 
# (we also count 0 swaps as an outcome)
stencil = [1 1; 1 1];
x = rand(500, 500); # enough data that all outcomes should be covered.

o = SpatialBubbleSortSwaps(stencil, x)
cts, outs = counts_and_outcomes(o, x)
@test length(cts) == length(outs) == 7

@test total_outcomes(o) == 7
@test outcome_space(o) == 0:(7 - 1)

# Pretty printing
# ----------------------------------------------------------------
# hidden fields: [:viewer, :arraysize, :valid, :encoding]
s = repr(o)

@test occursin("stencil = ", s)
@test occursin("m = ", s)
@test !occursin("viewer = ", s)
@test !occursin("arraysize = ", s)
@test !occursin("valid = ", s)
@test !occursin("encoding = ", s)




