using Test
using ComplexityMeasures
using Random
rng = Xoshiro(1234)

# -----------------------------------------------------------------------------------------
# 1D estimation.
# -----------------------------------------------------------------------------------------
# Non-numeric data
choices = ['a', 'b', 'c', 'q']
x = rand(rng, choices, 100)
c = counts(x)
@test c isa Counts{<:Integer, 1}
@test outcomes(c) == sort!(unique(x))

c, outs = counts_and_outcomes(x)
@test c isa Counts{<:Integer, 1}
@test outs == sort!(unique(x))

# Numeric data
choices = 1:400 # more letters in the alphabet than samples => some guaranteed zero counts
x = rand(rng, choices, 100)
c = counts(x)
@test c isa Counts{<:Integer, 1}
@test outcomes(c) == sort!(unique(x))

c, outs = counts_and_outcomes(x)
@test c isa Counts{<:Integer, 1}
@test outs == sort!(unique(x))
