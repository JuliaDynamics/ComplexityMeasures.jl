using ComplexityMeasures
using Random
rng = MersenneTwister(1234)
# Analytical test from Zhou et al., (2022).
# -----------------------------------------

# Their values:
x = [12, 5, 2, 1.5, -1, 3.8, 9, 10, 4.5, 0.9, 8, 2.99]
s = [3, 2, 1, 1, 1, 2, 3, 3, 2, 1, 3, 1]
dispest = Dispersion(c = 3, m = 2)
est = MissingDispersionPatterns(dispest)
# If we embed take the discretized sequence s = [3, 2, 1, 1, 1, 2, 3, 3, 2, 1, 3, 1],
# with m = 2 and τ = 1, hen the symbol (2, 2) doesn't occur, so the number of missing
# dispersion patterns is 1
@test complexity(est, x) == 1
@test complexity_normalized(est, x) == 1/9

# For uniformly distributed noise, with sufficiently low-dimensional embeddings and
# few enough categories for the symbolization, it is expected that all dispersion patterns
# are eventually encountered, so the number of missing dispersion patterns would be 0,
# regardless of parameters.
cs = [2, 3]
ms = [2, 3]
for (c, m) in zip(cs, ms)
    local d = Dispersion(c = c, m = m)
    local est = MissingDispersionPatterns(d)
    @test complexity(est, rand(1000000)) == 0.0
end

# For time series of length N >> c^m, it is expected
# that all dispersion patterns are eventually encountered, so the number of missing
# dispersion patterns would be 0, regardless of parameters.
cs = [2, 3, 4, 5]
ms = [2, 3, 4, 5]
for (c, m) in zip(cs, ms)
    local d = Dispersion(c = c, m = m)
    local est = MissingDispersionPatterns(d)
    @test complexity(est, randn(rng, c^m * 100) |> collect) == 0
end
