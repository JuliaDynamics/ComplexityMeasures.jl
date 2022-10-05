using Distributions
# Analytical test from Zhou et al., (2022).
# -----------------------------------------

# Their values:
x = [12, 5, 2, 1.5, -1, 3.8, 9, 10, 4.5, 0.9, 8, 2.99]
s = [3, 2, 1, 1, 1, 2, 3, 3, 2, 1, 3, 1]
# There might be a typo in the paper. With the same Gaussian symbolization parameters,
# we get an identical symbol sequence, with the exception of the one element. To see this,
# run count(!isone, symbolize(x, est.symbolization) .== s), which gives 1 as a result.
#
# In the following, I'll assume that this is a typo from their side, because the
# symbolization they claim to use is the same as for the `Dispersion` estimator,
# which we analytically test elsewhere using test cases from the original paper.
est = Dispersion(symbolization = GaussianSymbolization(c = 3), m = 2)

# If we take *our* symbolized sequence with m = 2,
# s = [3, 2, 1, 1, 1, 2, 3, 3, 2, 1, 3, 1],
# then the symbols (3, 1) and (2, 2) don't occur, so the number of missing dispersion
# patterns is 2, not 1 (as in the paper).
@test missing_dispersion(x, est = est, normalize = false) == 2.0
@test missing_dispersion(x, est = est, normalize = true) == 2/9

# For uniformly distributed noise, with sufficiently low-dimensional embeddings and
# few enough categories for the symbolization, it is expected that all dispersion patterns
# are eventually encountered, so the number of missing dispersion patterns would be 0,
# regardless of parameters.
cs = [2, 3]
ms = [2, 3]
for (c, m) in zip(cs, ms)
    est = Dispersion(symbolization = GaussianSymbolization(c = c), m = m)
    @test missing_dispersion(rand(1000000), est = est) == 0.0
end

# For uniformly distributed noise, provided time series length N >> c^m, it is expected
# that all dispersion patterns are eventually encountered, so the number of missing
# dispersion patterns would be 0, regardless of parameters.
cs = [2, 3, 4, 5]
ms = [2, 3, 4, 5]
for (c, m) in zip(cs, ms)
    est = Dispersion(symbolization = GaussianSymbolization(c = c), m = m)
    @test missing_dispersion(rand(Normal(), c^m*100), est = est) == 0.0
end
