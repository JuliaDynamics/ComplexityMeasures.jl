using Test, Random
rng = MersenneTwister(1234)

x = rand(1:5, 1000)
o = UniqueElements()

h = information(ChaoShen(Shannon()), o, x)
@test h isa Real
@test h >= 0.0

# Since the plugin estimator is a lower bound for the Shannon entropy (see Arora et al.;
# https://arxiv.org/pdf/2204.01469.pdf), we should get a larger or equal value with the
# Chao-Shen estimator.
@test h >= information(PlugIn(Shannon()), o, x)

# If every outcome is a singleton (f₁ = N), f₁ is set to N - 1, so the coverage is
# C = 1/N instead of 0, and the estimate is finite and positive.
N = 10
h = information(ChaoShen(Shannon(; base = ℯ)), o, collect(1:N))
C = 1 / N
p = 1 / N
@test h ≈ -N * (C * p * log(C * p)) / (1 - (1 - C * p)^N)
@test isfinite(h) && h > 0
