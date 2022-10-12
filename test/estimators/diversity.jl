# Analytical example from Wang et al. (2020)
x = [1, 2, 13, 7, 9, 5, 4]
est = Diversity(; nbins = 10, m = 3, τ = 1)
@test probabilities(x, est) == [0.5, 0.5]
@test round(entropy_normalized(x, est), digits = 4) == 0.3010
