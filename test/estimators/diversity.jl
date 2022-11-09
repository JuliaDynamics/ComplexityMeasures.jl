using Test, Entropies

# Analytical example from Wang et al. (2020)
x = [1, 2, 13, 7, 9, 5, 4]
nbins = 10
m = 3
τ = 1
est = Diversity(; nbins, m, τ)
@test probabilities(x, est) == [0.5, 0.5]
@test total_outcomes(est) == 10

Ω = outcome_space(x, est)
@test length(Ω) == 10
@test round(entropy_normalized(x, est), digits = 4) == 0.3010


# Diversity divides the interval [-1, 1] into nbins subintervals.
binsize = nextfloat((1-(-1))/10, 2)
probs, events = probabilities_and_outcomes(x, est)

ds = [0.605, 0.698, 0.924, 0.930] # value from Wang et al. (2020)
# These distances should be in the following distance bins: [8, 8, 9, 9],
# Which means that the probability distribution should be [0.5, 0.5]
bins = floor.(Int, (ds .- (-1.0)) / binsize)
@test sort(bins) == [8, 8, 9, 9]
@test probs == [0.5, 0.5]
