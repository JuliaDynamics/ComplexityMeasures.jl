using ComplexityMeasures, Test

# Analytical example from Wang et al. (2020)
x = [1, 2, 13, 7, 9, 5, 4]
nbins = 10
m = 3
τ = 1
o = CosineSimilarityBinning(; nbins, m, τ)
@test probabilities(o, x) == [0.5, 0.5]
@test round(information_normalized(o, x), digits = 4) == 0.3010

@test total_outcomes(o) == 10

# CosineSimilarityBinning divides the interval [-1, 1] into nbins subintervals.
binsize = (1-(-1))/10
probs, events = probabilities_and_outcomes(o, x)

@test issorted(events)
@test issorted(outcome_space(o, x))

ds = [0.605, 0.698, 0.924, 0.930] # value from Wang et al. (2020)
# These distances should be in the following distance bins: [8, 8, 9, 9].
# Which means that the probability distribution should be [0.5, 0.5]
bins = floor.(Int, (ds .- (-1.0)) / binsize)
@test sort(bins) == [8, 8, 9, 9]
@test probs == [0.5, 0.5]

# The coordinates corresponding to the left corners of these bins are as
# follows, and should correspond with the events.
coords = (bins .* binsize) .+ (-1.0)
@test all(round.(only.(events), digits = 13) == round.(unique(coords), digits = 13))
