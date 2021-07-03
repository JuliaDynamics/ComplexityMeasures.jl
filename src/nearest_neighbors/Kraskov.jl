export Kraskov, genentropy

"""
## k-th nearest neighbour(kNN) based
    
    Kraskov(; k::Int = 1, w::Int = 0) <: EntropyEstimator

Entropy estimator based on `k`-th nearest neighbor searches[^Kraskov2004].
`w` is the [Theiler window](@ref).

!!! info
    This estimator is only available for entropy estimation. 
    Probabilities cannot be obtained directly.

[^Kraskov2004]: Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical review E, 69(6), 066138.
"""
Base.@kwdef struct Kraskov <: NearestNeighborEntropyEstimator
    k::Int = 1
    w::Int = 0
end

function genentropy(x::AbstractDataset{D, T}, est::Kraskov; base::Real = MathConstants.e) where {D, T}
    N = length(x)
    ρs = maximum_neighbor_distances(x, est)
    h = -digamma(est.k) + digamma(N) + log(base, ball_volume(D)) + D/N*sum(log.(base, ρs))
    return h
end