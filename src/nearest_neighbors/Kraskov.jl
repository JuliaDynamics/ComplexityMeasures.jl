export Kraskov, genentropy

"""
## k-th nearest neighbour(kNN) based
    
    Kraskov(k::Int = 1, w::Int = 1) <: NearestNeighborEntropyEstimator

Entropy estimator based on `k`-th nearest neighbor searches[^Kraskov2004].
`w` is the number of nearest neighbors to exclude when searching for neighbours 
(defaults to `0`, meaning that only the point itself is excluded).

!!! info
    This estimator is only available for entropy estimation. Probabilities cannot be obtained directly.

[^Kraskov2004]: Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical review E, 69(6), 066138.
"""
struct Kraskov <: NearestNeighborEntropyEstimator
    k::Int 
    w::Int

    function Kraskov(;k::Int = 1, w::Int = 0)
        new(k, w)
    end
end

function genentropy(x::AbstractDataset{D, T}, est::Kraskov; base::Real = MathConstants.e) where {D, T}
    N = length(x)
    ρs = get_ρs(x, est)
    h = -digamma(est.k) + digamma(N) + log(base, V(D)) + D/N*sum(log.(base, ρs))
    return h
end