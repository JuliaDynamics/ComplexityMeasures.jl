export entropy_kraskov

"""
    entropy_kraskov(x::AbstractDataset{D, T}; k::Int = 1, w::Int = 0,
        base::Real = MathConstants.e) where {D, T}

Estimate Shannon entropy to the given `base` using `k`-th nearest neighbor
searches (Kraskov, 2004)[^Kraskov2004].

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

See also: [`entropy_kozachenkoleonenko`](@ref).

[^Kraskov2004]: Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical review E, 69(6), 066138.
"""
function entropy_kraskov(x::AbstractDataset{D, T}; k::Int = 1, w::Int = 0, base::Real = MathConstants.e) where {D, T}
    N = length(x)
    ρs = maximum_neighbor_distances(x, w, k)
    h = -digamma(k) + digamma(N) + log(base, ball_volume(D)) + D/N*sum(log.(base, ρs))
    return h
end
