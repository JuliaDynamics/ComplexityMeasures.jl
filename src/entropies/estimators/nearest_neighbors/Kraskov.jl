export Kraskov

"""
    Kraskov <: EntropyEstimator
    Kraskov(; k::Int = 1, w::Int = 1, base = 2)

The `Kraskov` estimator computes the [`Shannon`](@ref) [`entropy`](@ref) of `x`
(a multi-dimensional `Dataset`) to the given `base`, using the `k`-th nearest neighbor
searches method from [^Kraskov2004].

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

See also: [`entropy`](@ref), [`KozachenkoLeonenko`](@ref).

[^Kraskov2004]:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information. Physical review E, 69(6), 066138.
"""
Base.@kwdef struct Kraskov{B} <: EntropyEstimator
    k::Int = 1
    w::Int = 1
    base::B = 2
end

function entropy(e::Renyi, est::Kraskov, x::AbstractDataset{D, T}) where {D, T}
    e.q == 1 || throw(ArgumentError(
        "Renyi entropy with q = $(e.q) not implemented for $(typeof(est)) estimator"
    ))
    (; k, w, base) = est
    N = length(x)
    ρs = maximum_neighbor_distances(x, w, k)
    # The estimated entropy has "unit" [nats]
    h = -digamma(k) + digamma(N) +
        log(MathConstants.e, ball_volume(D)) +
        D/N*sum(log.(MathConstants.e, ρs))
    return h / log(base, MathConstants.e) # Convert to target unit
end
