export Kraskov

"""
    Kraskov <: DiffEntropyEst
    Kraskov(; k::Int = 1, w::Int = 1, base = 2)

The `Kraskov` estimator computes the [`Shannon`](@ref) differential [`entropy`](@ref) of
a multi-dimensional [`Dataset`](@ref) using the `k`-th nearest neighbor
searches method from [^Kraskov2004] at the given `base`.

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

## Description

Assume we have samples ``\\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function``f : \\mathbb{R}^d \\to \\mathbb{R}``. `Kraskov` estimates the
[Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))].
```

See also: [`entropy`](@ref), [`KozachenkoLeonenko`](@ref), [`DifferentialEntropyEstimator`](@ref).

[^Kraskov2004]:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information. Physical review E, 69(6), 066138.
"""
Base.@kwdef struct Kraskov{B} <: DiffEntropyEst
    k::Int = 1
    w::Int = 1
    base::B = 2
end

function entropy(est::Kraskov, x::AbstractDataset{D}) where {D}
    (; k, w) = est
    N = length(x)
    ρs = maximum_neighbor_distances(x, w, k)
    # The estimated entropy has "unit" [nats]
    h = -digamma(k) + digamma(N) +
        log(MathConstants.e, ball_volume(D)) +
        D/N*sum(log.(MathConstants.e, ρs))
    return h / log(est.base, MathConstants.e) # Convert to target unit
end
