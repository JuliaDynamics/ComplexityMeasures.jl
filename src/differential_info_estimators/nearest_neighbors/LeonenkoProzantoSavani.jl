using SpecialFunctions: gamma
using Neighborhood: bulksearch
using Neighborhood: Euclidean, Theiler

export LeonenkoProzantoSavani

"""
    LeonenkoProzantoSavani <: DifferentialInfoEstimator
    LeonenkoProzantoSavani(definition = Shannon(); k = 1, w = 0)

The `LeonenkoProzantoSavani` estimator [LeonenkoProzantoSavani2008](@cite)
computes the  [`Shannon`](@ref), [`Renyi`](@ref), or
[`Tsallis`](@ref) differential [`information`](@ref) of a multi-dimensional
[`StateSpaceSet`](@ref), with logarithms to the `base` specified in `definition`.

## Description

The estimator uses `k`-th nearest-neighbor searches.  `w` is the Theiler window, which
determines if temporal neighbors are excluded during neighbor searches (defaults to `0`,
meaning that only the point itself is excluded when searching for neighbours).

For details, see [LeonenkoProzantoSavani2008](@citet).
"""
struct LeonenkoProzantoSavani{I} <: DifferentialInfoEstimator{I}
    definition::I
    k::Int
    w::Int

    function LeonenkoProzantoSavani(definition::I, k, w)
        if !(I isa Shannon || I isa Renyi || I isa Tsallis)
            s = "`definition` must be either a `Shannon`, `Renyi` or `Tsallis` instance."
            throw(ArgumentError(s))
        end
        new{I}(definition, k, w)
    end
end
function LeonenkoProzantoSavani(definition = Shannon(); k = 1, w = 0)
    return LeonenkoProzantoSavani(definition, k, w)
end

function information(est::LeonenkoProzantoSavani{<:Shannon}, x::AbstractStateSpaceSet)
    h = Î(1.0, est, x) # measured in nats
    return convert_logunit(h, ℯ, e.base)
end

function information(est::LeonenkoProzantoSavani{<:Renyi}, x::AbstractStateSpaceSet)
    q = est.definition.q
    base = est.definition.base

    if q ≈ 1.0
        h = Î(q, est, x) # measured in nats
    else
        h = log(Î(q, est, x)) / (1 - q) # measured in nats
    end
    return convert_logunit(h, ℯ, base)
end

function entropy(est::LeonenkoProzantoSavani{<:Tsallis}, x::AbstractStateSpaceSet)
    q = est.definition.q
    base = est.definition.base

    if q ≈ 1.0
        h = Î(q, est, x) # measured in nats
    else
        h = (Î(q, est, x) - 1) / (1 - q) # measured in nats
    end
    return convert_logunit(h, ℯ, base)
end

# TODO: this gives nan??
# Use notation from original paper
function Î(q, est::LeonenkoProzantoSavani, x::AbstractStateSpaceSet{D}) where D
    (; k, w) = est
    N = length(x)
    Vₘ = ball_volume(D)
    Cₖ = (gamma(k) / gamma(k + 1 - q))^(1 / (1 - q))
    tree = KDTree(x, Euclidean())
    idxs, ds = bulksearch(tree, x, NeighborNumber(k), Theiler(w))
    if q ≈ 1.0 # equations 3.9 & 3.10 in Leonenko et al. (2008)
        h = (1 / N) * sum(log.(ξᵢ_shannon(last(dᵢ), Vₘ, N, D, k) for dᵢ in ds))
    else # equations 3.1 & 3.2 in Leonenko et al. (2008)
        h = (1 / N) * sum(ξᵢ_renyi_tsallis(last(dᵢ), Cₖ, Vₘ, N, D)^(1 - q) for dᵢ in ds)
    end
    return h
end
ξᵢ_renyi_tsallis(dᵢ, Cₖ, Vₘ, N::Int, D::Int) = (N - 1) * Cₖ * Vₘ * (dᵢ)^D
ξᵢ_shannon(dᵢ, Vₘ, N::Int, D::Int, k) = (N - 1) * exp(-digamma(k)) * Vₘ * (dᵢ)^D
