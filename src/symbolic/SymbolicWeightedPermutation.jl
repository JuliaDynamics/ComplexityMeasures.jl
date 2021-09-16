import DelayEmbeddings: genembed, Dataset
import Statistics: mean

export SymbolicWeightedPermutation

"""
    SymbolicWeightedPermutation(; τ = 1, m = 3, lt = Entropies.isless_rand) <: PermutationProbabilityEstimator

See docstring for [`SymbolicPermutation`](@ref).
"""
struct SymbolicWeightedPermutation{F} <: PermutationProbabilityEstimator
    τ::Int
    m::Int
    lt::F
    function SymbolicWeightedPermutation(; τ::Int = 1, m::Int = 3, lt::Function = isless_rand)
        m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
        new(τ, m, lt)
    end
end

function weights_from_variance(x, m::Int)
    sum((x .- mean(x)) .^ 2)/m
end


function probabilities(x::AbstractDataset{m, T}, est::SymbolicWeightedPermutation) where {m, T}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    πs = symbolize(x, SymbolicPermutation(m = m, lt = est.lt))  # motif length controlled by dimension of input data
    wts = weights_from_variance.(x.data, m)

    Probabilities(symprobs(πs, wts, normalize = true))
end

function probabilities(x::AbstractVector{T}, est::SymbolicWeightedPermutation) where {T<:Real}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    emb = genembed(x, τs)
    πs = symbolize(emb, SymbolicPermutation(m = est.m, lt = est.lt)) # motif length controlled by estimator m
    wts = weights_from_variance.(emb.data, est.m)

    Probabilities(symprobs(πs, wts, normalize = true))
end
