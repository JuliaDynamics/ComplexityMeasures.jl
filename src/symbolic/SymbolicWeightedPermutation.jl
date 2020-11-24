import DelayEmbeddings: genembed, Dataset
import Statistics: mean

export SymbolicWeightedPermutation

"""
    SymbolicWeightedPermutation(; τ = 1, m = 3, lt = Entropies.isless_rand) <: PermutationProbabilityEstimator

See docstring for [`SymbolicPermutation`](@ref).
"""
struct SymbolicWeightedPermutation
    τ
    m
    lt::Function
    function SymbolicWeightedPermutation(; τ::Int = 1, m::Int = 3, lt::Function = isless_rand)
        m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
        new(τ, m, lt)
    end
end

function weights_from_variance(x, m::Int)
    sum((x .- mean(x)) .^ 2)/m
end


""" Compute probabilities of symbols `Π`, given weights `wts`. """
function probs(Π::AbstractVector, wts::AbstractVector; normalize = true, 
        lt::Function = isless_rand)
    length(Π) == length(wts) || error("Need length(Π) == length(wts)")
    N = length(Π)
    idxs = sortperm(Π, alg = QuickSort, lt = lt)
    sΠ = Π[idxs]   # sorted symbols
    sw = wts[idxs] # sorted weights

    i = 1   # symbol counter
    W = 0.0 # Initialize weight
    ps = Float64[]

    prev_sym = sΠ[1]

    while i <= length(sΠ)
        symᵢ = sΠ[i]
        wtᵢ = sw[i]
        if symᵢ == prev_sym
            W += wtᵢ
        else
            # Finished counting weights for the previous symbol, so push
            # the summed weights (normalization happens later).
            push!(ps, W)

            # We are at a new symbol, so refresh sum with the first weight
            # of the new symbol.
            W = wtᵢ
        end
        prev_sym = symᵢ
        i += 1
    end
    push!(ps, W) # last entry

    # Normalize
    Σ = sum(sw)
    if normalize
        return ps ./ Σ
    else
        return ps
    end
end

function probabilities(x::AbstractDataset{m, T}, est::SymbolicWeightedPermutation) where {m, T}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    πs = symbolize(x, SymbolicPermutation(m = m))  # motif length controlled by dimension of input data
    wts = weights_from_variance.(x.data, m)

    Probabilities(probs(πs, wts, normalize = true, lt = est.lt))
end

function probabilities(x::AbstractVector{T}, est::SymbolicWeightedPermutation) where {T<:Real}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    emb = genembed(x, τs)
    πs = symbolize(emb, SymbolicPermutation(m = est.m)) # motif length controlled by estimator m
    wts = weights_from_variance.(emb.data, est.m)

    Probabilities(probs(πs, wts, normalize = true, lt = est.lt))
end
