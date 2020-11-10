

import Entropies.symbolize
import DelayEmbeddings: genembed, Dataset
import Statistics: mean

export SymbolicWeightedPermutation, probabilities, genentropy, genentropy!

"""
    SymbolicWeightedPermutation <: PermutationProbabilityEstimator

A symbolic, weighted permutation based probabilities/entropy estimator.

## Properties of original signal preserved

Weighted permutations of a signal preserve not only ordinal patterns (sorting information), 
but also encodes amplitude information. This implementation is based on Fadlallah et al. 
(2013)[^Fadlallah2013].

## Description

Consider the ``n``-element univariate time series ``\\{x(t) = x_1, x_2, \\ldots, x_n\\}``. 
Let ``\\mathbf{x_i}^{m, \\tau} = \\{x_j, x_{j+\\tau}, \\ldots, x_{j+(m-1)\\tau}\\}`` for 
``j = 1, 2, \\ldots n - (m-1)\\tau`` be the ``i``-th state vector in a delay reconstruction 
with embedding dimension ``m`` and reconstruction lag ``\\tau``. There are then 
``N = n - (m-1)\\tau`` state vectors. 

For an ``m``-dimensional vector, there are ``m!`` possible ways of sorting it in ascending 
order of magnitude. Each such possible sorting ordering is called a *motif*. 
Let ``\\pi_i^{m, \\tau}`` denote the motif associated with the ``m``-dimensional state 
vector ``\\mathbf{x_i}^{m, \\tau}``, and let ``R`` be the number of distinct motifs that 
can be constructed from the ``N`` state vectors. Then there are at most ``R`` motifs; 
``R = N`` precisely when all motifs are unique, and ``R = 1`` when all motifs are the same. 
Each unique motif ``\\pi_i^{m, \\tau}`` can be mapped to a unique integer symbol 
``0 \\leq s_i \\leq M!-1``. Let ``S(\\pi) : \\mathbb{R}^m \\to \\mathbb{N}_0`` be the 
function that maps the motif ``\\pi`` to its symbol ``s``, and let ``\\Pi`` denote the set 
    of symbols ``\\Pi = \\{ s_i \\}_{i\\in \\{ 1, \\ldots, R\\}}``.

Weighted permutation entropy is computed analogously to regular permutation entropy, but 
adds weights that encode amplitude information too:

```math
p(\\pi_i^{m, \\tau}) = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, w_k}{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) \\in \\Pi} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\,w_k} = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, w_k}{\\sum_{k=1}^N w_k}.
```

The weighted permutation entropy is equivalent to regular permutation entropy when weights 
are positive and identical (``w_j = \\beta \\,\\,\\, \\forall \\,\\,\\, j \\leq N`` and 
``\\beta > 0)``. Weights are dictated by the variance of the state vectors.

Let the aritmetic mean of state vector ``\\mathbf{x}_i`` be denoted 
by

```math
\\mathbf{\\hat{x}}_j^{m, \\tau} = \\frac{1}{m} \\sum_{k=1}^m x_{j + (k+1)\\tau}.
```

Weights are then computed as 

```math
w_j = \\dfrac{1}{m}\\sum_{k=1}^m (x_{j+(k+1)\\tau} - \\mathbf{\\hat{x}}_j^{m, \\tau})^2.
```

!!! question "Implementation details"
    *Note: in equation 7, section III, of the original paper, the authors write*

    ```math
    w_j = \\dfrac{1}{m}\\sum_{k=1}^m (x_{j-(k-1)\\tau} - \\mathbf{\\hat{x}}_j^{m, \\tau})^2.
    ``` 
    *But given the formula they give for the arithmetic mean, this is **not** the variance 
    of ``\\mathbf{x}_i``, because the indices are mixed: ``x_{j+(k-1)\\tau}`` in the weights 
    formula, vs. ``x_{j+(k+1)\\tau}`` in the arithmetic mean formula. This seems to imply 
    that amplitude information about previous delay vectors 
    are mixed with mean amplitude information about current vectors. The authors also mix the 
    terms "vector" and "neighboring vector" (but uses the same notation for both), making it 
    hard to interpret whether the sign switch is a typo or intended. Here, we use the notation 
    above, which actually computes the variance for ``\\mathbf{x}_i``*.


### Estimation from univariate time series/datasets

- To compute weighted permutation entropy for a univariate signal `x`, use the signature 
    `entropy(x::AbstractVector, est::SymbolicWeightedPermutation; τ::Int = 1, m::Int = 3)`.

- The corresponding (unordered) probability distribution of the permutation symbols for a 
    univariate signal `x` can be computed using `probabilities(x::AbstractVector, 
    est::SymbolicWeightedPermutation; τ::Int = 1, m::Int = 3)`.  

!!! info "Default embedding dimension and embedding lag" 
    By default, embedding dimension ``m = 3`` with embedding lag ``\\tau = 1`` is used. You 
    should probably make a more informed decision about embedding parameters when computing the 
    permutation entropy of a real dataset. In all cases, ``m`` must be at least 2 (there are 
    no permutations of a single-element state vector, so need ``m \\geq 2``).

### Estimation from multivariate time series/datasets

Although not dealt with in the original paper, numerically speaking, weighted permutation 
entropy, just like regular permutation entropy, can also be computed for multivariate 
datasets (either embedded or consisting of multiple time series 
variables). This assumes that the mixed symbols described above are actually a typo.

Then, just skip the delay reconstruction step, compute symbols 
directly from the ``L`` existing state vectors 
``\\{\\mathbf{x}_1, \\mathbf{x}_2, \\ldots, \\mathbf{x_L}\\}``, symbolize 
each ``\\mathbf{x_i}`` precisely as above, then compute the 
quantity 

```math
H = - \\sum_j p(\\pi) \\ln p(\\pi_j).
```

- To compute weighted permutation entropy for a multivariate/embedded dataset `x`, use the 
    signature `entropy(x::AbstractDataset, est::SymbolicWeightedPermutation)`.

- To get the corresponding probability distribution for a multivariate/embedded dataset `x`, 
    use `probabilities(x::AbstractDataset, est::SymbolicWeightedPermutation)`.

!!! warn "Dynamical interpretation"
    A dynamical interpretation of the permutation entropy does not necessarily hold if 
    computing it on generic multivariate datasets. Method signatures for `Dataset`s are 
    provided for convenience, and should only be applied if you understand the relation 
    between your input data, the numerical value for the weighted permutation entropy, and 
    its interpretation.

See also: [`SymbolicPermutation`](@ref), [`SymbolicAmplitudeAwarePermutation`](@ref).

[^Fadlallah2013]: Fadlallah, Bilal, et al. "Weighted-permutation entropy: A complexity 
    measure for time series incorporating amplitude information." Physical 
    Review E 87.2 (2013): 022911.
"""
struct SymbolicWeightedPermutation

    function SymbolicWeightedPermutation()
        new()
    end
end

function weights_from_variance(x, m::Int)
    sum((x .- mean(x)) .^ 2)/m
end


""" Compute probabilities of symbols `Π`, given weights `wts`. """
function probs(Π::AbstractVector, wts::AbstractVector; normalize = true)
    length(Π) == length(wts) || error("Need length(Π) == length(wts)")
    N = length(Π)
    idxs = sortperm(Π, alg = QuickSort)
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
    
"""
# Weighted permutation-based symbol probabilities

    probabilities(x::AbstractDataset, est::SymbolicWeightedPermutation) → Vector{<:Real}  
    probabilities(x::AbstractVector{<:Real}, est::SymbolicWeightedPermutation; m::Int = 3, τ::Int = 1) → Vector{<:Real}

    probabilities!(s::Vector{Int}, x::AbstractDataset, est::SymbolicWeightedPermutation) → Vector{<:Real}  
    probabilities!(s::Vector{Int}, x::AbstractVector, est::SymbolicWeightedPermutation; m::Int = 3, τ::Int = 1) → Vector{<:Real}

Compute the unordered probabilities of the occurrence of weighted symbol sequences 
constructed from `x`. 

If `x` is a multivariate `Dataset`, then symbolization is performed directly on the state 
vectors. If `x` is a univariate signal, then a delay reconstruction with embedding lag `τ` 
and embedding dimension `m` is used to construct state vectors, on which symbolization is 
then performed.

A pre-allocated symbol array `s` can be provided to save some memory allocations if the 
probabilities are to be computed for multiple data sets. If provided, it is required that 
`length(x) == length(s)` if `x` is a `Dataset`, or  `length(s) == length(x) - (m-1)τ` 
if `x` is a univariate signal`.

See also: [`SymbolicWeightedPermutation`](@ref).
"""
function probabilities(x::AbstractDataset{m, T}, est::SymbolicWeightedPermutation) where {m, T}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    πs = symbolize(x, SymbolicPermutation()) 
    wts = weights_from_variance.(x.data, m)

    probs(πs, wts, normalize = true)
end

function probabilities(x::AbstractVector{T}, est::SymbolicWeightedPermutation; 
        m::Int = 3, τ::Int = 1) where {T<:Real}
    
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    τs = tuple([τ*i for i = 0:m-1]...)
    emb = genembed(x, τs)
    πs = symbolize(emb, SymbolicPermutation()) 
    wts = weights_from_variance.(emb.data, m)
    
    probs(πs, wts, normalize = true)
end

"""
# Weighted permutation entropy 

    genentropy(x::AbstractDataset, est::SymbolicWeightedPermutation, α::Real = 1; base = 2) → Real
    genentropy(x::AbstractVector{<:Real}, est::SymbolicWeightedPermutation, α::Real = 1; m::Int = 3, τ::Int = 1, base = 2) → Real

Compute the generalized order `α` entropy based on a weighted permutation 
symbolization of `x`, using symbol size/order `m` for the permutations.

If `x` is a multivariate `Dataset`, then symbolization is performed directly on the state 
vectors. If `x` is a univariate signal, then a delay reconstruction with embedding lag `τ` 
and embedding dimension `m` is used to construct state vectors, on which symbolization is 
then performed.

## Probability and entropy estimation 

An unordered symbol frequency histogram is obtained by symbolizing the points in `x` by
a weighted procedure, using [`probabilities(::AbstractDataset, ::SymbolicWeightedPermutation)`](@ref).
Sum-normalizing this histogram yields a probability distribution over the weighted symbols.

After the symbolization histogram/distribution has been obtained, the order `α` generalized 
entropy[^Rényi1960], to the given `base`, is computed from that sum-normalized symbol 
distribution, using [`genentropy`](@ref).

!!! hint "Generalized entropy order vs. permutation order"
    Do not confuse the order of the generalized entropy (`α`) with the order `m` of the 
    permutation entropy (`m`, which controls the symbol size). Permutation entropy is usually 
    estimated with `α = 1`, but the implementation here allows the generalized entropy of any 
    dimension to be computed from the symbol frequency distribution.

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960). This reference contains some extra text.

See also: [`SymbolicWeightedPermutation`](@ref), [`genentropy`](@ref).
"""
function genentropy(x::AbstractDataset{m, T}, est::SymbolicWeightedPermutation, α::Real = 1; base = 2) where {m, T}
    
    ps = probabilities(x, est)
    genentropy(α, ps; base = base)
end

function genentropy(x::AbstractArray{T}, est::SymbolicWeightedPermutation, α::Real = 1; 
        m::Int = 3, τ::Int = 1, base = 2) where {T<:Real}
    
    ps = probabilities(x, est, m = m, τ = τ)
    genentropy(α, ps; base = base)
end
