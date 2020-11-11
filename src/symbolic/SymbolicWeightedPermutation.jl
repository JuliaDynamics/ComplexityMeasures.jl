import DelayEmbeddings: genembed, Dataset
import Statistics: mean

export SymbolicWeightedPermutation

"""
    SymbolicWeightedPermutation(; τ = 1, m = 3) <: PermutationProbabilityEstimator

A symbolic, weighted permutation based probabilities/entropy estimator.

## Properties of original signal preserved

Weighted permutations of a signal preserve not only ordinal patterns (sorting information),
but also encodes amplitude information. This implementation is based on Fadlallah et al.
(2013)[^Fadlallah2013].


## Estimation

### Univariate time series

To estimate probabilities or entropies from univariate time series, use the following methods:

- `probabilities(x::AbstractVector, est::SymbolicWeightedPermutation)`. Constructs state vectors 
    from `x` using embedding lag `τ` and embedding dimension `m`. The ordinal patterns of the 
    state vectors are then symbolized, and probabilities are taken as the (weighted) relative 
    frequency of symbols.
- `genentropy(x::AbstractVector, est::SymbolicWeightedPermutation; α=1, base = 2)` computes
    weighted probabilities by calling `probabilities(x::AbstractVector, est::SymbolicPermutation)`,
    then computer the order-`α` generalized entropy to the given base.

See below for in-place versions below allow you to provide a pre-allocated symbol array `s`
for faster repeated computations of input data of the same length.

!!! info "Default embedding dimension and embedding lag"
    By default, embedding dimension ``m = 3`` with embedding lag ``\\tau = 1`` is used when
    embedding a time series for symbolization. You should probably make a more informed
    decision about embedding parameters when computing the permutation entropy of a real
    time series. In all cases, ``m`` must be at least 2 (there are
    no permutations of a single-element state vector, so need ``m \\geq 2``).

### Multivariate datasets

As for regular permutation entropy, numerically speaking, weighted permutation entropy 
can also be computed for multivariate datasets with dimension ≥ 2. Such
datasets may be, for example, preembedded time series. Then, just skip the delay 
reconstruction step, compute and symbols directly from the ``L`` existing state vectors 
``\\{\\mathbf{x}_1, \\mathbf{x}_2, \\ldots, \\mathbf{x_L}\\}``.

- `probabilities(x::Dataset, est::SymbolicWeightedPermutation)`. Compute ordinal patterns of the 
    state vectors of `x` directly (without doing any embedding), symbolize those patterns,
    and compute probabilities as relative frequencies of symbols.
- `genentropy(x::Dataset, est::SymbolicWeightedPermutation)`. Computes probabilities from 
    symbol frequencies using `probabilities(x::Dataset, est::SymbolicWeightedPermutation)`,
    then computes the order-`α` generalized (permutation) entropy to the given base.

!!! warn "Dynamical interpretation"
    A dynamical interpretation of the permutation entropy does not necessarily hold if
    computing it on generic multivariate datasets. Method signatures for `Dataset`s are
    provided for convenience, and should only be applied if you understand the relation
    between your input data, the numerical value for the permutation entropy, and
    its interpretation.

## Description

### Embedding, ordinal patterns and symbolization

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

### Probability computation

Weighted permutation entropy is computed analogously to regular permutation entropy
(see [`SymbolicPermutation`](@ref)), but
adds weights that encode amplitude information too:

```math
p(\\pi_i^{m, \\tau}) = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} 
\\left( \\mathbf{x}_k^{m, \\tau} \\right) 
\\, w_k}{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) \\in \\Pi} 
\\left( \\mathbf{x}_k^{m, \\tau} \\right) \\,w_k} = \\dfrac{\\sum_{k=1}^N 
\\mathbf{1}_{u:S(u) = s_i} 
\\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, w_k}{\\sum_{k=1}^N w_k}.
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

### Entropy computation

The generalized order-`α` Renyi entropy[^Rényi1960] can be computed over the probability 
distribution of symbols as 
``
H(m, \\tau, \\alpha) = \\dfrac{\\alpha}{1-\\alpha} \\log 
\\left( \\sum_{j=1}^R p_j^\\alpha \\right)
``. Permutation entropy, as described in 
Bandt and Pompe (2002), is just the limiting case as ``α \\to1``, that is
``
H(m, \\tau) = - \\sum_j^R p(\\pi_j^{m, \\tau}) \\ln p(\\pi_j^{m, \\tau})
``.

!!! hint "Generalized entropy order vs. permutation order"
    Do not confuse the order of the generalized entropy (`α`) with the order `m` of the
    permutation entropy (`m`, which controls the symbol size). Weighted permutation entropy is usually
    estimated with `α = 1`, but the implementation here allows the generalized entropy of any
    dimension to be computed from the symbol frequency distribution.

See also: [`SymbolicPermutation`](@ref), [`SymbolicAmplitudeAwarePermutation`](@ref).

[^Fadlallah2013]: Fadlallah, Bilal, et al. "Weighted-permutation entropy: A complexity
    measure for time series incorporating amplitude information." Physical
    Review E 87.2 (2013): 022911.
"""
struct SymbolicWeightedPermutation
    τ
    m
    function SymbolicWeightedPermutation(; τ::Int = 1, m::Int = 3)
        m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
        new(τ, m)
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

function probabilities(x::AbstractDataset{m, T}, est::SymbolicWeightedPermutation) where {m, T}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    πs = symbolize(x, SymbolicPermutation(m = m))  # motif length controlled by dimension of input data
    wts = weights_from_variance.(x.data, m)

    Probabilities(probs(πs, wts, normalize = true))
end

function probabilities(x::AbstractVector{T}, est::SymbolicWeightedPermutation) where {T<:Real}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    emb = genembed(x, τs)
    πs = symbolize(emb, SymbolicPermutation(m = est.m)) # motif length controlled by estimator m
    wts = weights_from_variance.(emb.data, est.m)

    Probabilities(probs(πs, wts, normalize = true))
end
