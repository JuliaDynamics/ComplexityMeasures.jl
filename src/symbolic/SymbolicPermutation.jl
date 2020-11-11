export PermutationProbabilityEstimator, SymbolicPermutation
export symbolize

"""
A probability estimator based on permutations.
"""
abstract type PermutationProbabilityEstimator <: SymbolicProbabilityEstimator end

"""
    SymbolicPermutation <: PermutationProbabilityEstimator

A symbolic, permutation based probabilities/entropy estimator.

## Properties of original signal preserved

Permutations of a signal preserve ordinal patterns (sorting information). This
implementation is based on Bandt & Pompe et al. (2002)[^BandtPompe2002].

## Description

Consider the ``n``-element univariate time series ``\\{x(t) = x_1, x_2, \\ldots, x_n\\}``.
Let ``\\mathbf{x_i}^{m, \\tau} = \\{x_j, x_{j+\\tau}, \\ldots, x_{j+(m-1)\\tau}\\}``
for ``j = 1, 2, \\ldots n - (m-1)\\tau`` be the ``i``-th state vector in a delay
reconstruction with embedding dimension ``m`` and reconstruction lag ``\\tau``.
There are then ``N = n - (m-1)\\tau`` state vectors.

For an ``m``-dimensional vector, there are ``m!`` possible ways of sorting it in
ascending order of magnitude. Each such possible sorting ordering is called a
*motif*. Let ``\\pi_i^{m, \\tau}`` denote the motif associated with the
``m``-dimensional state vector ``\\mathbf{x_i}^{m, \\tau}``, and let ``R``
be the number of distinct motifs that can be constructed from the ``N`` state
vectors. Then there are at most ``R`` motifs; ``R = N`` precisely when all motifs
are unique, and ``R = 1`` when all motifs are the same.

Each unique motif ``\\pi_i^{m, \\tau}`` can be mapped to a unique integer
symbol ``0 \\leq s_i \\leq M!-1``. Let ``S(\\pi) : \\mathbb{R}^m \\to \\mathbb{N}_0`` be
the function that maps the motif ``\\pi`` to its symbol ``s``, and let ``\\Pi``
denote the set of symbols ``\\Pi = \\{ s_i \\}_{i\\in \\{ 1, \\ldots, R\\}}``.

The probability of a given motif is its frequency of occurrence, normalized by the total
number of motifs (with notation from [^Fadlallah2013]),

```math
p(\\pi_i^{m, \\tau}) = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left(\\mathbf{x}_k^{m, \\tau} \\right) }{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) \\in \\Pi} \\left(\\mathbf{x}_k^{m, \\tau} \\right)} = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left(\\mathbf{x}_k^{m, \\tau} \\right) }{N},
```

where the function ``\\mathbf{1}_A(u)`` is the indicator function of a set ``A``. That
    is, ``\\mathbf{1}_A(u) = 1`` if ``u \\in A``, and ``\\mathbf{1}_A(u) = 0`` otherwise.

Permutation entropy can be computed over the probability distribution of symbols
as ``H(m, \\tau) = - \\sum_j^R p(\\pi_j^{m, \\tau}) \\ln p(\\pi_j^{m, \\tau})``.

### Estimation from univariate time series/datasets

- To compute permutation entropy for a univariate signal `x`, use the signature
    `entropy(x::AbstractVector, est::SymbolicPermutation; τ::Int = 1, m::Int = 3)`.

- The corresponding (unordered) probability distribution of the permutation symbols for a
    univariate signal `x` can be computed using `probabilities(x::AbstractVector, est::SymbolicPermutation; τ::Int = 1, m::Int = 3)`.

!!! info "Default embedding dimension and embedding lag"
    By default, embedding dimension ``m = 3`` with embedding lag ``\\tau = 1`` is used when
    embedding a time series for symbolization. You should probably make a more informed
    decision about embedding parameters when computing the permutation entropy of a real
    time series. In all cases, ``m`` must be at least 2 (there are
    no permutations of a single-element state vector, so need ``m \\geq 2``).

### Estimation from multivariate time series/datasets

Although not dealt with in the original paper, numerically speaking, permutation entropy
can also be computed for multivariate datasets (either embedded or
consisting of multiple time series variables).

Then, just skip the delay reconstruction step, compute symbols directly from the ``L`` existing
state vectors ``\\{\\mathbf{x}_1, \\mathbf{x}_2, \\ldots, \\mathbf{x_L}\\}``, symbolize
each ``\\mathbf{x_i}`` precisely as above, then compute the
quantity

```math
H = - \\sum_j p(\\pi) \\ln p(\\pi_j).
```

- To compute permutation entropy for a multivariate/embedded dataset `x`, use the
    signature `entropy(x::AbstractDataset, est::SymbolicPermutation)`.

- To get the corresponding probability distribution for a multivariate/embedded dataset `x`,
    use `probabilities(x::AbstractDataset, est::SymbolicPermutation)`.

!!! warn "Dynamical interpretation"
    A dynamical interpretation of the permutation entropy does not necessarily hold if
    computing it on generic multivariate datasets. Method signatures for `Dataset`s are
    provided for convenience, and should only be applied if you understand the relation
    between your input data, the numerical value for the weighted permutation entropy, and
    its interpretation.

## Speeding up repeated computations

!!! tip
    A pre-allocated symbol array `s` can be provided to save some memory allocations if the
    probabilities are to be computed for multiple data sets.

    *Note: it is not the array that will hold the final probabilities that is pre-allocated,
    but the temporary integer array containing the symbolized data points. Thus, if
    provided, it is required that `length(x) == length(s)` if `x` is a Dataset, or
    `length(s) == length(x) - (m-1)τ` if `x` is a univariate signal that is to be embedded
    first*.

    Use the following signatures.

    ```julia
    probabilities!(s::Vector{Int}, x::AbstractDataset, est::SymbolicPermutation) → ps::Probabilities
    probabilities!(s::Vector{Int}, x::AbstractVector, est::SymbolicPermutation;  m::Int = 2, τ::Int = 1) → ps::Probabilities
    ```

[^BandtPompe2002]: Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural
    complexity measure for time series." Physical review letters 88.17 (2002): 174102.
[^Fadlallah2013]: Fadlallah, Bilal, et al. "Weighted-permutation entropy: A complexity
    measure for time series incorporating amplitude information." Physical
    Review E 87.2 (2013): 022911.
"""
struct SymbolicPermutation <: PermutationProbabilityEstimator
    τ
    m
    function SymbolicPermutation(; m::Int = 2, τ::Int = 1)
        m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
        new(τ, m)
    end
end

"""
    symbolize(x::AbstractDataset, est::SymbolicPermutation) → Vector{Int}

Symbolize the vectors in `x` using Algorithm 1 from Berger et al. (2019)[^Berger2019].

The symbol length is automatically determined from the dimension of the input data vectors.

## Example

Computing the order-5 permutation entropy for a 7-dimensional dataset.

```julia
using DelayEmbeddings, Entropies
D = Dataset([rand(7) for i = 1:1000])
symbolize(D, SymbolicPermutation(5))
```

[^Berger2019]: Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.
"""
function symbolize(x::AbstractDataset{m, T}, est::PermutationProbabilityEstimator) where {m, T}
    m >= 2 || error("Data must be at least 2-dimensional to compute the permutation entropy. If data is a univariate time series, embed it using `genembed` first.")
    s = zeros(Int, length(x))
    symbolize!(s, x, est)
    return s
end

function fill_symbolvector!(s, x, sp, N::Int)
    @inbounds for i = 1:length(x)
        sortperm!(sp, x[i])
        s[i] = encode_motif(sp, N)
    end
end

"""
    symbolize!(s::AbstractVector{Int}, x::AbstractDataset, est::SymbolicPermutation) → Vector{Int}

Symbolize the vectors in `x`, storing the symbols in the pre-allocated length-`L` integer
container `s`, where `L = length(x)`.
"""
function symbolize!(s::AbstractVector{Int}, x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    #=
    Loop over embedding vectors `E[i]`, find the indices `p_i` that sort each `E[i]`,
    then get the corresponding integers `k_i` that generated the
    permutations `p_i`. Those integers are the symbols for the embedding vectors
    `E[i]`.
    =#
    sp = zeros(Int, m) # pre-allocate a single symbol vector that can be overwritten.
    fill_symbolvector!(s, x, sp, m)

    return s
end

function probabilities!(s::Vector{Int}, x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    length(s) == length(x) || throw(ArgumentError("Need length(s) == length(x), got `length(s)=$(length(s))` and `length(x)==$(length(x))`."))
    m >= 2 || error("Data must be at least 2-dimensional to compute the permutation entropy. If data is a univariate time series embed it using `genembed` first.")

    @inbounds for i = 1:length(x)
        s[i] = encode_motif(x[i], m)
    end
    probabilities(s)
end

function probabilities!(s::Vector{Int}, x::AbstractVector{T}, est::SymbolicPermutation) where {T<:Real}
    L = length(x)
    N = L - (est.m-1)*est.τ
    length(s) == N || error("Pre-allocated symbol vector `s`needs to have length `length(x) - (m-1)*τ` to match the number of state vectors after `x` has been embedded. Got length(s)=$(length(s)) and length(x)=$(L).")

    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)

    probabilities!(s, x_emb, est)
end

"""
# Permutation-based symbol probabilities

    probabilities(x::AbstractDataset, est::SymbolicPermutation) → ps::Probabilities
    probabilities(x::AbstractVector, est::SymbolicPermutation;  m::Int = 2, τ::Int = 1) → ps::Probabilities

    probabilities!(s::Vector{Int}, x::AbstractDataset, est::SymbolicPermutation) → ps::Probabilities
    probabilities!(s::Vector{Int}, x::AbstractVector, est::SymbolicPermutation;  m::Int = 2, τ::Int = 1) → ps::Probabilities

Compute the unordered probabilities of the occurrence of symbol sequences constructed from
the data `x`.

If `x` is a multivariate `Dataset`, then symbolization is performed directly on the state
vectors. If `x` is a univariate signal, then a delay reconstruction with embedding lag `τ`
and embedding dimension `m` is used to construct state vectors, on which symbolization is
then performed.

A pre-allocated symbol array `s` can be provided to save some memory allocations if the
probabilities are to be computed for multiple data sets. If provided, it is required that
`length(x) == length(s)` if `x` is a `Dataset`, or  `length(s) == length(x) - (m-1)τ`
if `x` is a univariate signal.

See also: [`SymbolicPermutation`](@ref).
"""
function probabilities(x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    s = zeros(Int, length(x))
    probabilities!(s, x, est)
end

function probabilities(x::AbstractVector{T}, est::SymbolicPermutation) where {T<:Real}
    est.m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)

    s = zeros(Int, length(x_emb))
    probabilities!(s, x_emb, est)
end

function genentropy!(s::Vector{Int}, x::AbstractDataset{m, T}, est::SymbolicPermutation; α::Real = 1,
        base::Real = 2) where {m, T}

    length(s) == length(x) || error("Pre-allocated symbol vector s need the same number of elements as x. Got length(s)=$(length(s)) and length(x)=$(L).")
    ps = probabilities!(s, x, est)

    genentropy(ps, α = α, base = base)
end

function genentropy!(s::Vector{Int}, x::Vector{T}, est::SymbolicPermutation; α::Real = 1,
        base::Real = 2) where {T<:Real}

    L = length(x)
    N = L - (est.m-1)*est.τ
    length(s) == N || error("Pre-allocated symbol vector `s`needs to have length `length(x) - (m-1)*τ` to match the number of state vectors after `x` has been embedded. Got length(s)=$(length(s)) and length(x)=$(L).")

    ps = probabilities!(s, x, est)
    genentropy(ps, α = α, base = base)
end

"""
# Permutation entropy

    genentropy(x::AbstractDataset, est::SymbolicPermutation; α::Real = 1, base = 2) → Real
    genentropy(x::AbstractVector{<:Real}, est::SymbolicPermutation; α::Real = 1, base = 2) → Real

    genentropy!(s::Vector{Int}, x::AbstractDataset, est::SymbolicPermutation;
        α::Real = 1, base = 2) → Real
    genentropy!(s::Vector{Int}, x::AbstractVector{<:Real}, est::SymbolicPermutation;
        α::Real = 1; m::Int = 3, τ::Int = 1, base = 2) → Real

Compute the generalized order-`α` entropy over a permutation symbolization of `x`, using
symbol size/order `m`.

If `x` is a multivariate `Dataset`, then symbolization is performed directly on the state
vectors. If `x` is a univariate signal, then a delay reconstruction with embedding lag `τ`
and embedding dimension `m` is used to construct state vectors, on which symbolization is
then performed.

A pre-allocated symbol array `s` can be provided to save some memory allocations if
probabilities are to be computed for multiple data sets. If provided, it is required that
`length(x) == length(s)` if `x` is a `Dataset`, or  `length(s) == length(x) - (m-1)τ`
if `x` is a univariate signal.

## Probability and entropy estimation

An unordered symbol frequency histogram is obtained by symbolizing the points in `x`,
using [`probabilities(::AbstractDataset, ::SymbolicPermutation)`](@ref).
Sum-normalizing this histogram yields a probability distribution over the symbols.

After the symbolization histogram/distribution has been obtained, the order `α` generalized
entropy[^Rényi1960], to the given `base`, is computed from that sum-normalized symbol
distribution, using [`genentropy`](@ref).

!!! hint "Generalized entropy order vs. permutation order"
    Do not confuse the order of the generalized entropy (`α`) with the order `m` of the
    permutation entropy (`m`, which controls the symbol size). Permutation entropy is usually
    estimated with `α = 1`, but the implementation here allows the generalized entropy of any
    dimension to be computed from the symbol frequency distribution.

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960)

See also: [`SymbolicPermutation`](@ref), [`genentropy`](@ref).
"""
# function genentropy(x::AbstractDataset{N, T}, est::SymbolicPermutation; α::Real = 1, base::Real = 2) where {N, T}
#     s = zeros(Int, length(x))
#     genentropy!(s, x, est; α = α, base = base)
# end


# function genentropy(x::AbstractArray{T}, est::SymbolicPermutation; α::Real = 1,
#         m::Int = 3, τ::Int = 1, base = 2) where {T<:Real}

#     N = length(x)
#     s = zeros(Int, N - (m-1)*τ)
#     ps = probabilities!(s, x, est, m = m, τ = τ)
#     genentropy(ps; α = α, base = base)
# end
