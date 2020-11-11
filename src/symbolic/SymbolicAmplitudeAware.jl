export SymbolicAmplitudeAwarePermutation, probabilities

"""
    SymbolicAmplitudeAwarePermutation <: PermutationProbabilityEstimator


A symbolic, amplitude-aware permutation based probabilities/entropy estimator.

## Properties of original signal preserved

Amplitude-aware permutations of a signal preserve not only ordinal patterns (sorting
information), but also encodes amplitude information. This implementation is based on Azami & Escudero
(2016) [^Azami2016].

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

Amplitude-aware permutation entropy is computed analogously to regular permutation entropy, but
adds weights that encode amplitude information too:

```math
p(\\pi_i^{m, \\tau}) = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, a_k}{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) \\in \\Pi} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\,a_k} = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, a_k}{\\sum_{k=1}^N a_k}.
```

The weight encoding amplitude information about state vector ``\\mathbf{x}_i = (x_1^i, x_2^i, \\ldots, x_m^i)`` are

```math
a_i = \\dfrac{A}{m} \\sum_{k=1}^m |x_k^i | + \\dfrac{1-A}{d-1} \\sum_{k=2}^d |x_{k}^i - x_{k-1}^i|,
```

with ``0 \\leq A \\leq 1``. When ``A=0`` , only internal differences between the elements of
``\\mathbf{x}_i`` are weighted. Only mean amplitude of the state vector
elements are weighted when ``A=1``. With, ``0<A<1``, a combined weighting is used.

### Estimation from univariate time series/datasets

- To compute amplitude-aware permutation entropy for a univariate signal `x`, use the signature
    `entropy(x::AbstractVector, est::SymbolicAmplitudeAwarePermutation; τ::Int = 1, m::Int = 3)`.

- The corresponding (unordered) probability distribution of the permutation symbols for a
    univariate signal `x` can be computed using `probabilities(x::AbstractVector,
    est::SymbolicAmplitudeAwarePermutation; τ::Int = 1, m::Int = 3)`.

!!! info "Default embedding dimension and embedding lag"
    By default, embedding dimension ``m = 3`` with embedding lag ``\\tau = 1`` is used when
    embedding a time series for symbolization. You should probably make a more informed
    decision about embedding parameters when computing the permutation entropy of a real
    time series. In all cases, ``m`` must be at least 2 (there are
    no permutations of a single-element state vector, so need ``m \\geq 2``).

### Estimation from multivariate time series/datasets

Although not dealt with in the original paper, numerically speaking, amplitude-aware
permutation entropy, just like regular permutation entropy, can also be computed for
multivariate datasets (either embedded or consisting of multiple time series
variables).

Then, just skip the delay reconstruction step, compute symbols
directly from the ``L`` existing state vectors
``\\{\\mathbf{x}_1, \\mathbf{x}_2, \\ldots, \\mathbf{x_L}\\}``, symbolize
each ``\\mathbf{x_i}`` precisely as above, then compute the
quantity

```math
H = - \\sum_j p(\\pi) \\ln p(\\pi_j).
```

- To compute amplitude-aware permutation entropy for a multivariate/embedded dataset `x`, use the
    signature `entropy(x::AbstractDataset, est::SymbolicAmplitudeAwarePermutation)`.

- To get the corresponding probability distribution for a multivariate/embedded dataset `x`, use
    `probabilities(x::AbstractDataset, est::SymbolicAmplitudeAwarePermutation)`.


!!! warn "Dynamical interpretation"
    A dynamical interpretation of the permutation entropy does not necessarily hold if
    computing it on generic multivariate datasets. Method signatures for `Dataset`s are
    provided for convenience, and should only be applied if you understand the relation
    between your input data, the numerical value for the weighted permutation entropy, and
    its interpretation.

[^Azami2016]: Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy: Illustration in spike detection and signal segmentation. Computer methods and programs in biomedicine, 128, 40-51.

"""
struct SymbolicAmplitudeAwarePermutation <: PermutationProbabilityEstimator end

"""
    AAPE(x, A::Real = 0.5, m::Int = length(a))

Encode relative amplitude information of the elements of `a`.
- `A = 1` emphasizes only average values.
- `A = 0` emphasizes changes in amplitude values.
- `A = 0.5` equally emphasizes average values and changes in the amplitude values.
"""
function AAPE(x; A::Real = 0.5, m::Int = length(x))
    (A/m)*sum(abs.(x)) + (1-A)/(m-1)*sum(abs.(diff(x)))
end

"""
# Amplitude-aware permutation-based symbol probabilities

    probabilities(x::AbstractDataset, est::SymbolicAmplitudeAwarePermutation) → ps::Probabilities
    probabilities(x::AbstractVector{<:Real}, est::SymbolicAmplitudeAwarePermutation; m::Int = 3, τ::Int = 1) → ps::Probabilities

    probabilities!(s::Vector{Int}, x::AbstractDataset, est::SymbolicAmplitudeAwarePermutation) → ps::Probabilities
    probabilities!(s::Vector{Int}, x::AbstractVector, est::SymbolicAmplitudeAwarePermutation; m::Int = 3, τ::Int = 1) → ps::Probabilities

Compute the unordered probabilities of the occurrence of amplitude-encoding symbol sequences
constructed from `x`.

If `x` is a multivariate `Dataset`, then symbolization is performed directly on the state
vectors. If `x` is a univariate signal, then a delay reconstruction with embedding lag `τ`
and embedding dimension `m` is used to construct state vectors, on which symbolization is
then performed.

A pre-allocated symbol array `s` can be provided to save some memory allocations if the
probabilities are to be computed for multiple data sets. If provided, it is required that
`length(x) == length(s)` if `x` is a `Dataset`, or  `length(s) == length(x) - (m-1)τ`
if `x` is a univariate signal`.

See also: [`SymbolicAmplitudeAwarePermutation`](@ref).
"""
function probabilities(x::AbstractDataset{m, T}, est::SymbolicAmplitudeAwarePermutation; A::Real = 0.5) where {m, T}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    πs = symbolize(x, SymbolicPermutation())
    wts = AAPE.(x.data, A = A, m = m)

    Probabilities(probs(πs, wts, normalize = true))
end

function probabilities(x::AbstractVector{T}, est::SymbolicAmplitudeAwarePermutation;
        A::Real = 0.5, m::Int = 3, τ::Int = 1) where {T<:Real}

    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    τs = tuple([τ*i for i = 0:m-1]...)
    emb = genembed(x, τs)
    πs = symbolize(emb, SymbolicPermutation())
    wts = AAPE.(emb.data, A = A, m = m)

    Probabilities(probs(πs, wts, normalize = true))
end

"""
# Amplitude-aware permutation entropy

    genentropy(x::AbstractDataset, est::SymbolicAmplitudeAwarePermutation; α::Real = 1, base = 2) → Real
    genentropy(x::AbstractVector{<:Real}, est::SymbolicAmplitudeAwarePermutation;
        α::Real = 1, m::Int = 3, τ::Int = 1, base = 2) → Real

Compute the generalized order `α` entropy based on an amplitude-sensitive permutation
symbolization of `x`, using symbol size/order `m` for the permutations.

If `x` is a multivariate `Dataset`, then symbolization is performed directly on the state
vectors. If `x` is a univariate signal, then a delay reconstruction with embedding lag `τ`
and embedding dimension `m` is used to construct state vectors, on which symbolization is
then performed.

## Probability and entropy estimation

An unordered symbol frequency histogram is obtained by symbolizing the points in `x` by
an amplitude-aware procedure, using
[`probabilities(::AbstractDataset, ::SymbolicAmplitudeAwarePermutation)`](@ref).
Sum-normalizing this histogram yields a probability distribution over the amplitude-encoding
 symbols.

After the symbolization histogram/distribution has been obtained, the order `α` generalized
entropy[^Rényi1960], to the given `base`, is computed from that sum-normalized symbol
distribution, using [`genentropy`](@ref).

!!! hint "Generalized entropy order vs. permutation order"
    Do not confuse the order of the generalized entropy (`α`) with the order `m` of the
    permutation entropy (`m`, which controls the symbol size). Permutation entropy is usually
    estimated with `α = 1`, but the implementation here allows the generalized entropy of any
    dimension to be computed from the symbol frequency distribution.

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics,
    Statistics and Probability*, pp 547 (1960)

See also: [`SymbolicAmplitudeAwarePermutation`](@ref), [`genentropy`](@ref).
"""
function genentropy(x::AbstractDataset{m, T}, est::SymbolicAmplitudeAwarePermutation;
        α::Real = 1, A::Real = 0.5, base = 2) where {m, T}

    ps = probabilities(x, est, A = A)
    genentropy(ps, α = α, base = base)
end

function genentropy(x::AbstractArray{T}, est::SymbolicAmplitudeAwarePermutation;
        α::Real = 1, A::Real = 0.5, m::Int = 3, τ::Int = 1, base = 2) where {T<:Real}

    ps = probabilities(x, est, A = A, m = m, τ = τ)
    genentropy(ps, α = α, base = base)
end
