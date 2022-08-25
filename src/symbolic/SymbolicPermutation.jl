export SymbolicPermutation

"""
A probability estimator based on permutations.
"""
abstract type PermutationProbabilityEstimator <: SymbolicProbabilityEstimator end

"""
    SymbolicPermutation(; τ = 1, m = 3, lt = Entropies.isless_rand) <: ProbabilityEstimator
    SymbolicWeightedPermutation(; τ = 1, m = 3, lt = Entropies.isless_rand) <: ProbabilityEstimator
    SymbolicAmplitudeAwarePermutation(; τ = 1, m = 3, A = 0.5, lt = Entropies.isless_rand) <: ProbabilityEstimator

Symbolic, permutation-based probabilities/entropy estimators.
`m` is the permutation order (or the symbol size or the embedding dimension) and
`τ` is the delay time (or lag).

## Repeated values during symbolization

In the original implementation of permutation entropy [^BandtPompe2002], equal values are
ordered after their order of appearance, but this can lead to erroneous temporal
correlations, especially for data with low-amplitude resolution [^Zunino2017]. Here, we
resolve this issue by letting the user provide a custom "less-than" function. The keyword
`lt` accepts a function that decides which of two state vector elements are smaller. If two
elements are equal, the default behaviour is to randomly assign one of them as the largest
(`lt = Entropies.isless_rand`). For data with low amplitude resolution, computing
probabilities multiple times using the random approach may reduce these erroneous
effects.

To get the behaviour described in Bandt and Pompe (2002), use `lt = Base.isless`).

## Properties of original signal preserved

- **`SymbolicPermutation`**: Preserves ordinal patterns of state vectors (sorting information). This
    implementation is based on Bandt & Pompe et al. (2002)[^BandtPompe2002] and
    Berger et al. (2019) [^Berger2019].
- **`SymbolicWeightedPermutation`**: Like `SymbolicPermutation`, but also encodes amplitude
    information by tracking the variance of the state vectors. This implementation is based
    on Fadlallah et al. (2013)[^Fadlallah2013].
- **`SymbolicAmplitudeAwarePermutation`**: Like `SymbolicPermutation`, but also encodes
    amplitude information by considering a weighted combination of *absolute amplitudes*
    of state vectors, and *relative differences between elements* of state vectors. See
    description below for explanation of the weighting parameter `A`. This implementation
    is based on Azami & Escudero (2016) [^Azami2016].

## Probability estimation

### Univariate time series

To estimate probabilities or entropies from univariate time series, use the following methods:

- `probabilities(x::AbstractVector, est::SymbolicProbabilityEstimator)`. Constructs state vectors
    from `x` using embedding lag `τ` and embedding dimension `m`, symbolizes state vectors,
    and computes probabilities as (weighted) relative frequency of symbols.
- `genentropy(x::AbstractVector, est::SymbolicProbabilityEstimator; α=1, base = 2)` computes
    probabilities by calling `probabilities(x::AbstractVector, est)`,
    then computer the order-`α` generalized entropy to the given base.

#### Speeding up repeated computations

A pre-allocated integer symbol array `s` can be provided to save some memory
allocations if the probabilities are to be computed for multiple data sets.

*Note: it is not the array that will hold the final probabilities that is pre-allocated,
but the temporary integer array containing the symbolized data points. Thus, if
provided, it is required that `length(x) == length(s)` if `x` is a Dataset, or
`length(s) == length(x) - (m-1)τ` if `x` is a univariate signal that is to be embedded
first*.

Use the following signatures (only works for `SymbolicPermutation`).

```julia
probabilities!(s::Vector{Int}, x::AbstractVector, est::SymbolicPermutation) → ps::Probabilities
probabilities!(s::Vector{Int}, x::AbstractDataset, est::SymbolicPermutation) → ps::Probabilities
```

### Multivariate datasets

Although not dealt with in the original paper describing the estimators, numerically speaking,
permutation entropies can also be computed for multivariate datasets with dimension ≥ 2
(but see caveat below). Such datasets may be, for example, preembedded time series. Then,
just skip the delay reconstruction step, compute and symbols directly from the ``L``
existing state vectors ``\\{\\mathbf{x}_1, \\mathbf{x}_2, \\ldots, \\mathbf{x_L}\\}``.

- `probabilities(x::AbstractDataset, est::SymbolicProbabilityEstimator)`. Compute ordinal patterns of the
    state vectors of `x` directly (without doing any embedding), symbolize those patterns,
    and compute probabilities as (weighted) relative frequencies of symbols.
- `genentropy(x::AbstractDataset, est::SymbolicProbabilityEstimator)`. Computes probabilities from
    symbol frequencies using `probabilities(x::AbstractDataset, est::SymbolicProbabilityEstimator)`,
    then computes the order-`α` generalized (permutation) entropy to the given base.

*Caveat: A dynamical interpretation of the permutation entropy does not necessarily
hold if computing it on generic multivariate datasets. Method signatures for `Dataset`s are
provided for convenience, and should only be applied if you understand the relation
between your input data, the numerical value for the permutation entropy, and
its interpretation.*

## Description

All symbolic estimators use the same underlying approach to estimating probabilities.

### Embedding, ordinal patterns and symbolization

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

### Probability computation

#### `SymbolicPermutation`

The probability of a given motif is its frequency of occurrence, normalized by the total
number of motifs (with notation from [^Fadlallah2013]),

```math
p(\\pi_i^{m, \\tau}) = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left(\\mathbf{x}_k^{m, \\tau} \\right) }{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) \\in \\Pi} \\left(\\mathbf{x}_k^{m, \\tau} \\right)} = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left(\\mathbf{x}_k^{m, \\tau} \\right) }{N},
```

where the function ``\\mathbf{1}_A(u)`` is the indicator function of a set ``A``. That
    is, ``\\mathbf{1}_A(u) = 1`` if ``u \\in A``, and ``\\mathbf{1}_A(u) = 0`` otherwise.

#### `SymbolicAmplitudeAwarePermutation`


Amplitude-aware permutation entropy is computed analogously to regular permutation entropy
but probabilities are weighted by amplitude information as follows.

```math
p(\\pi_i^{m, \\tau}) = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, a_k}{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) \\in \\Pi} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\,a_k} = \\dfrac{\\sum_{k=1}^N \\mathbf{1}_{u:S(u) = s_i} \\left( \\mathbf{x}_k^{m, \\tau} \\right) \\, a_k}{\\sum_{k=1}^N a_k}.
```

The weights encoding amplitude information about state vector ``\\mathbf{x}_i = (x_1^i, x_2^i, \\ldots, x_m^i)`` are

```math
a_i = \\dfrac{A}{m} \\sum_{k=1}^m |x_k^i | + \\dfrac{1-A}{d-1} \\sum_{k=2}^d |x_{k}^i - x_{k-1}^i|,
```

with ``0 \\leq A \\leq 1``. When ``A=0`` , only internal differences between the elements of
``\\mathbf{x}_i`` are weighted. Only mean amplitude of the state vector
elements are weighted when ``A=1``. With, ``0<A<1``, a combined weighting is used.

#### `SymbolicWeightedPermutation`


Weighted permutation entropy is also computed analogously to regular permutation entropy, but
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

[^BandtPompe2002]: Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural
    complexity measure for time series." Physical review letters 88.17 (2002): 174102.
[^Berger2019]: Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.
[^Fadlallah2013]: Fadlallah, Bilal, et al. "Weighted-permutation entropy: A complexity
    measure for time series incorporating amplitude information." Physical
    Review E 87.2 (2013): 022911.
[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960)
[^Azami2016]: Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy: Illustration in spike detection and signal segmentation. Computer methods and programs in biomedicine, 128, 40-51.
[^Fadlallah2013]: Fadlallah, Bilal, et al. "Weighted-permutation entropy: A complexity measure for time series incorporating amplitude information." Physical Review E 87.2 (2013): 022911.
[^Zunino2017]: Zunino, L., Olivares, F., Scholkmann, F., & Rosso, O. A. (2017). Permutation entropy based time series analysis: Equalities in the input signal can lead to false conclusions. Physics Letters A, 381(22), 1883-1892.
"""
struct SymbolicPermutation{F} <: PermutationProbabilityEstimator
    τ::Int
    m::Int
    lt::F
end
function SymbolicPermutation(; τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F <: Function}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    SymbolicPermutation{F}(τ, m, lt)
end

function symbolize(x::AbstractDataset{m, T}, est::PermutationProbabilityEstimator) where {m, T}
    m >= 2 || error("Data must be at least 2-dimensional to symbolize. If data is a univariate time series, embed it using `genembed` first.")
    s = zeros(Int, length(x))
    symbolize!(s, x, est)
    return s
end

function symbolize(x::AbstractVector{T}, est::PermutationProbabilityEstimator) where {T}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)

    s = zeros(Int, length(x_emb))
    symbolize!(s, x_emb, est)
    return s
end

function fill_symbolvector!(s, x, sp, m::Int; lt::Function = isless_rand)
    @inbounds for i = 1:length(x)
        sortperm!(sp, x[i], lt = lt)
        s[i] = encode_motif(sp, m)
    end
end

function symbolize!(s::AbstractVector{Int}, x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    @assert length(s) == length(x)
    #=
    Loop over embedding vectors `E[i]`, find the indices `p_i` that sort each `E[i]`,
    then get the corresponding integers `k_i` that generated the
    permutations `p_i`. Those integers are the symbols for the embedding vectors
    `E[i]`.
    =#
    sp = zeros(Int, m) # pre-allocate a single symbol vector that can be overwritten.
    fill_symbolvector!(s, x, sp, m, lt = est.lt)

    return s
end

function symbolize!(s::AbstractVector{Int}, x::AbstractVector{T}, est::SymbolicPermutation) where T
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)
    symbolize!(s, x_emb, est)
end

function probabilities!(s::AbstractVector{Int}, x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    length(s) == length(x) || throw(ArgumentError("Need length(s) == length(x), got `length(s)=$(length(s))` and `length(x)==$(length(x))`."))
    m >= 2 || error("Data must be at least 2-dimensional to compute the permutation entropy. If data is a univariate time series embed it using `genembed` first.")

    @inbounds for i = 1:length(x)
        s[i] = encode_motif(x[i], m)
    end
    probabilities(s)
end

function probabilities!(s::AbstractVector{Int}, x::AbstractVector{T}, est::SymbolicPermutation) where {T<:Real}
    L = length(x)
    N = L - (est.m-1)*est.τ
    length(s) == N || error("Pre-allocated symbol vector `s`needs to have length `length(x) - (m-1)*τ` to match the number of state vectors after `x` has been embedded. Got length(s)=$(length(s)) and length(x)=$(L).")

    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)

    probabilities!(s, x_emb, est)
end

function probabilities(x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T}
    s = zeros(Int, length(x))
    probabilities!(s, x, est)
end

function probabilities(x::AbstractVector{T}, est::SymbolicPermutation) where {T<:Real}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    x_emb = genembed(x, τs)

    s = zeros(Int, length(x_emb))
    probabilities!(s, x_emb, est)
end

function genentropy!(
    s::AbstractVector{Int}, x::AbstractDataset{m, T}, est::SymbolicPermutation;
    q = 1.0, α = nothing, base::Real = MathConstants.e
    ) where {m, T}

    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    length(s) == length(x) || error("Pre-allocated symbol vector s need the same number of elements as x. Got length(s)=$(length(s)) and length(x)=$(L).")
    ps = probabilities!(s, x, est)

    genentropy(ps, α = α, base = base)
end

function genentropy!(
        s::AbstractVector{Int}, x::AbstractVector{T}, est::SymbolicPermutation;
        q::Real = 1.0, α = nothing, base::Real = MathConstants.e
    ) where {T<:Real}
    if α ≠ nothing
        @warn "Keyword `α` is deprecated in favor of `q`."
        q = α
    end
    L = length(x)
    N = L - (est.m-1)*est.τ
    length(s) == N || error("Pre-allocated symbol vector `s` needs to have length `length(x) - (m-1)*τ` to match the number of state vectors after `x` has been embedded. Got length(s)=$(length(s)) and length(x)=$(L).")

    ps = probabilities!(s, x, est)
    genentropy(ps, α = α, base = base)
end
