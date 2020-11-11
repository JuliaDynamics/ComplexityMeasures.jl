export SymbolicAmplitudeAwarePermutation

"""
    SymbolicAmplitudeAwarePermutation(; τ = 1, m = 3, A = 0.5) <: PermutationProbabilityEstimator


A symbolic, amplitude-aware permutation based probabilities/entropy estimator.

## Properties of original signal preserved

Amplitude-aware permutations of a signal preserve not only ordinal patterns (sorting 
information), but also encodes amplitude information (see description below for explanation 
of the parameter `A`). This implementation is based on Azami & Escudero (2016) [^Azami2016].

## Estimation

### Univariate time series

To estimate probabilities or entropies from univariate time series, use the following methods:

- `probabilities(x::AbstractVector, est::SymbolicAmplitudeAwarePermutation)`. Constructs state vectors 
    from `x` using embedding lag `τ` and embedding dimension `m`. The ordinal patterns of the 
    state vectors are then symbolized, and probabilities are taken as the relative 
    frequency of symbols.
- `genentropy(x::AbstractVector, est::SymbolicAmplitudeAwarePermutation; α=1, base = 2)` computes
    probabilities by calling `probabilities(x::AbstractVector, est::SymbolicAmplitudeAwarePermutation)`,
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

As for regular permutation entropy, numerically speaking, 
amplitude-adjusted permutation entropy can also be computed for multivariate datasets with 
dimension ≥ 2. Such
datasets may be, for example, preembedded time series. Then, just skip the delay 
reconstruction step, compute and symbols directly from the ``L`` existing state vectors 
``\\{\\mathbf{x}_1, \\mathbf{x}_2, \\ldots, \\mathbf{x_L}\\}``.

- `probabilities(x::Dataset, est::SymbolicAmplitudeAwarePermutation)`. Compute ordinal patterns of the 
    state vectors of `x` directly (without doing any embedding), symbolize those patterns,
    and compute probabilities as relative frequencies of symbols.
- `genentropy(x::Dataset, est::SymbolicAmplitudeAwarePermutation)`. Computes probabilities from 
    symbol frequencies using `probabilities(x::Dataset, est::SymbolicAmplitudeAwarePermutation)`,
    then computes the order-`α` generalized (permutation) entropy to the given base.

!!! warn "Dynamical interpretation"
    A dynamical interpretation of the permutation entropy does not necessarily hold if
    computing it on generic multivariate datasets. Method signatures for `Dataset`s are
    provided for convenience, and should only be applied if you understand the relation
    between your input data, the numerical value for the permutation entropy, and
    its interpretation.

!!! hint "Generalized entropy order vs. permutation order"
    Do not confuse the order of the generalized entropy (`α`) with the order `m` of the
    permutation entropy (`m`, which controls the symbol size). Amplitude-adjusted 
    permutation entropy is usually estimated with `α = 1`, but the implementation here 
    allows the generalized entropy of any dimension to be computed from the symbol 
    frequency distribution.

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

Amplitude-aware permutation entropy is computed analogously to regular permutation entropy
(see [`SymbolicPermutation`](@ref)), but probabilities are weighted by amplitude information as follows.

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

[^Azami2016]: Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy: Illustration in spike detection and signal segmentation. Computer methods and programs in biomedicine, 128, 40-51.

"""
struct SymbolicAmplitudeAwarePermutation <: PermutationProbabilityEstimator
    τ
    m
    A
    function SymbolicAmplitudeAwarePermutation(; τ::Int = 1, m::Int = 2, A::Real = 0.5)
        2 ≤ m || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
        0 ≤ A ≤ 1 || error("Weighting factor A must be on interval [0, 1]. Got A=$A.")
        new(τ, m, A)
    end
end

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

function probabilities(x::AbstractDataset{m, T}, est::SymbolicAmplitudeAwarePermutation) where {m, T}
    πs = symbolize(x, SymbolicPermutation(m = m)) # motif length controlled by dimension of input data
    wts = AAPE.(x.data, A = est.A, m = est.m)

    Probabilities(probs(πs, wts, normalize = true))
end

function probabilities(x::AbstractVector{T}, est::SymbolicAmplitudeAwarePermutation) where {T<:Real}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    emb = genembed(x, τs)
    πs = symbolize(emb, SymbolicPermutation(m = est.m))  # motif length controlled by estimator m
    wts = AAPE.(emb.data, A = est.A, m = est.m)

    Probabilities(probs(πs, wts, normalize = true))
end
