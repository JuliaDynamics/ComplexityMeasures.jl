export OrdinalPatterns
export WeightedOrdinalPatterns
export AmplitudeAwareOrdinalPatterns
using DelayEmbeddings: embed

"""
The supertype for probability estimators based on permutation patterns.

Subtypes must implement fields:

- `m::Int`: The dimension of the permutation patterns.
- `lt::Function`: A function determining how ties are to be broken when constructing
    permutation patterns from embedding vectors.
"""
abstract type PermutationOutcomeSpace{m} <: CountBasedOutcomeSpace end
const PermProbEst = PermutationOutcomeSpace

###########################################################################################
# Types and docstrings
###########################################################################################
"""
    OrdinalPatterns <: OutcomeSpace
    OrdinalPatterns(; m = 3, τ = 1, lt::Function = ComplexityMeasures.isless_rand)

An [`OutcomeSpace`](@ref) based on ordinal permutation patterns, originally introduced in
[BandtPompe2002](@citet)'s paper on permutation entropy.

When passed to [`probabilities`](@ref) the output depends on the input data type:

- **Univariate data**. If applied to a univariate timeseries (`AbstractVector`), then the timeseries
    is first embedded using embedding delay `τ` and dimension `m`, resulting in embedding
    vectors ``\\{ \\bf{x}_i \\}_{i=1}^{N-(m-1)\\tau}``. Then, for each ``\\bf{x}_i``,
    we find its permutation pattern ``\\pi_{i}``. Probabilities are then
    estimated as the frequencies of the encoded permutation symbols
    by using [`CountOccurrences`](@ref). When giving the resulting probabilities to
    [`information`](@ref), the original permutation entropy is computed [BandtPompe2002](@cite).
- **Multivariate data**. If applied to a an `D`-dimensional `StateSpaceSet`,
    then no embedding is constructed, `m` must be equal to `D` and `τ` is ignored.
    Each vector ``\\bf{x}_i`` of the dataset is mapped
    directly to its permutation pattern ``\\pi_{i}`` by comparing the
    relative magnitudes of the elements of ``\\bf{x}_i``.
    Like above, probabilities are estimated as the frequencies of the permutation symbols.
    The resulting probabilities can be used to compute multivariate permutation
    entropy [He2016](@cite), although here we don't perform any further subdivision
    of the permutation patterns (as in Figure 3 of [He2016](@citet)).

Internally, [`OrdinalPatterns`](@ref) uses the [`OrdinalPatternEncoding`](@ref)
to represent ordinal patterns as integers for efficient computations.

See [`WeightedOrdinalPatterns`](@ref) and [`AmplitudeAwareOrdinalPatterns`](@ref)
for estimators that not only consider ordinal (sorting) patterns, but also incorporate
information about within-state-vector amplitudes.
For a version of this estimator that can be used on spatial data, see
[`SpatialOrdinalPatterns`](@ref).

!!! note "Handling equal values in ordinal patterns"
    In [BandtPompe2002](@citet), equal values are ordered after their order of appearance, but
    this can lead to erroneous temporal correlations, especially for data with
    low amplitude resolution [Zunino2017](@cite). Here, by default, if two values are equal,
    then one of the is randomly assigned as "the largest", using
    `lt = ComplexityMeasures.isless_rand`.
    To get the behaviour from [BandtPompe2002](@citet), use
    `lt = Base.isless`.

## Outcome space

The outcome space `Ω` for `OrdinalPatterns` is the set of length-`m` ordinal
patterns (i.e. permutations) that can be formed by the integers `1, 2, …, m`.
There are `factorial(m)` such patterns.

For example, the outcome `[2, 3, 1]` corresponds to the ordinal pattern of having
the smallest value in the second position, the next smallest value in the third
position, and the next smallest, i.e. the largest value in the first position.
See also [`OrdinalPatternEncoding`](@ref).

## In-place symbolization

`OrdinalPatterns` also implements the in-place [`probabilities!`](@ref)
for `StateSpaceSet` input (or embedded vector input) for reducing allocations in looping
scenarios. The length of the pre-allocated symbol vector must be the length of the dataset.
For example

```julia
using ComplexityMeasures
m, N = 2, 100
est = OrdinalPatterns(; m, τ)
x = StateSpaceSet(rand(N, m)) # some input dataset
πs_ts = zeros(Int, N) # length must match length of `x`
p = probabilities!(πs_ts, est, x)
```
"""
struct OrdinalPatterns{M,F} <: PermutationOutcomeSpace{M}
    encoding::OrdinalPatternEncoding{M,F}
    τ::Int
end

"""
    WeightedOrdinalPatterns <: OutcomeSpace
    WeightedOrdinalPatterns(; τ = 1, m = 3, lt::Function = ComplexityMeasures.isless_rand)

A variant of [`OrdinalPatterns`](@ref) that also incorporates amplitude information,
based on the weighted permutation entropy [Fadlallah2013](@cite). The outcome space and
keywords are the same as in [`OrdinalPatterns`](@ref).

## Description

For each ordinal pattern extracted from each state (or delay) vector, a weight is attached
to it which is the variance of the vector. Probabilities are then estimated by summing
the weights corresponding to the same pattern, instead of just counting the occurrence
of the same pattern.

!!! note "An implementation note"
    *Note: in equation 7, section III, of the original paper, the authors write*

    ```math
    w_j = \\dfrac{1}{m}\\sum_{k=1}^m (x_{j-(k-1)\\tau} - \\mathbf{\\hat{x}}_j^{m, \\tau})^2.
    ```
    *But given the formula they give for the arithmetic mean, this is **not** the variance
    of the delay vector ``\\mathbf{x}_i``, because the indices are mixed:
    ``x_{j+(k-1)\\tau}`` in the weights formula, vs. ``x_{j+(k+1)\\tau}`` in the arithmetic
    mean formula. Here, delay embedding and computation of the patterns and their weights
    are completely separated processes, ensuring that we compute the arithmetic mean
    correctly for each vector of the input dataset (which may be a delay-embedded
    timeseries).
"""
struct WeightedOrdinalPatterns{M,F} <: PermutationOutcomeSpace{M}
    encoding::OrdinalPatternEncoding{M,F}
    τ::Int
end

is_counting_based(o::WeightedOrdinalPatterns) = false

"""
    AmplitudeAwareOrdinalPatterns <: OutcomeSpace
    AmplitudeAwareOrdinalPatterns(; τ = 1, m = 3, A = 0.5, lt = ComplexityMeasures.isless_rand)

A variant of [`OrdinalPatterns`](@ref) that also incorporates amplitude information,
based on the amplitude-aware permutation entropy [Azami2016](@cite). The outcome space and
keywords are the same as in [`OrdinalPatterns`](@ref).

## Description

Similarly to [`WeightedOrdinalPatterns`](@ref), a weight ``w_i`` is attached to each
ordinal pattern extracted from each state (or delay) vector
``\\mathbf{x}_i = (x_1^i, x_2^i, \\ldots, x_m^i)`` as

```math
w_i = \\dfrac{A}{m} \\sum_{k=1}^m |x_k^i | + \\dfrac{1-A}{d-1}
\\sum_{k=2}^d |x_{k}^i - x_{k-1}^i|,
```

with ``0 \\leq A \\leq 1``. When ``A=0`` , only internal differences between the
elements of
``\\mathbf{x}_i`` are weighted. Only mean amplitude of the state vector
elements are weighted when ``A=1``. With, ``0<A<1``, a combined weighting is used.
"""
struct AmplitudeAwareOrdinalPatterns{M,F} <: PermutationOutcomeSpace{M}
    encoding::OrdinalPatternEncoding{M,F}
    τ::Int
    A::Float64
end

is_counting_based(o::AmplitudeAwareOrdinalPatterns) = false

# Initializations
function OrdinalPatterns(; τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F}
    m >= 2 || throw(ArgumentError("Need order m ≥ 2."))
    return OrdinalPatterns{m, F}(OrdinalPatternEncoding{m}(lt), τ)
end
function WeightedOrdinalPatterns(; τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F}
    m >= 2 || throw(ArgumentError("Need order m ≥ 2."))
    return WeightedOrdinalPatterns{m, F}(OrdinalPatternEncoding{m}(lt), τ)
end
function AmplitudeAwareOrdinalPatterns(; A = 0.5, τ::Int = 1, m::Int = 3, lt::F=isless_rand) where {F}
    m >= 2 || throw(ArgumentError("Need order m ≥ 2."))
    return AmplitudeAwareOrdinalPatterns{m, F}(OrdinalPatternEncoding{m}(lt), τ, A)
end

###########################################################################################
# Implementation of the whole `probabilities` API on abstract `PermProbEst`
###########################################################################################
# Probabilities etc. simply initialize the datasets and containers of the encodings
# and just map everythihng using `encode`. The only difference between the three
# types is whether they compute some additional weights that are affecting
# how the probabilities are counted.

function probabilities(est::PermProbEst{m}, x::AbstractVector{T}) where {m, T<:Real}
    dataset::StateSpaceSet{m,T} = embed(x, m, est.τ)
    return probabilities(est, dataset)
end

function probabilities(est::PermProbEst{m}, x::AbstractStateSpaceSet{D}) where {m, D}
    m != D && throw(ArgumentError(
        "Order of ordinal patterns and dimension of `StateSpaceSet` must match!"
    ))
    πs = zeros(Int, length(x))
    return probabilities!(πs, est, x)
end

function probabilities!(::Vector{Int}, ::PermProbEst, ::AbstractVector)
    error("""
    In-place `probabilities!` for `OrdinalPatterns` can only be used by
    StateSpaceSet input, not timeseries. First embed the timeseries or use the
    normal version `probabilities`.
    """)
end

function fasthist!(πs::Vector{Int}, est::PermProbEst{m}, x::AbstractStateSpaceSet{m}) where {m}
    # TODO: The following loop can probably be parallelized!
    @inbounds for (i, χ) in enumerate(x)
        πs[i] = encode(est.encoding, χ)
    end
    weights = permutation_weights(est, x)
    # careful: the following histogram is normalized if weights are provided,
    # so it doesn't actually return integer counts!
    cts = fasthist!(πs, weights)
    return cts
end

function symbolize(est::PermProbEst{m}, x) where m
    if x isa AbstractVector
        dataset = embed(x, m, est.τ)
    else
        dataset = x
    end
    m != dimension(dataset) && throw(ArgumentError(
        "Order of ordinal patterns and dimension of `StateSpaceSet` must match!"
    ))
    πs = zeros(Int, length(dataset))
    @inbounds for (i, χ) in enumerate(dataset)
        πs[i] = encode(est.encoding, χ)
    end
    return πs
end

function probabilities!(πs::Vector{Int}, est::PermProbEst{m}, x::AbstractStateSpaceSet{m}) where {m}
    return Probabilities(fasthist!(πs, est, x))
end

# A generic "weighted counts" + outcomes function. We need this because
# `OrdinalPatterns` is counting-compatible (i.e. returns a true histogram `Vector{Int}`),
# while `WeightedOrdinalPatterns` and `AmplitudeAwareOrdinalPatterns` returns
# a normalized histogram, and are thus *not* counting compatible.
function weighted_counts_and_outcomes(est::PermProbEst{m}, x::Vector_or_SSSet) where {m}
    # A bit of code duplication here, because we actually need the processed
    # `πs` to invert it with `decode`. This can surely be optimized with some additional
    # function that both maps to integers with `decode` but also keeps track of
    # the permutation pattern vectors. Anyways, I don't think `outcomes` is a function
    # that will be called often, so we can live with this as is.
    if x isa AbstractVector
        dataset = embed(x, m, est.τ)
    else
        dataset = x
    end
    m != dimension(dataset) && throw(ArgumentError(
        "Order of ordinal patterns and dimension of `StateSpaceSet` must match!"
    ))
    πs = zeros(Int, length(dataset))
    freqs = fasthist!(πs, est, dataset)
    # Okay, now we compute the outcomes. (`πs` is already sorted in `fasthist!`)
    outcomes = decode.(Ref(est.encoding), unique!(πs))
    return freqs, outcomes
end

function probabilities_and_outcomes(est::PermProbEst{m}, x::Vector_or_SSSet) where {m}
    freqs, outcomes = weighted_counts_and_outcomes(est, x)
    return Probabilities(freqs), outcomes
end

# fallback
total_outcomes(est::PermutationOutcomeSpace) = total_outcomes(est.encoding)
outcome_space(est::PermutationOutcomeSpace) = outcome_space(est.encoding)

# If `x` is state space set, delay embedding is ignored and all elements
# are mapped to outcomes. Otherwise, delay embedding is done.
function encoded_space_cardinality(o::PermProbEst{m}, x::AbstractArray) where {m}
    N = length(x)
    return N - (m - 1)*o.τ
end

###########################################################################################
# Permutation weights definition
###########################################################################################
permutation_weights(::OrdinalPatterns, ::Any) = nothing

function permutation_weights(::WeightedOrdinalPatterns{m}, x::AbstractStateSpaceSet) where {m}
    weights_from_variance.(vec(x), m)
end

function weights_from_variance(χ, m::Int)
    z = mean(χ)
    s = sum(e -> (e - z)^2, χ)
    return s/m
end

function permutation_weights(est::AmplitudeAwareOrdinalPatterns{m}, x::AbstractStateSpaceSet) where {m}
    AAPE.(vec(x), est.A, m)
end

# TODO: This has absolutely terrible performance, allocating liek 10 vectors for each
# element of a dataset...
"""
    AAPE(x, A::Real = 0.5, m::Int = length(x))

Encode relative amplitude information of the elements of `a`.
- `A = 1` emphasizes only average values.
- `A = 0` emphasizes changes in amplitude values.
- `A = 0.5` equally emphasizes average values and changes in the amplitude values.
"""
function AAPE(x, A::Real = 0.5, m::Int = length(x))
    (A/m)*sum(abs.(x)) + (1-A)/(m-1)*sum(abs.(diff(x)))
end

###########################################################################################
# `OrdinalPatterns` specific
###########################################################################################
# We can actually get raw non-normalized counts for `OrdinalPatterns`. This is not
# possible for the weighted alternatives.
function counts_and_outcomes(est::OrdinalPatterns{m}, x::Vector_or_SSSet) where {m}
    return weighted_counts_and_outcomes(est, x)
end
