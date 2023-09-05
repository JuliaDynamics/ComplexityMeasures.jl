using DimensionalData: DimArray
using DimensionalData.Dimensions: Dim

export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export probabilities_and_outcomes
export allprobabilities
export allprobabilities_and_outcomes
export missing_outcomes

###########################################################################################
# Types
###########################################################################################
"""
    Probabilities <: DimArray
    Probabilities(x) → p

`Probabilities` is a simple wrapper around `x::DimArray{<:Real, N}` that ensures its
values sum to 1, so that `p` can be interpreted as `N`-dimensional probability mass
function. In most use cases, `p` will be a vector. `p` behaves exactly like
its contained data `x` with respect to indexing and iteration.
"""
struct Probabilities{T, N, A <: AbstractDimArray} <: AbstractArray{T, N}
    p::A
    function Probabilities(x::AbstractArray{T, N},
            dimlabels::Union{NamedTuple, Tuple, Nothing} = nothing,
            normed::Bool = false) where {T <: Real, N}
        if !(typeof(x) <: AbstractDimArray)
            if dimlabels isa NamedTuple || dimlabels isa Dim
                # Use provided dimension labels and outcomes directly.
                p = DimArray(x, dimlabels)
            else
                # Enumerate outcomes with positive integers and name dimensions generically
                # x1, x2, ...
                if dimlabels isa Nothing
                    s = size(x)
                    dims = (Pair(Symbol("x$i"), 1:s[i]) for i = 1:N)
                    p = DimArray(x, NamedTuple(dims))
                # `dimlabels` isa Tuple; use the provided outcomes, but assign them
                # to generically named dimensions x1, x2, ...
                else
                    dims = (Pair(Symbol("x$i"), dimlabels[i]) for i = 1:N)
                    p = DimArray(x, NamedTuple(dims))
                end
            end
        else
            p = x
        end
        if !normed # `normed` is an internal argument that skips checking the sum.
            s = sum(p.data, dims = 1:N)
            if s ≠ 1
                p = p ./ s
            end
        end

        return new{T, N, typeof(p)}(p)
    end

    function Probabilities(x::AbstractDimArray{T, N}) where {T, N}
        return new{T, N, typeof(x)}(x)
    end
end
# Backwards compatibility
Probabilities(x, normed::Bool) = Probabilities(x, nothing, normed)

function Probabilities(x::AbstractArray{<:Integer, N}) where N
    s = sum(x)
    return Probabilities(x ./ s, true)
end
Probabilities(x::Counts) = Probabilities(x.cts, x.cts.dims)

# default show used by display() on the REPL
function Base.show(io::IO, mime::MIME"text/plain", p::Probabilities)
    print_dims(io, mime, p.p, typeof(p))
end

# extend DimensionalData interface:
for f in (:dims, :refdims, :data, :name, :metadata, :layerdims)
    @eval $(f)(c::Probabilities) = $(f)(c.p)
end

# extend base Array interface:
for f in (:length, :size, :eachindex, :eltype, :parent,
    :lastindex, :firstindex, :vec, :getindex, :iterate)
    @eval Base.$(f)(d::Probabilities{T, N}, args...) where {T, N} = $(f)(d.p, args...)
end

# Other useful methods:
Base.sort(p::Probabilities) = sort(p.p)

Base.IteratorSize(::Probabilities) = Base.HasLength()
# Special extension due to the rules of the API
@inline Base.sum(::Probabilities{T}) where T = one(T)
# -----------------------------------------------------------------
# Outcomes are simply the labels on the marginal dimensional.
# For 1D, we return the outcomes as-is. For ND, we return
# a tuple of the outcomes --- one element per dimension.
# -----------------------------------------------------------------
outcomes(c::Probabilities{T, 1}) where T= first(c.p.dims).val.data

# Integer indexing returns the outcomes for that dimension directly.
function outcomes(c::Probabilities{T, N}, i::Int) where {T, N}
    return c.p.dims[i].val.data
end

function outcomes(c::Probabilities{T, N}) where {T, N}
    return map(i -> c.p.dims[i].val.data, tuple(1:N...))
end

function outcomes(c::Probabilities{T, N}, idxs) where {T, N}
    map(i -> c.p.dims[i].val.data, tuple(idxs...))
end

"""
    ProbabilitiesEstimator

The supertype for all probabilities estimators.

The role of the probabilities estimator is to convert (pseudo-)counts to probabilities.
Currently, the implementation of all probabilities estimators assume *finite* outcome
space with known cardinality (i.e. the user must *model*/*assume* what this outcome space
is). Therefore, `ProbabilitiesEstimator` accept an [`OutcomeSpace`](@ref) as the first
argument, which specifies the set of possible outcomes.

## Implementations

The default probabilities estimator is [`RelativeAmount`](@ref), which is compatible with any
[`OutcomeSpace`](@ref). The following estimators only support counting-based outcomes.

- [`Shrinkage`](@ref).
- [`BayesianRegularization`](@ref).
- [`AddConstant`](@ref).

## Description

In ComplexityMeasures.jl, probability mass functions
are estimated from data by defining a set of
possible outcomes ``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}`` (by
specifying an [`OutcomeSpace`](@ref)), and
assigning to each outcome ``\\omega_i`` a probability ``p(\\omega_i)``, such that
``\\sum_{i=1}^N p(\\omega_i) = 1`` (by specifying a `ProbabilitiesEstimator`).

## Used with

- [`probabilities`](@ref)/[`probabilities_and_outcomes`](@ref) for estimating a probability
    distribution over some [`OutcomeSpace`](@ref) from input data.
- [`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref) for estimating a
    probability distribution from input data, guaranteeing inclusion of zero-probability
    outcomes.

The returned probabilities `p` are a [`Probabilities`](@ref) (`Vector`-like), where each
element `p[i]` is the probability of the outcome `ω[i]`. Using an [`OutcomeSpace`](@ref)
directly as input to [`probabilities`](@ref) or related functions is also possible, and is
equivalent to using the [`RelativeAmount`](@ref) estimator.
"""
abstract type ProbabilitiesEstimator end

###########################################################################################
# probabilities and combo function
###########################################################################################
"""

    probabilities([est::ProbabilitiesEstimator], cts::Counts) → p::Probabilities

Estimate probabilities from `cts` using the given [`ProbabilitiesEstimator`](@ref) `est`
(if no estimator is provided, [`RelativeAmount`](@ref) is used).

    probabilities([est::ProbabilitiesEstimator], o::OutcomeSpace, x::Array_or_SSSet) → p::Probabilities

Estimate a probability distribution over the set of possible outcomes `Ω`
defined by the [`OutcomeSpace`](@ref) `o`, given input data `x`.

The input data is typically an `Array` or a `StateSpaceSet` (or `SSSet` for short); see
[Input data for ComplexityMeasures.jl](@ref). Configuration options are always given as
arguments to the chosen outcome space.

Returns a [`Probabilities`](@ref) instance where the outcomes are conveniently
displayed in the margin of the probabilities array. Use [`outcomes`](@ref) on the
probabilities to get the outcomes explicitly. Alternatively, use
[`probabilities_and_outcomes`](@ref) to get both in one operation.

## Description

Probabilities are computed by:

1. Discretizing/encoding `x` into a finite set of outcomes `Ω` specified by the provided
    [`OutcomeSpace`](@ref) `o`.
2. Assigning to each outcome `Ωᵢ ∈ Ω` either a count (how often it appears among the
    discretized data points), or a pseudo-count (some pre-normalized probability such
    that `sum(Ωᵢ for Ωᵢ in Ω) == 1`).

For outcome spaces that result in pseudo counts, these pseudo counts are simply treated
as probabilities and returned directly (that is, `est` is ignored). For counting-based
outcome spaces (see [`OutcomeSpace`](@ref) docstring), probabilities
are estimated from the counts using some [`ProbabilitiesEstimator`](@ref) (first signature).

## Observed vs all probabilities

Due to performance optimizations, whether the returned probabilities
contain `0`s as entries or not depends on the outcome space.
E.g., in [`ValueHistogram`](@ref) `0`s are skipped, while in
[`PowerSpectrum`](@ref) `0` are not skipped, because we get them for free.

Use [`allprobabilities`](@ref)/[`allprobabilities_and_outcomes`](@ref) to guarantee that
zero probabilities are also returned (may be slower).

## Examples

```julia
x = randn(500)
ps = probabilities(OrdinalPatterns(m = 3), x)
ps = probabilities(ValueHistogram(RectangularBinning(5)), x)
ps = probabilities(WaveletOverlap(), x)
```

The outcome space is here given as the first argument to `est`.

## Examples

```julia
x = randn(500)

# Syntactically equivalent to `probabilities(OrdinalPatterns(m = 3), x)`
ps = probabilities(RelativeAmount(OrdinalPatterns(m = 3)), x)

# Some more sophisticated ways of estimating probabilities:
ps = probabilities(BayesianRegularization(OrdinalPatterns(m = 3)), x)
ps = probabilities(Shrinkage(ValueHistogram(RectangularBinning(5))), x)

# Only the `RelativeAmount` estimator works with non-counting based outcome spaces,
# like for example `WaveletOverlap`.
ps = probabilities(RelativeAmount(WaveletOverlap()), x) # works
ps = probabilities(BayesianRegularization(WaveletOverlap()), x) # errors
```

    probabilities(x::Vector_or_SSSet) → p::Probabilities

Estimate probabilities by using directly counting the elements of `x`, assuming that
`Ω = sort(unique(x))`, i.e. that the outcome space is the unique elements of `x`.
This is mostly useful when `x` contains categorical data. It is syntactically equivalent
to `probabilities(RelativeAmount(), CountOccurrences(), x)`.

See also: [`counts`](@ref), [`probabilities_and_outcomes`](@ref),
[`allprobabilities`](@ref), [`allprobabilities_and_outcomes`](@ref),
[`Probabilities`](@ref), [`ProbabilitiesEstimator`](@ref).
"""
function probabilities(o::OutcomeSpace, x)
    probabilities(RelativeAmount(), o, x)
end
# The above method is overriden for non-count based outcome spaces. For count-based
# outcome space, `counts(o::OutcomeSpace, x)` must be defined.

function probabilities(x)
    return probabilities(RelativeAmount(), CountOccurrences(), x)
end


"""
    probabilities_and_outcomes([est::ProbabilitiesEstimator],
        o::OutcomeSpace, x) → (probs::Probabilities, Ω)

Like [`probabilities`](@ref), but also return the outcomes `Ω` explicitly. `Ω[i]` is the
outcome corresponding to the probabilities `probs[i]`.

The element type of `Ω` depends on the estimator. `Ω` is a subset of the
[`outcome_space`](@ref) of `o`.

See also [`outcomes`](@ref), [`total_outcomes`](@ref).
"""
function probabilities_and_outcomes(est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    probs::Probabilities = probabilities(est, o, x)
    return probs, outcomes(probs)
end

function probabilities_and_outcomes(est::ProbabilitiesEstimator, args...)
    s = "`probabilities_and_outcomes` not implemented for estimator $(typeof(est))."
    throw(ArgumentError(s))
end

function outcomes(est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    return outcomes(o, x)
end
function outcome_space(est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    return outcome_space(o, x)
end

function total_outcomes(est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    return total_outcomes(o, x)
end

"""
    probabilities!(s, args...)

Similar to `probabilities(args...)`, but allows pre-allocation of temporarily used
containers `s`.

Only works for certain estimators. See for example [`OrdinalPatterns`](@ref).
"""
function probabilities! end

###########################################################################################
# All probabilities
###########################################################################################
# This method is overriden by non-counting-based `OutcomeSpace`s. For counting-based
# `OutcomeSpace`s, we just utilize `counts_and_outcomes` to get the histogram, then
# normalize it when converting to `Probabilities`.
function probabilities_and_outcomes(o::OutcomeSpace, x)
    return probabilities_and_outcomes(RelativeAmount(), o, x)
end

"""
    allprobabilities(est::ProbabilitiesEstimator, x::Array_or_SSSet) → p
    allprobabilities(o::OutcomeSpace, x::Array_or_SSSet) → p

The same as [`probabilities`](@ref), but ensures that outcomes with `0` probability
are explicitly added in the returned vector. This means that `p[i]` is the probability
of `ospace[i]`, with `ospace = `[`outcome_space`](@ref)`(est, x)`.

This function is useful in cases where one wants to compare the probability mass functions
of two different input data `x, y` under the same estimator. E.g., to compute the
KL-divergence of the two PMFs assumes that the obey the same indexing. This is
not true for [`probabilities`](@ref) even with the same `est`, due to the skipping
of 0 entries, but it is true for [`allprobabilities`](@ref).
"""
function allprobabilities(est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    probs = probabilities(est, o, x)
    outs = outcomes(probs) # the observed outcomes
    ospace = vec(outcome_space(o, x)) # all possible outcomes
    # We first utilize that the outcome space is sorted and sort probabilities
    # accordingly (just in case we have an estimator that is not sorted)
    s = sortperm(outs)
    sort!(outs)
    ps = probs[s]
    # we now iterate over possible outocomes;
    # if they exist in the observed outcomes, we push their corresponding probability
    # into the probabilities vector. If not, we push 0 into the probabilities vector!
    allprobs = eltype(ps)[]
    observed_index = 1 # index of observed outcomes
    for j in eachindex(ospace) # we made outcome space a vector on purpose
        ω = ospace[j]
        ωobs = outs[observed_index]
        if ω ≠ ωobs
            push!(allprobs, 0)
        else
            push!(allprobs, ps[observed_index])
            observed_index += 1
        end
        # Check whether we have exhausted observed outcomes
        if observed_index > length(outs)
            remaining_0s = length(ospace) - j
            append!(allprobs, zeros(remaining_0s))
            break
        end
    end
    return Probabilities(allprobs, (x1 = ospace,))
end
function allprobabilities(o::OutcomeSpace, x)
    return allprobabilities(RelativeAmount(), o, x)
end

# If an outcome space model is provided without specifying a probabilities estimator,
# then naive plug-in estimation is used (the `RelativeAmount` estimator). In the case of
# counting-based `OutcomeSpace`s, we explicitly count occurrences of each
# outcome in the encoded data. For non-counting-based `OutcomeSpace`s, we
# just fill in the non-considered outcomes with zero probabilities.

# Each `ProbabilitiesEstimator` subtype must extend this method explicitly.

"""
    allprobabilities_and_outcomes(o::OutcomeSpace, x::Array_or_SSSet) → (p, Ω)
    allprobabilities_and_outcomes(est::ProbabilitiesEstimator, x::Array_or_SSSet) → (p, Ω)

The same as [`allprobabilities`](@ref), but also returns the outcome space `Ω`.
"""
function allprobabilities_and_outcomes(est::ProbabilitiesEstimator, o::OutcomeSpace,
        x::Array_or_SSSet)
    probs::Probabilities = allprobabilities(est, o, x)

    return probs, outcomes(probs)
end
function allprobabilities_and_outcomes(o::OutcomeSpace, x)
    return allprobabilities_and_outcomes(RelativeAmount(), o, x)
end


"""
    missing_outcomes(o::OutcomeSpace, x; all = true) → n_missing::Int

Count the number of missing (i.e., zero-probability) outcomes
specified by `o`, given input data `x`, using [`RelativeAmount`](@ref)
probabilities estimation.

If `all == true`, then [`allprobabilities`](@ref) is used to compute the probabilities.
If `all == false`, then [`probabilities`](@ ref) is used to compute the probabilities.

This is syntactically equivalent to `missing_outcomes(RelativeAmount(o), x)`.

    missing_outcomes(est::ProbabilitiesEstimator, o::OutcomeSpace, x) → n_missing::Int

Like above, but specifying a custom [`ProbabilitiesEstimator`](@ref) too.

See also: [`MissingDispersionPatterns`](@ref).
"""
function missing_outcomes(o::OutcomeSpace, x; all::Bool = true)
    if all
        probs = allprobabilities(o, x)
        L = length(probs)
    else
        probs = probabilities(o, x)
        L = total_outcomes(o, x)
    end
    O = count(!iszero, probs)
    return L - O
end

function missing_outcomes(est::ProbabilitiesEstimator, o::OutcomeSpace, x; all::Bool = true)
    if all
        probs = allprobabilities(est, o, x)
        L = length(probs)
    else
        probs = probabilities(est, o, x)
        L = total_outcomes(o, x)
    end
    O = count(!iszero, probs)
    return L - O
end
