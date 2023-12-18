
export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export allprobabilities
export missing_outcomes

###########################################################################################
# Types
###########################################################################################
"""
    Probabilities <: Array{N, <: AbstractFloat}
    Counts(counts [, outcomes [, outnames]]) → p

`Probabilities` stores an `N`-dimensional array of probabilities, while ensuring that
the array sums to 1 (normalized probability mass). The probabilities correspond to outcomes
that describe the axes of the array. In most cases the array is a standard vector.

If `p isa Counts`, then `p.outcomes[i]` is the outcomes along the `i`-th dimension,
each being an abstract vector whose order is the same one corresponding to `p`,
and `p.dimlabels[i]` is the label of the `i`-th dimension.
Both labels and outcomes are assigned automatically if not given.
`p` itself can be manipulated and iterated over like its stored array.
"""
struct Probabilities{T, N, S} <: AbstractArray{T, N}
    # The frequency table.
    p::AbstractArray{T, N}

    # Outcomes[i] has the same number of elements as `cts` along dimension `i`.
    outcomes::Tuple{Vararg{<:AbstractVector, N}}

    # A label for each dimension
    dimlabels::NTuple{N, S}

    function Probabilities(x::AbstractArray{T, N},
            outcomes::Tuple{Vararg{V, N} where V},
            dimlabels::NTuple{N, S};
            normed::Bool = false) where {T, N, S}
        if !normed # `normed` is an internal argument that skips checking the sum.
            s = sum(x, dims = 1:N)
            if s ≠ 1
                p = x ./ s
            end
        else
            if T <: Integer
                p = float.(x)
            else
                p = x
            end
        end

        s = size(p)
        for dim = 1:N
            L = length(outcomes[dim])
            if L != s[dim]
                msg = "The number of outcomes for dimension $dim must match the number " *
                    "of probabilities for dimension $dim. Got $L outcomes but $(s[dim]) probabilities."
                throw(ArgumentError(msg))
            end
        end

        return new{eltype(p), N, S}(p, outcomes, dimlabels)
    end

end

# If no names are give for the dimension assign generic ones
function Probabilities(x::AbstractArray{T, N}, outcomes::Tuple; normed::Bool = false) where {T, N}
    return Probabilities(x, outcomes, tuple((Symbol("x$i") for i = 1:N)...); normed)
end
# If no outcomes are given, assign generic `EnumeratedOutcome`s.
function Probabilities(x::AbstractArray{T, N}; normed::Bool = false) where {T, N}
    return Probabilities(x, generate_outcomes(x); normed)
end

# Backwards compatibility
Probabilities(x, normed::Bool) = Probabilities(x; normed)

function Probabilities(x::AbstractArray{<:Integer, N}) where N
    s = sum(x)
    return Probabilities(x ./ s; normed = true)
end
Probabilities(x::Counts) = Probabilities(x.cts, x.outcomes, x.dimlabels)


# Convenience wrappers for 1D case
function Probabilities(x::AbstractVector, outcomes::AbstractVector; normed::Bool = false)
    return Probabilities(x, (outcomes, ); normed)
end
function Probabilities(x::AbstractVector, outcomes::AbstractVector, label; normed::Bool = false)
    return Probabilities(x, (outcomes, ), (label, ); normed)
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
function outcomes(p::Probabilities{T, 1}) where T
    return first(p.outcomes)
end

# Integer indexing returns the outcomes for that dimension directly.
function outcomes(p::Probabilities{T, N}, i::Int) where {T, N}
    return p.outcomes[i]
end

function outcomes(p::Probabilities{T, N}) where {T, N}
    return map(i -> p.outcomes[i], tuple(1:N...))
end

function outcomes(p::Probabilities{T, N}, idxs) where {T, N}
    map(i -> p.outcomes[i], tuple(idxs...))
end

"""
    ProbabilitiesEstimator

The supertype for all probabilities estimators.

The role of the probabilities estimator is to convert (pseudo-)counts to probabilities.
Currently, the implementation of all probabilities estimators assume *finite* outcome
space with known cardinality. Therefore, `ProbabilitiesEstimator` accept an
[`OutcomeSpace`](@ref) as the first
argument, which specifies the set of possible outcomes.

Probabilities estimators are used with [`probabilities`](@ref) and [`allprobabilities`](@ref).

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
"""
abstract type ProbabilitiesEstimator end

###########################################################################################
# probabilities and combo function
###########################################################################################
"""

    probabilities([est::ProbabilitiesEstimator], counts::Counts) → p::Probabilities

Estimate probabilities from the pre-computed `counts` using the given
[`ProbabilitiesEstimator`](@ref) `est`.

If no estimator is provided, then [`RelativeAmount`](@ref) is used.

    probabilities([est::ProbabilitiesEstimator], o::OutcomeSpace, x::Array_or_SSSet) → p::Probabilities

Estimate a probability distribution over the set of possible outcomes `Ω`
defined by the [`OutcomeSpace`](@ref) `o`, given input data `x`.

The input data is typically an `Array` or a `StateSpaceSet` (or `SSSet` for short); see
[Input data for ComplexityMeasures.jl](@ref input_data). Configuration options are always given as
arguments to the chosen outcome space.

The returned probabilities `p` are a [`Probabilities`](@ref) (`Vector`-like), where each
element `p[i]` is the probability of the outcome `ω[i]`. The outcomes are displayed
to the left of the probabilities as marginals when `p` is displayed.
Use [`outcomes`](@ref) on `p` to obtain the outcomes explicitly.

If `est` is not given, it defaults to the [`RelativeAmount`](@ref) estimator.

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
E.g., in [`ValueBinning`](@ref) `0`s are skipped, while in
[`PowerSpectrum`](@ref) `0` are not skipped, because we get them for free.

Use [`allprobabilities`](@ref) to guarantee that
zero probabilities are also returned (may be slower).

## Examples

```julia
x = randn(500)
ps = probabilities(OrdinalPatterns(m = 3), x)
ps = probabilities(ValueBinning(RectangularBinning(5)), x)
ps = probabilities(WaveletOverlap(), x)
```

The outcome space is here given as the first argument to `est`.

## Examples

```julia
x = randn(500)

# Syntactically equivalent to `probabilities(OrdinalPatterns(m = 3), x)`
ps = probabilities(RelativeAmount, OrdinalPatterns(m = 3), x)

# Some more sophisticated ways of estimating probabilities:
ps = probabilities(BayesianRegularization(), OrdinalPatterns{3}(), x)
ps = probabilities(Shrinkage(), ValueBinning(RectangularBinning(5)), x)

# Only the `RelativeAmount` estimator works with non-counting based outcome spaces,
# like for example `WaveletOverlap`.
ps = probabilities(RelativeAmount(), WaveletOverlap(), x) # works
ps = probabilities(BayesianRegularization(), WaveletOverlap(), x) # errors
```
"""
function probabilities end

# Functions related to outcomes are propagated
for f in (:outcomes, :outcome_space, :total_outcomes)
    @eval function $(f)(::ProbabilitiesEstimator, o::OutcomeSpace, x)
        return $(f)(o, x)
    end
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
# This method is overriden by non-counting-based `OutcomeSpace`s and for outcome spaces
# where explicitly creating the outcomes is expensive.
function probabilities(o::OutcomeSpace, x)
    return first(probabilities_and_outcomes(o, x))
end

function probabilities_and_outcomes(o::CountBasedOutcomeSpace, x)
    cts, outs = counts_and_outcomes(o, x)
    probs = Probabilities(cts, outs)
    return probs, outcomes(probs)
end

# If an outcome space is provided without specifying a probabilities estimator,
# then naive plug-in estimation is used (the `RelativeAmount` estimator). In the case of
# counting-based `OutcomeSpace`s, we explicitly count occurrences of each
# outcome in the encoded data. For non-counting-based `OutcomeSpace`s, we
# just fill in the non-considered outcomes with zero probabilities.
# Each `ProbabilitiesEstimator` subtype must extend this method explicitly.

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
    # the observed outcomes and their probabilities
    probs, os = probabilities_and_outcomes(est, o, x)
    # todo os is a tuple.
    if os isa AbstractRange
        outs = collect(os)
    else
        outs = os
    end
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
    return Probabilities(allprobs, (ospace, ))
end
function allprobabilities(o::OutcomeSpace, x)
    return allprobabilities(RelativeAmount(), o, x)
end


"""
    missing_outcomes(o::OutcomeSpace, x; all = true) → n_missing::Int

Count the number of missing (i.e., zero-probability) outcomes
specified by `o`, given input data `x`, using [`RelativeAmount`](@ref)
probabilities estimation.

If `all == true`, then [`allprobabilities`](@ref) is used to compute the probabilities.
If `all == false`, then [`probabilities`](@ref) is used to compute the probabilities.

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
