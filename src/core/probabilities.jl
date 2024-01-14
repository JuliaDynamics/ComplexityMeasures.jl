
export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export probabilities_and_outcomes
export allprobabilities_and_outcomes
export missing_outcomes

###########################################################################################
# Types
###########################################################################################
"""
    Probabilities <: Array{<:AbstractFloat, N}
    Probabilities(probs::Array, [, outcomes [, dimlabels]]) → p
    Probabilities(counts::Counts, [, outcomes [, dimlabels]]) → p

`Probabilities` stores an `N`-dimensional array of probabilities, while ensuring that
the array sums to 1 (normalized probability mass). In most cases the array is a standard
vector. `p` itself can be manipulated and iterated over, just like its stored array.

## Outcomes and labels

The probabilities correspond to outcomes that describe the axes of the array. 
If `p isa Probabilities`, then `p.outcomes[i]` is an an abstract vector containing 
the outcomes along the `i`-th dimension. The outcomes have the same ordering as the 
probabilities, so that `p[i][j]` is the probability for outcome `p.outcomes[i][j]`. 
The dimensions of the array are named, and can be accessed by `p.dimlabels`, where 
`p.dimlabels[i]` is the label of the `i`-th dimension. Both `outcomes` and `dimlabels`
are assigned automatically if not given. If the input is a
set of [`Counts`](@ref), and `outcomes` and `dimlabels` are not given, 
then the labels and outcomes are inherited from the counts.

## Examples 

```julia
julia> probs = [0.2, 0.2, 0.2, 0.2]; 

julia> Probabilities(probs) # will be normalized to sum to 1
 Probabilities{Float64,1} over 4 outcomes
 Outcome(1)  0.25
 Outcome(2)  0.25
 Outcome(3)  0.25
 Outcome(4)  0.25
```

```julia
julia> c = Counts([12, 16, 12], ["out1", "out2", "out3"]);

julia> Probabilities(c)
 Probabilities{Float64,1} over 3 outcomes
 "out1"  0.3
 "out2"  0.4
 "out3"  0.3
```
"""
struct Probabilities{T, N, S} <: AbstractArray{T, N}
    # The probabilities table.
    p::AbstractArray{T, N}

    # outcomes[i] has the same number of elements as `cts` along dimension `i`.
    outcomes::Tuple{Vararg{AbstractVector, N}}

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
for f in (:length, :size, :eltype, :parent,
    :lastindex, :firstindex, :vec, :getindex, :iterate)
    @eval Base.$(f)(d::Probabilities, args...)= $(f)(d.p, args...)
end

# One-argument definitions to avoid type ambiguities with Base:
Base.eachindex(p::Probabilities) = eachindex(p.p)
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
outcomes(p::Probabilities) = p.outcomes
outcomes(p::Probabilities{T, 1}) where T = first(p.outcomes)
# Integer indexing returns the outcomes for that dimension directly.
function outcomes(p::Probabilities{T, N}, i::Int) where {T, N}
    return p.outcomes[i]
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

Probabilities estimators are used with [`probabilities`](@ref) and
[`allprobabilities_and_outcomes`](@ref).

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
    probabilities_and_outcomes(
        [est::ProbabilitiesEstimator], o::OutcomeSpace, x::Array_or_SSSet
    ) → (p::Probabilities, Ω)

Estimate a probability distribution over the set of possible outcomes `Ω`
defined by the [`OutcomeSpace`](@ref) `o`, given input data `x`.
Probabilities are estimated according to the given probabilities estimator `est`,
which defaults to [`RelativeAmount`](@ref).

The input data is typically an `Array` or a `StateSpaceSet` (or `SSSet` for short); see
[Input data for ComplexityMeasures.jl](@ref input_data). Configuration options are always
given as arguments to the chosen outcome space and probabilities estimator.

Return a tuple where the first element is a [`Probabilities`](@ref) instance, which is
vector-like and contains the probabilities, and where the second element `Ω` are the outcomes
corresponding to the probabilities, such that `p[i]` is the probability for the outcome `Ω[i]`.

The outcomes are actually included in `p`, and you can use the [`outcomes`](@ref)
function on the `p` to get them. `probabilities_and_outcomes` returns both for
backwards compatibility.


    probabilities_and_outcomes(
        [est::ProbabilitiesEstimator], counts::Counts
    ) → (p::Probabilities, Ω)

Estimate probabilities from the pre-computed `counts` using the given
[`ProbabilitiesEstimator`](@ref) `est`.

## Description

Probabilities are computed by:

1. Discretizing/encoding `x` into a finite set of outcomes `Ω` specified by the provided
    [`OutcomeSpace`](@ref) `o`.
2. Assigning to each outcome `Ωᵢ ∈ Ω` either a count (how often it appears among the
    discretized data points), or a pseudo-count (some pre-normalized probability such
    that `sum(Ωᵢ for Ωᵢ in Ω) == 1`).

For outcome spaces that result in pseudo counts, such as [`PowerSpectrum`](@ref),
these pseudo counts are simply treated
as probabilities and returned directly (that is, `est` is ignored). For counting-based
outcome spaces (see [`OutcomeSpace`](@ref) docstring), probabilities
are estimated from the counts using some [`ProbabilitiesEstimator`](@ref) (first signature).

## Observed vs all probabilities

Due to performance optimizations, whether the returned probabilities
contain `0`s as entries or not depends on the outcome space.
E.g., in [`ValueBinning`](@ref) `0`s are skipped, while in
[`PowerSpectrum`](@ref) `0` are not skipped, because we get them for free.

Use [`allprobabilities_and_outcomes`](@ref) to guarantee that
zero probabilities are also returned (may be slower).
"""
function probabilities_and_outcomes end

"""
    probabilities(
        [est::ProbabilitiesEstimator], o::OutcomeSpace, x::Array_or_SSSet
    ) → (p::Probabilities, Ω)

Like [`probabilities_and_outcomes`](@ref), but returns the [`Probabilities`](@ref) `p`
directly.

Compute the same probabilities as in the [`probabilities_and_outcomes`](@ref) function,
with two differences:

1. Do not explicitly return the outcomes.
2. If the outcomes are not estimated for free while estimating the counts,
   a special integer type is used to enumerate the outcomes, to avoid the computational
   cost of estimating the outcomes.


    probabilities([est::ProbabilitiesEstimator], counts::Counts) → (p::Probabilities, Ω)

The same as above, but estimate the probability directly from a set of [`Counts`](@ref).
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

function probabilities(est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    p, outs = probabilities_and_outcomes(est, o, x)
    return p
end

# If an outcome space is provided without specifying a probabilities estimator,
# then naive plug-in estimation is used (the `RelativeAmount` estimator). In the case of
# counting-based `OutcomeSpace`s, we explicitly count occurrences of each
# outcome in the encoded data. For non-counting-based `OutcomeSpace`s, we
# just fill in the non-considered outcomes with zero probabilities.
# Each `ProbabilitiesEstimator` subtype must extend this method explicitly.

"""
    allprobabilities_and_outcomes(est::ProbabilitiesEstimator, x::Array_or_SSSet) → (p::Probabilities, outs)
    allprobabilities_and_outcomes(o::OutcomeSpace, x::Array_or_SSSet) → (p::Probabilities, outs)

The same as [`probabilities_and_outcomes`](@ref), but ensures that outcomes with `0`
probability are explicitly added in the returned vector. This means that `p[i]` is the
probability of `ospace[i]`, with `ospace = `[`outcome_space`](@ref)`(est, x)`.

This function is useful in cases where one wants to compare the probability mass functions
of two different input data `x, y` under the same estimator. E.g., to compute the
KL-divergence of the two PMFs assumes that the obey the same indexing. This is
not true for [`probabilities`](@ref) even with the same `est`, due to the skipping
of 0 entries, but it is true for [`allprobabilities_and_outcomes`](@ref).
"""
function allprobabilities_and_outcomes(est::ProbabilitiesEstimator, o::OutcomeSpace, x)
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
    p = Probabilities(allprobs, (ospace, ))
    return p, outcomes(p)
end
function allprobabilities_and_outcomes(o::OutcomeSpace, x)
    return allprobabilities_and_outcomes(RelativeAmount(), o, x)
end

function allprobabilities(est::ProbabilitiesEstimator, o::OutcomeSpace, x)
    p, outs = allprobabilities_and_outcomes(est, o, x)
    return p
end

"""
    missing_outcomes(o::OutcomeSpace, x; all = true) → n_missing::Int

Count the number of missing (i.e., zero-probability) outcomes
specified by `o`, given input data `x`, using [`RelativeAmount`](@ref)
probabilities estimation.

If `all == true`, then [`allprobabilities_and_outcomes`](@ref) is used to compute the probabilities.
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
