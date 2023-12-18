using StateSpaceSets: AbstractStateSpaceSet
import Base.unique!

export Counts
export counts
export allcounts
export is_counting_based

###########################################################################################
# Counts.
#
# The fundamental quantity used for probabilities estimation are *counts* of how often
# a certain outcome is observed in the input data. These counts are then translated into
# probability mass functions by dedicated `ProbabilitiesEstimator`s.
#
# For example, the most basic probabilities estimator is [`RelativeAmount`](@ref) - the maximum
# likelihood estimator - and it take the relative proportions of counts as the
# probabilities.
#
# If `counts_and_outcomes` and `allcounts_and_outcomes` are implemented for an
# `OutcomeSpace`, then the outcome space is automatically compatible with all
# `ProbabilitiesEstimator`s. For some outcome spaces, however, this is not possible,
# because counting is not defined over their outcome spaces (e.g. [`WaveletOverlap`](@ref)
#  use pre-normalized relative "frequencies", not counts, to estimate probabilities).
###########################################################################################
"""
    Counts <: Array{N, <: Integer}
    Counts(counts [, outcomes [, outnames]]) → c

`Counts` stores an `N`-dimensional array of integer `counts` corresponding to a set of
`outcomes`. This is typically called a "frequency table" or
["contingency table"](https://en.wikipedia.org/wiki/Contingency_table).

If `c isa Counts`, then `c.outcomes[i]` is the outcomes along the `i`-th dimension,
each being an abstract vector whose order is the same one corresponding to `c`,
and `c.dimlabels[i]` is the label of the `i`-th dimension.
Both labels and outcomes are assigned automatically if not given.
`c` itself can be manipulated and iterated over like its stored array.
"""
struct Counts{T <: Integer, N, O <: Tuple, S} <: AbstractArray{T, N}
    # The frequency table.
    cts::AbstractArray{T, N}

    # Outcomes[i] has the same number of elements as `cts` along dimension `i`.
    outcomes::O

    # A label for each dimension
    dimlabels::NTuple{N, S}
end
# If dimlabels are not given, simply label them as symbols (:x1, :x2, ... :xN)
function Counts(x::AbstractArray{T, N}, outcomes) where {T <: Integer, N}
    return Counts(x, outcomes, tuple((Symbol("x$i") for i = 1:N)...))
end
function Counts(x::AbstractVector{Int}, outcomes::AbstractVector, dimlabel::Union{Symbol, AbstractString})
    return Counts(x, (outcomes, ), (dimlabel, ))
end
function Counts(x::AbstractVector{Int}, outcomes::AbstractVector)
    return Counts(x, (outcomes, ), (:x1, ))
end

# If no outcomes are given, assign generic `Outcome`s.
function Counts(x::AbstractArray{Int, N}) where {N}
    return Counts(x, generate_outcomes(x))
end

function generate_outcomes(x::AbstractArray{T, N}) where {T, N}
    # One set of outcomes per dimension
    s = size(x)
    gen = (Outcome(1):1:Outcome(s[i]) for i = 1:N)
    return tuple(gen...)
end

# extend base Array interface:
for f in (:length, :size, :eachindex, :eltype, :parent,
    :lastindex, :firstindex, :vec, :getindex, :iterate)
    @eval Base.$(f)(c::Counts, args...) = $(f)(c.cts, args...)
end
Base.IteratorSize(::Counts) = Base.HasLength()

# We strictly deal with single inputs here. For multi-inputs, see CausalityTools.jl
"""
    counts(o::OutcomeSpace, x) → cts::Counts

Discretize/encode `x` into a finite set of outcomes `Ω` specified by the provided
[`OutcomeSpace`](@ref) `o`, then count how often each outcome `Ωᵢ ∈ Ω` (i.e.
each "discretized value", or "encoded symbol") appears.

Return a [`Counts`](@ref) instance which is a vector-like containing the counts.
When displayed, the marginals of the vector are labelled with the outcomes,
so that it is easy to trace what is being counted. Use [`outcomes`](@ref) on the
resulting [`Counts`](@ref) to get these marginals explicitly.

    counts(x) → cts::Counts

If no [`OutcomeSpace`](@ref) is specified, then [`UniqueElements`](@ref) is used
as the outcome space.

## Description

For [`OutcomeSpace`](@ref)s that uses [`encode`](@ref) to discretize, it is possible to
count how often each outcome ``\\omega_i \\in \\Omega``, where ``\\Omega`` is the
set of possible outcomes, is observed in the discretized/encoded input data.
Thus, we can assign to each outcome ``\\omega_i`` a count ``f(\\omega_i)``, such that
``\\sum_{i=1}^N f(\\omega_i) = N``, where ``N`` is the number of observations in the
(encoded) input data.
`counts` returns the counts ``f(\\omega_i)_{obs}``
and outcomes only for the *observed* outcomes ``\\omega_i^{obs}`` (those outcomes
that actually appear in the input data). If you need the counts for
*unobserved* outcomes as well, use [`allcounts`](@ref)/[`allcounts_and_outcomes`](@ref).
"""
function counts(x)
    xc = copy(x)
    cts = fasthist!(xc) # sorts `xc` in-place
    outs = unique!(xc)
    # Generically call the first dimension `x1` (convention: additional dimensions
    # are named `x2`, `x3`, etc..., but this is defined in CausalityTools.jl)
    return Counts(cts, (outs, ), (:x1, ))
end
# TODO: This is type-piracy that should be moved to StateSpaceSets.jl!
unique!(xc::AbstractStateSpaceSet) = unique!(vec(xc))

# `CountBasedOutcomeSpace`s must implement `counts_and_outcomes`, so we use the 
# following generic fallback.
function counts(o::CountBasedOutcomeSpace, x)
    return first(counts_and_outcomes(o, x))
end

# Generic fallback with informative error
function counts(o::OutcomeSpace, x)
    if !is_counting_based(o)
        throw(ArgumentError(
            "`counts` only works with counting based outcome spaces. "*
            "You provided $(nameof(typeof(o))) which is not one."
        ))
    else
        error("`counts` not yet implemented for outcome space $(nameof(typeof(o)))")
    end
end

# -----------------------------------------------------------------
# Outcomes are simply the labels on the marginal dimensional.
# For 1D, we return the outcomes as-is. For ND, we return
# a tuple of the outcomes --- one element per dimension.
# -----------------------------------------------------------------
outcomes(c::Counts{<:Integer, 1}) = first(c.outcomes)

# Integer indexing returns the outcomes for that dimension directly.
function outcomes(c::Counts{<:Integer, N}, i::Int) where N
    return c.outcomes[i]
end

function outcomes(c::Counts{<:Integer, N}) where N
    return map(i -> c.outcomes[i], tuple(1:N...))
end

function outcomes(c::Counts{<:Integer, N}, idxs) where N
    map(i -> c.outcomes[i], tuple(idxs...))
end

"""
    allcounts(o::OutcomeSpace, x::Array_or_SSSet) → cts::Counts{<:Integer, 1}

Like [`counts`](@ref), but ensures that *all* outcomes `Ωᵢ ∈ Ω`,
where `Ω = outcome_space(o, x)`), are included.
Outcomes that do not occur in the data `x` get 0 count.
"""
function allcounts(o::OutcomeSpace, x::Array_or_SSSet)
    cts, outs = counts_and_outcomes(o, x)
    outs = outcomes(cts)
    ospace = vec(outcome_space(o, x))
    m = length(ospace)
    allcts = zeros(Int, m)
    for (i, ω) in enumerate(ospace)
        idx = findfirst(oⱼ -> oⱼ == ω, outs)
        if !isnothing(idx)
            allcts[i] = cts[idx]
        end
    end
    return Counts(allcts, (ospace,), (:x1, ))
end

"""
    is_counting_based(o::OutcomeSpace)

Return `true` if the [`OutcomeSpace`](@ref) `o` is counting-based, and `false` otherwise.
"""
is_counting_based(o::OutcomeSpace) = o isa CountBasedOutcomeSpace

# `encoded_space_cardinality` is an internal function that makes the
# estimation of count-based probability estimators correct. It returns
# the amount of elements of `x` that are mapped into outcomes. This is
# NOT the same as `length(outcomes(o, x))`, as this counts the unique outcomes.
# For almost all cases, the return value is `length(x)`. It only needs to be
# corrected for few outcome spaces that do e.g., delay embedding first.
# This function does not need to be implemented for non-count based outcome spaces.
encoded_space_cardinality(o, x) = length(x)
