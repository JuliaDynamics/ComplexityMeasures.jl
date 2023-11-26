using DimensionalData
using StateSpaceSets: AbstractStateSpaceSet
import DimensionalData: dims, refdims, data, name, metadata, layerdims
using DimensionalData.Dimensions: Dim
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
    Counts(cts)

`Counts` stores a set of integer counts which is just a simple wrapper around
`DimArray{<:Integer, N}` from
[DimensionalData.jl](https://github.com/rafaqz/DimensionalData.jl), where the names
of the elements being counted are stored in the marginals of the array.

If `N ≥ 1`, then this is typically called a "frequency table" or
["contingency table"](https://en.wikipedia.org/wiki/Contingency_table).

## Implements

- [`counts`](@ref). Estimate a `Counts` instance from data.
- [`outcomes`](@ref). Return the outcomes corresponding to the counts. If `c::Counts` is
    one-dimensional, then the counts are returned directly. If the counts are
    higher-dimensional, then a tuple of the outcomes are returned (one for each dimension).
"""
struct Counts{T <: Integer, N, A <: AbstractDimArray} <: AbstractArray{T, N}
    cts::A
    function Counts(x::AbstractArray{T, N},
        dimlabels::Union{NamedTuple, Tuple, Dim, Nothing} = nothing) where {T <:Integer, N}
        if !(typeof(x) <: AbstractDimArray)
            if dimlabels isa NamedTuple || dimlabels isa Dim
                X = DimArray(x, dimlabels)
            else
                if dimlabels isa Nothing
                    s = size(x)
                    dims = (Pair(Symbol("x$i"), 1:s[i]) for i = 1:N)
                    X = DimArray(x, NamedTuple(dims))
                else # `dimlabels` isa Tuple
                    dims = (Pair(Symbol("x$i"), dimlabels[i]) for i = 1:N)
                    X = DimArray(x, NamedTuple(dims))
                end
            end
        else
            X = x
        end
        return new{T, N, typeof(X)}(X)
    end
    function Counts(x::AbstractDimArray{T, N}) where {T <: Integer, N}
        return new{T, N, typeof(x)}(x)
    end
end

# extend DimensionalData interface:
for f in (:dims, :refdims, :data, :name, :metadata, :layerdims)
    @eval $(f)(c::Counts) = $(f)(c.cts)
end

# extend base Array interface:
for f in (:length, :size, :eachindex, :eltype, :parent,
    :lastindex, :firstindex, :vec, :getindex, :iterate)
    @eval Base.$(f)(c::Counts, args...) = $(f)(c.cts, args...)
end
Base.IteratorSize(::Counts) = Base.HasLength()

# We strictly deal with single inputs here. For multi-inputs, see CausalityTools.jl
"""
    counts([o::OutcomeSpace,] x) → cts::Counts

Discretize/encode `x` into a finite set of outcomes `Ω` specified by the provided
[`OutcomeSpace`](@ref) `o`, then count how often each outcome `Ωᵢ ∈ Ω` (i.e.
each "discretized value", or "encoded symbol") appears.

Return a [`Counts`](@ref) instance which is a vector-like containing the counts.
When displayed, the marginals of the vector are labelled with the outcomes,
so that it is easy to trace what is being counted. Use [`outcomes`](@ref) on the
resulting [`Counts`](@ref) to get these marginals explicitly.

If no [`OutcomeSpace`](@ref) is specified, then [`UniqueElements`](@ref) is used.

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
    d = DimArray(cts, (x1 = outs,))
    return Counts(d)
end
unique!(xc::AbstractStateSpaceSet) = unique!(vec(xc))

# -----------------------------------------------------------------
# Outcomes are simply the labels on the marginal dimensional.
# For 1D, we return the outcomes as-is. For ND, we return
# a tuple of the outcomes --- one element per dimension.
# -----------------------------------------------------------------
outcomes(c::Counts{<:Integer, 1}) = first(c.cts.dims).val.data

# Integer indexing returns the outcomes for that dimension directly.
function outcomes(c::Counts{<:Integer, N}, i::Int) where N
    return c.cts.dims[i].val.data
end

function outcomes(c::Counts{<:Integer, N}) where N
    return map(i -> c.cts.dims[i].val.data, tuple(1:N...))
end

function outcomes(c::Counts{<:Integer, N}, idxs) where N
    map(i -> c.cts.dims[i].val.data, tuple(idxs...))
end

"""
    allcounts(o::OutcomeSpace, x::Array_or_SSSet) → cts::Counts{<:Integer, 1}

Like [`counts`](@ref), but ensures that *all* outcomes `Ωᵢ ∈ Ω`,
where `Ω = outcome_space(o, x)`), are included.
Outcomes that do not occur in the data `x` get 0 count.
"""
function allcounts(o::OutcomeSpace, x::Array_or_SSSet)
    cts = counts(o, x)
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
    return Counts(allcts, (x1 = ospace,))
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
