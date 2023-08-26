export counts
export counts_and_outcomes
export allcounts_and_outcomes
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

function counts(x)
    return fasthist!(copy(x))
end

"""
    counts_and_outcomes(o::OutcomeSpace, x) → (cts::Vector{Int}, Ω::Vector)

Count how often each outcome `Ωᵢ ∈ Ω` appears in the (encoded) input data `x`, where
`Ω = outcome_space(o, x)`.

## Description

For [`OutcomeSpace`](@ref)s that uses [`encode`](@ref) to discretize, it is possible to
count how often each outcome ``\\omega_i \\in \\Omega``, where ``\\Omega`` is the
set of possible outcomes, is observed in the discretized/encoded input data.
Thus, we can assign to each outcome ``\\omega_i`` a count ``f(\\omega_i)``, such that
``\\sum_{i=1}^N f(\\omega_i) = N``, where ``N`` is the number of observations in the
(encoded) input data.
`counts_and_outcomes` returns the counts ``f(\\omega_i)_{obs}``
and outcomes only for the *observed* outcomes ``\\omega_i^{obs}`` (those outcomes
that actually appear in the input data). If you need the counts for
*unobserved* outcomes as well, use [`allcounts_and_outcomes`](@ref).

Returns the `cts` and `Ω` as a tuple where `length(cts) == length(Ω)`.
"""
function counts_and_outcomes(o::OutcomeSpace, x)
    throw(ArgumentError("`counts_and_outcomes` not implemented for estimator $(typeof(o))."))
end

"""
    allcounts_and_outcomes(o::OutcomeSpace, x::Array_or_SSSet) → (cts::Vector{Int}, Ω::Vector)

Like [`counts_and_outcomes`](@ref), but ensures that *all* outcomes `Ωᵢ ∈ Ω`,
where `Ω = outcome_space(o, x)`), are included.

Returns the `cts` and `Ω` as a tuple where `length(cts) == length(Ω)`.
"""
function allcounts_and_outcomes(o::OutcomeSpace, x::Array_or_SSSet)
    cts, outs = counts_and_outcomes(o, x)
    ospace = vec(outcome_space(o, x))
    m = length(ospace)
    allcts = zeros(Int, m)
    for (i, ω) in enumerate(ospace)
        idx = findfirst(oⱼ -> oⱼ == ω, outs)
        if !isnothing(idx)
            allcts[i] = cts[idx]
        end
    end
    return allcts, ospace
end

"""
    allcounts(o::OutcomeSpace, x::Array_or_SSSet) → cts::Vector{Int}

Like [`allcounts_and_outcomes`](@ref), but only returns the counts.
"""
allcounts(o::OutcomeSpace, x::Array_or_SSSet) = first(allcounts_and_outcomes(o, x))

# This function *can* be overridden by count-based `OutcomeSpaces` if it is
# more performant to do so than dispatching to `counts_and_outcomes`.
"""
    counts(o::OutcomeSpace, x) → cts::Vector{Int}

Like [`counts_and_outcomes`](@ref), but only returns the counts.
"""
function counts(o::OutcomeSpace, x)
    return first(counts_and_outcomes(o, x))
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
