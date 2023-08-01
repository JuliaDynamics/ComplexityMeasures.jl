export OutcomeSpaceModel
export outcome_space
export total_outcomes
export missing_outcomes
export outcomes
export frequencies_and_outcomes
export frequencies

"""
    OutcomeSpaceModel

The supertype for all outcome space models.

In ComplexityMeasures.jl, an "outcome space model" has two roles. Firstly, it
defines a set of possible outcomes
``\\Omega = \\{\\omega_1, \\omega_2, \\ldots, \\omega_L \\}`` (some form of discretization),
which defines a set of rules for mapping observed data to each outcome ``\\omega_i``.
It also counts how often each outcome ``\\omega_i`` is observed in the
input data, thus assigning to each outcome ``\\omega_i`` a frequency ``f(\\omega_i)``, such
that ``\\sum_{i=1}^N f(\\omega_i) = N``, where ``N`` is the number of observations in the
input data.

At a higher level, an `OutcomeSpaceModel` is used as an input to a
[`ProbabilitiesEstimator`](@ref), which dictates how the frequency counts of the
outcome space is converted to probabilities.

The element type of ``\\Omega`` varies between outcome space models, but it is guaranteed to be
_hashable_ and _sortable_. This allows for conveniently tracking the frequency of a
specific event across experimental realizations, by using the outcome as a dictionary key
and the frequency as the value for that key (or, alternatively, the key remains the outcome
and one has a vector of probabilities, one for each experimental realization).

Some outcome space models can deduce ``\\Omega`` without knowledge of the input, such as
[`SymbolicPermutation`](@ref). For others, knowledge of input is necessary for concretely
specifying ``\\Omega``, such as [`ValueHistogram`](@ref) with [`RectangularBinning`](@ref).
This only matters for the functions [`outcome_space`](@ref) and [`total_outcomes`](@ref).

All currently implemented outcome space models are listed in a nice table in the
[probabilities estimators](@ref probabilities_estimators) section of the online documentation.
"""
abstract type OutcomeSpaceModel end

###########################################################################################
# Outcome space
###########################################################################################
"""
    outcome_space(o::OutcomeSpaceModel, x) → Ω

Return a sorted container containing all _possible_ outcomes of `o` for input `x`.

For some estimators the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `outcome_space(est)`.
In general it is recommended to use the 2-argument version irrespectively of estimator.
"""
function outcome_space(est::OutcomeSpaceModel)
    error(ErrorException("""
    `outcome_space(est)` not implemented for estimator $(typeof(est)).
    Try calling `outcome_space(est, input_data)`, and if you get the same error, open an issue.
    """))
end
outcome_space(est::OutcomeSpaceModel, x) = outcome_space(est.outcome_space)

"""
    total_outcomes(est::OutcomeSpaceModel, x)

Return the length (cardinality) of the outcome space ``\\Omega`` of `est`.

For some estimators the concrete outcome space is known without knowledge of input `x`,
in which case the function dispatches to `total_outcomes(est)`.
In general it is recommended to use the 2-argument version irrespectively of estimator.
"""
total_outcomes(est::OutcomeSpaceModel, x) = length(outcome_space(est, x))
total_outcomes(est::OutcomeSpaceModel) = length(outcome_space(est))

"""
    missing_outcomes(est::OutcomeSpaceModel, x) → n_missing::Int

Estimate a probability distribution for `x` using the given estimator, then count the number
of missing (i.e. zero-probability) outcomes.

See also: [`MissingDispersionPatterns`](@ref).
"""
function missing_outcomes(o::OutcomeSpaceModel, x::Array_or_SSSet)
    freqs = frequencies(o, x)
    L = total_outcomes(est, x)
    O = count(!iszero, freqs)
    return L - O
end

"""
    outcomes(o::OutcomeSpaceModel, x)

Return all (unique) outcomes contained in `x` according to the given outcome space.
Equivalent to `probabilities_and_outcomes(o, x)[2]`, but for some estimators
it may be explicitly extended for better performance.
"""
function outcomes(o::OutcomeSpaceModel, x)
    return last(frequencies_and_outcomes(o, x))
end

###########################################################################################
# Frequencies. Some estimators use frequencies directly, not probabilities.
###########################################################################################
"""
    frequencies_and_outcomes(o::OutcomeSpaceModel, x) → freqs, Ω

Count frequencies of outcomes in `x` according to the given outcome space.
"""
function frequencies_and_outcomes(o::OutcomeSpaceModel, x)
    error("`frequencies_and_outcomes` not implemented for estimator $(typeof(est)).")
end

"""
    frequencies(o::OutcomeSpaceModel, x)

Estimate frequencies/counts over the outcomes defined by `est` and the input data `x`.
"""
function frequencies(o::OutcomeSpaceModel, x)
    return first(frequencies_and_outcomes(o, x))
end

function allfrequencies(o::OutcomeSpaceModel, x::Array_or_SSSet)
    freqs, outs = frequencies_and_outcomes(o, x)
    ospace = vec(outcome_space(est, x))
    # We first utilize that the outcome space is sorted and sort probabilities
    # accordingly (just in case we have an estimator that is not sorted)
    s = sortperm(outs)
    sort!(outs)
    fs = freqs[s]
    # we now iterate over possible outcomes;
    # if they exist in the observed outcomes, we push their corresponding frequency
    # into the frequencies vector. If not, we push 0 into the frequencies vector!
    allfreqs = eltype(ps)[]
    observed_index = 1 # index of observed outcomes
    for j in eachindex(ospace) # we made outcome space a vector on purpose
        ω = ospace[j]
        ωobs = outs[observed_index]
        if ω ≠ ωobs
            push!(allprobs, 0)
        else
            push!(allprobs, fs[observed_index])
            observed_index += 1
        end
        # Check whether we have exhausted observed outcomes
        if observed_index > length(outs)
            remaining_0s = length(ospace) - j
            append!(allprobs, zeros(Int, remaining_0s))
            break
        end
    end
    return allfreqs
end
