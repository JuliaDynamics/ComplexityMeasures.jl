export extropy
export extropy_maximum
export extropy_normalized

"""
    DiscreteExtropyEstimator
    DiscExtropyEst # alias

Supertype of all discrete extropy estimators.

Currently only the [`MLExtropy`](@ref) estimator is provided,
which does not need to be used, as using an [`ExtropyDefinition`](@ref) directly in
[`extropy`](@ref) is possible. In the future, more advanced estimators will
be added.
"""
abstract type DiscreteExtropyEstimator end
const DiscExtropyEst = DiscreteExtropyEstimator

# Dummy estimator that doesn't actually change anything from the definitions
"""
    MLExtropy(e::ExtropyDefinition) <: DiscreteExtropyEstimator

Standing for "maximum likelihood extropy", and also called empirical/naive/plug-in,
this estimator calculates the extropy exactly as defined in the given
[`ExtropyDefinition`](@ref) directly from a probability mass function.
"""
struct MLExtropy{E<:ExtropyDefinition} <: DiscreteExtropyEstimator
    definition::E
end

"""
    extropy(e::ExtropyDefinition, est::ProbabilitiesEstimator, x)

Compute the extropy of type `e` from the probabilities estimated from `x`
using the given [`ProbabilitiesEstimator`](@ref).

See also: [`ExtropyDefinition`](@ref).
"""
function extropy end

function extropy(e::ExtropyDefinition, est::ProbabilitiesEstimator, x)
    ps = probabilities(est, x)
    return extropy(e, ps)
end

# dispatch for `extropy(e::EntropyDefinition, ps::Probabilities)`
# is in the individual extropy definitions files

# Convenience
extropy(est::ProbabilitiesEstimator, x) = entropy(ShannonExtropy(), est, x)
extropy(probs::Probabilities) = extropy(ShannonExtropy(), probs)
extropy(e::MLExtropy, args...) = extropy(e.definition, args...)

"""
    extropy_maximum(e::ExtropyDefinition, est::ProbabilitiesEstimator, x)

Compute the extropy of type `e` from the probabilities estimated from `x`
using the given [`ProbabilitiesEstimator`](@ref).

See also: [`ExtropyDefinition`](@ref).
"""
function extropy_maximum end

function extropy_maximum(e::ExtropyDefinition, est::ProbabilitiesEstimator, x)
    L = total_outcomes(est, x)
    return entropy_maximum(e, L)
end

function extropy_maximum(e::ExtropyDefinition, est::ProbabilitiesEstimator)
    L = total_outcomes(est)
    return entropy_maximum(e, L)
end

function extropy_maximum(e::ExtropyDefinition, ::Int)
    error("not implemented for extropy type $(nameof(typeof(e))).")
end
extropy_maximum(e::MLExtropy, args...) = entropy_maximum(e.definition, args...)

"""
    extropy_normalized([e::DiscreteExtropyEstimator,] est::ProbabilitiesEstimator, x) → h̃

Return `j̃ ∈ [0, 1]`, the normalized discrete extropy of `x`, i.e. the value of [`extropy`](@ref)
divided by the maximum value for `e`, according to the given probabilities estimator.

Instead of a discrete extropy estimator, an [`ExtropyDefinition`](@ref)
can be given as first argument. If `e` is not given, it defaults to `ShannonExtropy()`.

Notice that there is no method
`extropy_normalized(e::DiscreteExtropyEstimator, probs::Probabilities)`,
because there is no way to know
the amount of _possible_ events (i.e., the [`total_outcomes`](@ref)) from `probs`.
"""
function extropy_normalized(e::ExtropyDefinition, est::ProbabilitiesEstimator, x)
    return extropy(e, est, x) / extropy_maximum(e, est, x)
end
function extropy_normalized(est::ProbabilitiesEstimator, x::Array_or_SSSet)
    return extropy_normalized(ShannonExtropy(), est, x)
end
extropy_normalized(e::MLExtropy, est, x) = extropy_normalized(e.definition, est, x)
