export AbstractEntropy, Entropy, IndirectEntropy, entropy

abstract type AbstractEntropy end
abstract type Entropy <: AbstractEntropy end
abstract type IndirectEntropy <: AbstractEntropy end

"""
    entropy(e::Entropy, x, est::ProbabilitiesEstimator) → h::Real
    entropy(e::Entropy, ps::Probabilities) → h::Real
Compute a quantity `h`, that qualifies as a (generalized) entropy of `x`,
according to the specified entropy type `e` and the given probability estimator `est`.
Alternatively compute the entropy directly from the existing probabilities `ps`.
In fact, the first method is a 2-lines-of-code wrapper that calls [`probabilities`](@ref)
and gives the result to the second method.

The entropy types that support this interface are "direct" entropies. They always
yield an entropy value given a probability distribution.
Such entropies are theoretically well-founded and are typically called "generalized
entropies". Currently implemented types are:

- [`Renyi`](@ref).
- [`Tsallis`](@ref).
"""
function entropy(e::Entropy, x, est::ProbabilitiesEstimator)
    ps = probabilities(x, est)
    return entropy(e, ps)
end

"""
    entropy(e::IndirectEntropy, x) → h::Real
Compute a quantity `h`, that qualifies as a (generalized) entropy of `x`,
according to the specified inderect entropy estimator `e`.

In contrast to the "typical" way one obtains entropies in the above methods,
these entropy estimators are able to compute Shannon entropies via alternate means,
without explicitly computing some probability distributions.
The available indirect entropies are:

- [`Kraskov`](@ref).
- [`KozachenkoLeonenko`](@ref).
"""
function entropy(e::IndirectEntropy, x)
    error("Method not implemented for entropy type $(nameof(typeof(e)))")
end