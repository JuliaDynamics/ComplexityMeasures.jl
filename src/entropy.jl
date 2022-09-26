export AbstractEntropy, Entropy, IndirectEntropy, entropy

# TODO: Add docstrings here
abstract type AbstractEntropy end
abstract type Entropy <: AbstractEntropy end
abstract type IndirectEntropy <: AbstractEntropy end

"""
    entropy(e::Entropy, x, est::ProbabilitiesEstimator) → h::Real
    entropy(e::Entropy, probs::Probabilities) → h::Real
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

These entropies also have a well defined maximum value for a given probability estimator.
To obtain this value one only needs to call the [`maximum`](@ref) function with the
chosen entropy type and probability estimator.
"""
function entropy(e::Entropy, x, est::ProbabilitiesEstimator)
    ps = probabilities(x, est)
    return entropy(e, ps)
end



"""
    maximum(e::Entropy, est::ProbabilitiesEstimator) → m::Real
Return the maximum value `m` of the given entropy type based on the given estimator.
This function only works if the maximum value is deducable, which is possible only
when the estimator has a known [`alphabet_length`](@ref).

    maximum(e::Entropy, L::Int) → m::Real
Alternatively, compute the maximum entropy from the alphabet length `L` directly.
"""
function Base.maximum(e::Entropy, est::ProbabilitiesEstimator)
    L = alphabet_length(est)
    return maximum(e, L)
end
function Base.maximum(e::Entropy, L::Int)
    error("Method not implemented for entropy type $(nameof(typeof(e))).")
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
    error("Method not implemented for entropy type $(nameof(typeof(e))).")
end

"""
    log_with_base(base) → f
Return a function that computes the logarithm at a given base.
This definitely increases accuracy, and probably also performance.
"""
function log_with_base(base)
    if base == 2
        log2
    elseif base == MathConstants.e
        log
    elseif base == 10
        log10
    else
        x -> log(base, x)
    end
end