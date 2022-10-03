export AbstractEntropy, Entropy, IndirectEntropy
export entropy, entropy_normalized, entropy!

# TODO: Add docstrings here
abstract type AbstractEntropy end
abstract type Entropy <: AbstractEntropy end
abstract type IndirectEntropy <: AbstractEntropy end

###########################################################################################
# Direct API
###########################################################################################
# Notice that StatsBase.jl exports `entropy` and Wavelets.jl exports `Entropy`.
"""
    entropy([e::Entropy,] x, est::ProbabilitiesEstimator) → h::Real
    entropy([e::Entropy,] probs::Probabilities) → h::Real

Compute a (generalized) entropy `h` from `x` according to the specified
entropy type `e` and the given probability estimator `est`.

Alternatively compute the entropy directly from the existing probabilities `probs`.
In fact, the first method is a 2-lines-of-code wrapper that calls [`probabilities`](@ref)
and gives the result to the second method.

`x` is typically an `Array` or a `Dataset`, see [Input data for Entropies.jl](@ref).

The entropy types that support this interface are "direct" entropies. They always
yield an entropy value given a probability distribution.
Such entropies are theoretically well-founded and are typically called "generalized
entropies". Currently implemented types are:

- [`Renyi`](@ref).
- [`Tsallis`](@ref).
- [`Shannon`](@ref), which is a subcase of the above two in the limit `q → 1`.
- [`StretchedExponential`](@ref).

The entropy (first argument) is optional: if not given, `Shannon()` is used instead.

These entropies also have a well defined maximum value for a given probability estimator.
To obtain this value one only needs to call the [`maximum`](@ref) function with the
chosen entropy type and probability estimator. Or, one can use [`entropy_normalized`](@ref)
to obtain the normalized form of the entropy (divided by the maximum).

## Examples
```julia
x = [rand(Bool) for _ in 1:10000] # coin toss
ps = probabilities(x) # gives about [0.5, 0.5] by definition
h = entropy(ps) # gives 1, about 1 bit by definition
h = entropy(Shannon(), ps) # syntactically equivalent to above
h = entropy(Shannon(), x, CountOccurrences()) # syntactically equivalent to above
h = entropy(x, SymbolicPermutation(;m=3)) # gives about 2, again by definition
h = entropy(Renyi(2.0), ps) # also gives 1, order `q` doesn't matter for coin toss
```
"""
function entropy(e::Entropy, x, est::ProbabilitiesEstimator)
    ps = probabilities(x, est)
    return entropy(e, ps)
end
# Convenience
function entropy(x::Array_or_Dataset, est::ProbabilitiesEstimator)
    entropy(Shannon(), x, est)
end
entropy(probs::Probabilities) = entropy(Shannon(), probs)

"""
    entropy!(s, [e::Entropy,] x, est::ProbabilitiesEstimator)

Similar to `probabilities!`, this is an in-place version of [`entropy`](@ref) that allows
pre-allocation of temporarily used containers.

The entropy (second argument) is optional: if not given, `Shannon()` is used instead.

Only works for certain estimators. See for example [`SymbolicPermutation`](@ref).
"""
function entropy!(s::AbstractVector{Int}, e::Entropy, x, est::ProbabilitiesEstimator)
    probs = probabilities!(s, x, est)
    entropy(e, probs)
end

entropy!(s::AbstractVector{Int}, x, est::ProbabilitiesEstimator) =
    entropy!(s, Shannon(), x, est)

###########################################################################################
# Normalize API
###########################################################################################
"""
    maximum(e::Entropy, x, est::ProbabilitiesEstimator) → m::Real

Return the maximum value `m` of the given entropy type based on the given estimator
and the given input `x` (whose values are not important, but layout and type are).

This function only works if the maximum value is dedicable, which is possible only
when the estimator has a known [`alphabet_length`](@ref).

    maximum(e::Entropy, L::Int) → m::Real

Alternatively, compute the maximum entropy from the alphabet length `L` directly.
"""
function Base.maximum(e::Entropy, x, est::ProbabilitiesEstimator)
    L = alphabet_length(x, est)
    return maximum(e, L)
end
function Base.maximum(e::Entropy, ::Int)
    error("Maximum not implemented for entropy type $(nameof(typeof(e))).")
end

"""
    entropy_normalized([e::Entropy,] x, est::ProbabilitiesEstimator) → h̃ ∈ [0, 1]

Return the normalized entropy of `x`, i.e., the value of [`entropy`](@ref) divided
by the maximum value for `e`, according to the given probability estimator.
If `e` is not given, it defaults to `Shannon()`.

Notice that unlike for [`entropy`](@ref), here there is no method
`entropy_normalized(e::Entropy, probs::Probabilities)` because there is no way to know
the amount of _possible_ events (i.e., the [`alphabet_length`](@ref)) from `probs`.
"""
function entropy_normalized(e::Entropy, x, est::ProbabilitiesEstimator)
    return entropy(e, x, est)/maximum(e, x, est)
end
function entropy_normalized(x::Array_or_Dataset, est::ProbabilitiesEstimator)
    return entropy_normalized(Shannon(), x, est)
end

###########################################################################################
# Indirect API
###########################################################################################
"""
    entropy(e::IndirectEntropy, x) → h::Real

Compute the entropy of `x`, here named `h`, according to the specified indirect entropy
estimator `e`.

In contrast to the "typical" way one obtains entropies in the above methods, indirect
entropy estimators compute Shannon entropies via alternate means, without explicitly
computing probability distributions. The available indirect entropies are:

- [`Kraskov`](@ref).
- [`KozachenkoLeonenko`](@ref).
"""
function entropy(::IndirectEntropy, ::Array_or_Dataset) end
function entropy(e::IndirectEntropy, ::Array_or_Dataset, ::ProbabilitiesEstimator)
    error("Indirect entropies like $(nameof(typeof(e))) are not called with probabilities.")
end
function entropy(e::IndirectEntropy, ::Probabilities)
    error("Indirect entropies $(nameof(typeof(e))) are not called with probabilities.")
end

###########################################################################################
# Utils
###########################################################################################
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
