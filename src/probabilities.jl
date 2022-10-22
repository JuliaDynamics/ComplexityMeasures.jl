export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export probabilities_and_events
export alphabet_length

"""
    Probabilities(x) → p

A simple wrapper type around an `x::AbstractVector` which ensures that `p` sums to 1.
Behaves identically to `Vector`.
"""
struct Probabilities{T} <: AbstractVector{T}
    p::Vector{T}
    function Probabilities(x::AbstractVector{T}, normed = false) where T <: Real
        if !normed # `normed` is an internal argument that skips checking the sum.
            s = sum(x)
            if s ≠ 1
                x = x ./ s
            end
        end
        return new{T}(x)
    end
end
function Probabilities(x::AbstractVector{<:Integer})
    s = sum(x)
    return Probabilities(x ./ s, true)
end


# extend base Vector interface:
for f in (:length, :size, :eachindex, :eltype,
    :lastindex, :firstindex, :vec, :getindex, :iterate)
    @eval Base.$(f)(d::Probabilities, args...) = $(f)(d.p, args...)
end
Base.IteratorSize(::Probabilities) = Base.HasLength()
@inline Base.sum(::Probabilities{T}) where T = one(T)

"""
An abstract type for probabilities estimators.
"""
abstract type ProbabilitiesEstimator end

"""
    probabilities(x::Array_or_Dataset) → p::Probabilities

Directly count probabilities from the elements of `x` without any discretization,
binning, symbolizing, or any other common processing.
This is mostly useful when `x` contains categorical or integer data.

`probabilities` always returns a [`Probabilities`](@ref) container (`Vector`-like).

`x` is typically an `Array` or a [`Dataset`](@ref), see [Input data for Entropies.jl](@ref).


    probabilities(x::Array_or_Dataset, est::ProbabilitiesEstimator) → p::Probabilities

Calculate probabilities representing `x` based on the provided estimator.
The probabilities may, or may not be ordered, and may, or may not contain 0s, see the
documentation of the individual estimators for more.
Configuration options are always given as arguments to the chosen estimator.
"""
function probabilities end
# See visitation_frequency.jl and rectangular_binning.jl (all in histograms folder)
# for the dispatches of `probabilities` for the convenience methods shown above.


"""
    probabilities!(s, args...)

Similar to `probabilities(args...)`, but allows pre-allocation of temporarily used
containers `s`.

Only works for certain estimators. See for example [`SymbolicPermutation`](@ref).
"""
function probabilities! end



"""
    alphabet_length(x::Array_or_Dataset, est::ProbabilitiesEstimator) → Int

Return the total number of possible symbols/states implied by `estimator` for a given `x`.
For some estimators, this total number is independent of `x`, in which case the
input `x` is ignored and the method `alphabet_length(est)` is called.

If the total number of states cannot be known a priori, an error is thrown.
Primarily used in [`entropy_normalized`](@ref).

## Examples

```jldoctest setup = :(using Entropies)
julia> est = SymbolicPermutation(m = 4);

julia> alphabet_length(rand(42), est) # same as `factorial(m)` for any `x`
24
```
"""
function alphabet_length(::Array_or_Dataset, est::ProbabilitiesEstimator)
    return alphabet_length(est)
end
function alphabet_length(est::ProbabilitiesEstimator)
    error("`alphabet_length` not known/implemented for estimator of type $(typeof(est)).")
end


"""
    probabilities_and_events(x::Array_or_Dataset, est::ProbabilitiesEstimator)
Return `probs, events`. `probs` is exactly [`probabilities`](@ref)`(x, est)`.
`events` is a vector, so that `events[i]` is the event that has probability `probs[i]`.
Naturally, the element type of `events` depends on the estimator.
Each estimator's docstring describes what kind of events it returns.
"""
function probabilities_and_events(::Array_or_Dataset, est::ProbabilitiesEstimator)
    error("Events not yet implemented for estimator $(nameof(typeof(est))).")
end
