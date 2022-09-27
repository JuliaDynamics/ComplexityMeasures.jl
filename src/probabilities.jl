export ProbabilitiesEstimator, Probabilities
export probabilities, probabilities!
export alphabet_length

"""
    Probabilities(x) → p
A simple wrapper type around an `x::AbstractVector` which ensures that `p` sums to 1.
Behaves identically to `Vector`.
"""
struct Probabilities{T} <: AbstractVector{T}
    p::Vector{T}
    function Probabilities(x::AbstractVector{T}) where T
        s = sum(x)
        if s ≠ 1
            x = x ./ s
        end
        return new{T}(x)
    end
end

# extend base Vector interface:
for f in (:length, :size, :eachindex, :eltype, :lastindex, :firstindex)
    @eval Base.$(f)(d::Probabilities) = $(f)(d.p)
end
Base.IteratorSize(d::Probabilities) = Base.HasLength()
@inline Base.iterate(d::Probabilities, i = 1) = iterate(d.p, i)
@inline Base.getindex(d::Probabilities, i) = d.p[i]
@inline Base.setindex!(d::Probabilities, v, i) = (d.p[i] = v)
@inline Base.:*(d::Probabilities, x::Number) = d.p * x
@inline Base.sum(d::Probabilities{T}) where T = one(T)

"""
An abstract type for probabilities estimators.
"""
abstract type ProbabilitiesEstimator end
const ProbEst = ProbabilitiesEstimator # shorthand

"""
    probabilities(x::Array_or_Dataset) → p::Probabilities

Directly count probabilities from the elements of `x` without any discretization,
binning, or other processing (mostly useful when `x` contains categorical or integer data).
`probabilities` always returns a [`Probabilities`](@ref) container (`Vector`-like).

`x` is typically an `Array` or a `Dataset`, see [Input data for Entropies.jl](@ref).


    probabilities(x::Array_or_Dataset, est::ProbabilitiesEstimator) → p::Probabilities

Calculate probabilities representing `x` based on the provided estimator.
The probabilities are typically unordered and may or may not contain 0s, see the
documentation of the individual estimators for more.
Configuration options are always given as arguments to the chosen estimator.

    probabilities(x::Array_or_Dataset, ε::AbstractFloat) → p::Probabilities

Convenience syntax which provides probabilities for `x` based on rectangular binning
(i.e. performing a histogram). In short, the state space is divided into boxes of length
`ε`, and formally we use `est = VisitationFrequency(RectangularBinning(ε))`
as an estimator, see [`VisitationFrequency`](@ref).

This method has a linearithmic time complexity (`n log(n)` for `n = length(x)`)
and a linear space complexity (`l` for `l = dimension(x)`).
This allows computation of probabilities (histograms) of high-dimensional
datasets and with small box sizes `ε` without memory overflow and with maximum performance.
To obtain the bin information along with `p`, use [`binhist`](@ref).

    probabilities(x::Array_or_Dataset, n::Integer) → p::Probabilities

Same as the above method, but now each dimension of the data is binned into `n::Int` equal
sized bins instead of bins of length `ε::AbstractFloat`.

"""
function probabilities end

# The histogram related stuff are defined in histogram_estimation.jl file
probabilities(x) = _non0hist(x)

"""
    probabilities!(args...)
Identical to `probabilities(args...)`, but allows pre-allocation of temporarily used
containers.

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