export SequentialCategoryTransitions

"""
    SequentialCategoryTransitions(x; m = 2) <: CountBasedOutcomeSpace

An outcome space based on encoding all possible length-`m` transitions 
between the elements of the input `x`, where each `xᵢ ∈ x` are assumed
to be elements of a categorical set, and the categorical set is 
defined as `c = unique(x)`.

The input data `x` is required to determine all possible symbol transitions.
If `x` is an `AbstractStateSpaceSet`, then it is assumed that `xᵢ in x`
are pre-embedded category vectors. If `x` is an `AbstractVector`, then 
we first form a delay embedding from `x` using embedding dimension `m`
and embedding delay `τ`.

## Example

```julia
julia> x = split("red apples and red apples", " ");

julia> o = SequentialCategoryTransitions(x; m = 2)

julia> counts(o, x)
 Counts{Int64,1} over 3 outcomes
 SubString{String}["and", "red"]     1
 SubString{String}["apples", "and"]  1
 SubString{String}["red", "apples"]  2

julia> probabilities(BayesianRegularization(), o, x)
 Probabilities{Float64,1} over 3 outcomes
 SubString{String}["and", "red"]     0.28571428571428575
 SubString{String}["apples", "and"]  0.28571428571428575
 SubString{String}["red", "apples"]  0.4285714285714286
```
"""
struct SequentialCategoryTransitions{m} <: CountBasedOutcomeSpace
    encoding::SequentialCategoricalEncoding
    τ::Int

    function SequentialCategoryTransitions(x; m = 2, τ = 1)
        encoding = SequentialCategoricalEncoding(symbols = x, m = m)
        return new{m}(encoding,τ)
    end
end

function codify(o::SequentialCategoryTransitions, x)
    n_symbols = length(x) - 1
    encoding = UniqueElementsEncoding(x)
    encoded_category_transitions = Vector{Int}(undef, n_symbols)
    for i in 1:n_symbols
        encoded_category_transitions[i] = encode(encoding, (x[i], x[i+1]))
    end
    return encoded_category_transitions
end

function counts_and_outcomes(o::SequentialCategoryTransitions{m}, x) where m
    if x isa AbstractVector
        dataset = embed(x, m, o.τ)
    else
        dataset = x
    end
    m != dimension(dataset) && throw(ArgumentError(
        "Order of ordinal patterns and dimension of `StateSpaceSet` must match!"
    ))
    return counts_and_outcomes(UniqueElements(), dataset)
end