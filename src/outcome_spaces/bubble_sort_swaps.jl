export BubbleSortSwaps

"""
    BubbleSortSwaps <: CountBasedOutcomeSpace
    BubbleSortSwaps(; m = 3, τ = 1)

The `BubbleSortSwaps` outcome space is based on [Manis2017](@citet)'s 
paper on "bubble entropy". 

## Description

`BubbleSortSwaps` does the following:

- Embeds the input data using embedding dimension `m` and  embedding lag `τ`
- For each state vector in the embedding, counting how many swaps are necessary for
    the bubble sort algorithm to sort state vectors.

For [`counts_and_outcomes`](@ref), we then define a distribution over the number of 
necessary swaps. This distribution can then be used to estimate probabilities using 
[`probabilities_and_outcomes`](@ref), which again can be used to estimate any 
[`InformationMeasure`](@ref). An example of how to compute the "Shannon bubble entropy"
is given below.

## Outcome space

The [`outcome_space`](@ref) for `BubbleSortSwaps` are the integers
`0:N`, where `N = (m * (m - 1)) / 2 + 1` (the worst-case number of swaps). Hence,
the number of [`total_outcomes`](@ref) is `N + 1`.

## Implements 

- [`codify`](@ref). Returns the number of swaps required for each embedded state vector.

## Examples

With the `BubbleSortSwaps` outcome space, we can easily compute a "bubble entropy"
inspired by [Manis2017](@cite). Note: this is not actually a new entropy - it is just 
a new way of discretizing the input data. To reproduce the bubble entropy measure
from [Manis2017](@cite), see [`BubbleEntropy`](@ref).

## Examples

```julia
using ComplexityMeasures
x = rand(100000)
o = BubbleSortSwaps(; m = 5) # 5-dimensional embedding vectors
information(Shannon(; base = 2), o, x)

# We can also compute any other "bubble quantity", for example the 
# "Tsallis bubble extropy", with arbitrary probabilities estimators:
information(TsallisExtropy(), BayesianRegularization(), o, x)
```
"""
Base.@kwdef struct BubbleSortSwaps{M, T} <: CountBasedOutcomeSpace
    m::M = 3
    τ::T = 1
end

# Add one to the total number of possible swaps because it may happen that we don't 
# need to swap.
#total_outcomes(o::BubbleSortSwaps{m}, x) where {m} = total_outcomes(o)
total_outcomes(o::BubbleSortSwaps{m}) where {m} = round(Int, (o.m * (o.m - 1)) / 2) + 1
outcome_space(o::BubbleSortSwaps{m}) where {m} = 0:(total_outcomes(o) - 1)

function counts_and_outcomes(o::BubbleSortSwaps, x)
    observed_outs = codify(o, x)
    return counts_and_outcomes(UniqueElements(), observed_outs)
end

function codify(o::BubbleSortSwaps, x)
    encoding = BubbleSortSwapsEncoding{o.m}()
    x_embedded = vec(embed(x, o.m, o.τ))
    return encode.(Ref(encoding), x_embedded)
end