export BubbleSortSwaps

"""
    BubbleSortSwaps <: CountBasedOutcomeSpace
    BubbleSortSwaps(; m = 3, τ = 1)

The `BubbleSortSwaps` outcome space defined in [Manis2017](@citet)'s 
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
`0:N`, where `N = (m * (m - 1)) / 2 + 1` (the worst-case number of swaps).

## Implements 

- [`codify`](@ref). Returns the number of swaps required for each embedded state vector.

## Examples

```julia
x = rand(100000)
o = BubbleSortSwaps(; m = 5) # 5-dimensional embedding vectors
counts_and_outcomes(o, x)
```
"""
Base.@kwdef struct BubbleSortSwaps{M, T} <: CountBasedOutcomeSpace
    m::M = 3
    τ::T = 1
end

# Add one to the total number of possible swaps because it may happen that we don't 
# need to swap.
total_outcomes(o::BubbleSortSwaps{m}) where {m} =  round(Int, (o.m * (o.m - 1)) / 2) + 1
outcome_space(o::BubbleSortSwaps{m}) where {m} = collect(0:total_outcomes(o))

function counts_and_outcomes(o::BubbleSortSwaps, x)
    encoding = BubbleSwapEncoding{o.m}()
    x_embedded = embed(x, o.m, o.τ)
    observed_outs = encode.(Ref(encoding), x_embedded.data)
    return counts_and_outcomes(UniqueElements(), observed_outs)
end