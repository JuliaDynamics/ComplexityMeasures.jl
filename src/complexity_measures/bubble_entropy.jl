export BubbleEntropy

"""
    BubbleEntropy <: ComplexityEstimator
    BubbleEntropy(; m = 3, τ = 1, definition = Renyi(q = 2))

The `BubbleEntropy` complexity estimator [Manis2017](@cite) is just a difference
between two entropies, each computed with the [`BubbleSortSwaps`](@ref) outcome space, for
embedding dimensions `m + 1` and `m`, respectively. 

[Manis2017](@citet) use the [`Renyi`](@ref) entropy of order `q = 2` as the 
information measure `definition`, but here you can use any [`InformationMeasure`](@ref).
[Manis2017](@citet) formulates the "bubble entropy" as the normalized measure below, 
while here you can also compute the unnormalized measure.

## Definition

For input data `x`, the "bubble entropy" is computed by first embedding the input data
using embedding dimension `m` and embedding delay `τ` (call the embedded pts `y`), and 
then computing the difference between the two entropies:

```math
BubbleEn_T(τ) = H_T(y, m + 1) - H_T(y, m)
```

where ``H_T(y, m)`` and ``H_T(y, m + 1)`` are entropies of type ``T``
(e.g. [`Renyi`](@ref)) computed with the input data `x` embedded to dimension ``m`` and 
``m+1``, respectively. Use [`complexity`](@ref) to compute this non-normalized version. 
Use [`complexity_normalized`](@ref) to compute the normalized difference of entropies:

```math
BubbleEn_H(τ)^{norm} = 
\\dfrac{H_T(x, m + 1) - H_T(x, m)}{max(H_T(x, m + 1)) - max(H_T(x, m))},
```

where the maximum of the entropies for dimensions `m` and `m + 1` are computed using
[`information_maximum`](@ref).

## Example

```julia
using ComplexityMeasures
x = rand(1000)
est = BubbleEntropy(m = 5, τ = 3)
complexity(est, x)
```
"""
Base.@kwdef struct BubbleEntropy{M, T, D} <: ComplexityEstimator
    m::M = 3
    τ::T = 1
    definition::D = Renyi(q = 2)
end

function complexity(est::BubbleEntropy, x)
    o_m = BubbleSortSwaps(m = est.m)
    o_m₊₁ = BubbleSortSwaps(m = est.m + 1)
    h_m = information(est.definition, o_m, x)
    h_m₊₁ = information(est.definition, o_m₊₁, x)
    return h_m₊₁ - h_m
end

function complexity_normalized(est::BubbleEntropy, x)
    o_m = BubbleSortSwaps(m = est.m)
    o_m₊₁ = BubbleSortSwaps(m = est.m + 1)
    h_m =  information(est.definition, o_m, x)
    h_m₊₁ =  information(est.definition, o_m₊₁, x)

    # The normalized factor as (I think) described in Manis et al. (2017).
    # Their description is a bit unclear to me.
    h_max_m = information_maximum(est.definition, o_m, x)
    h_max_m₊₁ = information_maximum(est.definition, o_m₊₁, x)
    norm_factor = (h_max_m₊₁ - h_max_m) # maximum difference for dims `m` and `m + 1`

    return (h_m₊₁ - h_m)/norm_factor
end