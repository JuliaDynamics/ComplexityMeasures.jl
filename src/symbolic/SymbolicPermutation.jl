export PermutationProbabilityEstimator, SymbolicPermutation
export symbolize, entropy, entropy!, probabilities, probabilities!

import DelayEmbeddings: Dataset

"""
A probability estimator based on permutations.
"""
abstract type PermutationProbabilityEstimator <: SymbolicProbabilityEstimator end

"""
    SymbolicPermutation(m::Int; b::Real = 2)

A symbolic permutation probabilities estimator using motifs of length `m`, based on Bandt & Pompe (2002)[^BandtPompe2002].

If the estimator is used for entropy computation, then the entropy is computed 
to base `b` (the default `b = 2` gives the entropy in bits).

The motif length must be ≥ 2. By default `m = 2`, which is the shortest 
possible permutation length which retains any meaningful dynamical information.

[^BandtPompe2002]: Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural complexity measure for time series." Physical review letters 88.17 (2002): 174102.
"""
struct SymbolicPermutation <: PermutationProbabilityEstimator
    b::Real
    m::Int 
    
    function SymbolicPermutation(m::Int; b::Real = 2)
        m >= 2 || throw(ArgumentError("Dimensions of individual marginals must be at least 2. Otherwise, symbol sequences cannot be assigned to the marginals. Got m=$(m)."))

        new(b, m)
    end
end

""" 
    symbolize(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T} → Vector{Int}

Symbolize the vectors in `x` using Algorithm 1 from Berger et al. (2019)[^Berger2019].

The symbol length is automatically determined from the dimension of the input data.

## Example 

Computing the order 5 permutation entropy for a 7-dimensional dataset.

```julia
using DelayEmbeddings, Entropies
D = Dataset([rand(7) for i = 1:1000])
symbolize(D, SymbolicPermutation(5))
```

[^Berger2019]: Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.
"""
function symbolize(x::Dataset{N, T}, est::PermutationProbabilityEstimator) where {N, T}
    N >= 2 || error("Data must be at least 2-dimensional to compute the permutation entropy. If data is a univariate time series, embed it using `genembed` first.")
    s = zeros(Int, length(x))
    symbolize!(s, x, est)
    return s
end

function fill_symbolvector!(s, x, sp, N::Int)
    @inbounds for i = 1:length(x)
        sortperm!(sp, x[i])
        s[i] = encode_motif(sp, N)
    end
end

""" 
    symbolize!(s::T, x::Dataset, est::SymbolicPermutation) where T <: AbstractVector{Int} → T

Symbolize the vectors in `x`, storing the symbols in the pre-allocated length-`L` integer container `s`,
where `L = length(x)`.
"""
function symbolize!(s::AbstractVector{Int}, x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}
    #= 
    Loop over embedding vectors `E[i]`, find the indices `p_i` that sort each `E[i]`,
    then get the corresponding integers `k_i` that generated the 
    permutations `p_i`. Those integers are the symbols for the embedding vectors
    `E[i]`.
    =#
    sp = zeros(Int, N) # pre-allocate a single symbol vector that can be overwritten.
    fill_symbolvector!(s, x, sp, N)
    
    return s
end

function probabilities!(s::Vector{Int}, x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}
    length(s) == length(x) || throw(ArgumentError("Need length(s) == length(x), got `length(s)=$(length(s))` and `length(x)==$(length(x))`."))
    N >= 2 || error("Data must be at least 2-dimensional to compute the permutation entropy. If data is a univariate time series embed it using `genembed` first.")

    @inbounds for i = 1:length(x)
        s[i] = encode_motif(x[i], N)
    end
    non0hist(s)
end

"""
    probabilities(x::Dataset, est::SymbolicPermutation)
    probabilities!(s::Vector{Int}, x::Dataset, est::SymbolicPermutation)

Compute the unordered probabilities of the occurrence of symbol sequences constructed from the data `x`. 
A pre-allocated symbol array `s`, where `length(x) = length(s)`, can be provided to 
save some memory allocations if the probabilities are to be computed for multiple data sets.
"""
function probabilities(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}
    s = zeros(Int, length(x))
    probabilities!(s, x, est)
end

function entropy!(s::Vector{Int}, x::Dataset{N, T}, est::SymbolicPermutation, α::Real = 1) where {N, T}
    ps = probabilities!(s, x, est)

    α < 0 && throw(ArgumentError("Order of generalized entropy must be ≥ 0."))
    if α ≈ 0 # Hartley entropy, max-entropy
        return log(est.b, length(ps)) 
    elseif α ≈ 1
        return -sum( x*log(est.b, x) for x in ps ) #Shannon entropy
    elseif isinf(α)
        return -log(est.b, maximum(ps)) #Min entropy
    else
        return (1/(1-α))*log(est.b, sum(x^α for x in ps) ) #Renyi α entropy
    end
end

"""
    entropy(x::Dataset, est::SymbolicPermutation, α::Real = 1)
    entropy!(s::Vector{Int}, x::Dataset, est::SymbolicPermutation, α::Real = 1)

Compute the generalized order `α` permutation entropy of `x`, using symbol size `est.m`.

## Probability estimation 

An unordered symbol frequency histogram is obtained by symbolizing the points in `x`,
using [`probabilities(::Dataset{N, T}, ::SymbolicPermutation)`](@ref).
Sum-normalizing this histogram yields a probability distribution over the symbols 

A pre-allocated symbol array `s`, where `length(x) = length(s)`, can be provided to 
save some memory allocations if the permutation entropy is to be computed for multiple data sets.

## Entropy estimation

After the symbolization histogram/distribution has been obtained, the order `α` generalized entropy 
is computed from that sum-normalized symbol distribution.

*Note: Do not confuse the order of the generalized entropy (`α`) with the order `m` of the 
permutation entropy (`est.m`, which controls the symbol size). Permutation entropy is usually 
estimated with `α = 1`, but the implementation here allows the generalized entropy of any 
dimension to be computed from the symbol frequency distribution.*

Let ``p`` be an array of probabilities (summing to 1). Then the Rényi entropy is

```math
H_\\alpha(p) = \\frac{1}{1-\\alpha} \\log \\left(\\sum_i p[i]^\\alpha\\right)
```

and generalizes other known entropies,
like e.g. the information entropy
(``\\alpha = 1``, see [^Shannon1948]), the maximum entropy (``\\alpha=0``,
also known as Hartley entropy), or the correlation entropy
(``\\alpha = 2``, also known as collision entropy).

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960)
[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)


"""
function entropy(x::Dataset{N, T}, est::SymbolicPermutation, α::Real = 1) where {N, T}
    s = zeros(Int, length(x))
    entropy!(s, x, est, α)
end
