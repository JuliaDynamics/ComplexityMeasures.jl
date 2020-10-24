export PermutationProbabilityEstimator, SymbolicPermutation
export symbolize, entropy, entropy!, probabilities, probabilities!

import DelayEmbeddings: Dataset

"""
A probability estimator based on permutations.
"""
abstract type PermutationProbabilityEstimator <: SymbolicProbabilityEstimator end

"""
    SymbolicPermutation(; b::Real = 2, m::Int = 2)

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
    
    function SymbolicPermutation(; b::Real = 2, m::Int = 2)
        m >= 2 || throw(ArgumentError("Dimensions of individual marginals must be at least 2. Otherwise, symbol sequences cannot be assigned to the marginals. Got m=$(m)."))

        new(b, m)
    end
end

""" 
    symbolize(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T} → Vector{Int}

Symbolize the vectors in `x` using Algorithm 1 from Berger et al. (2019)[^Berger2019].

The symbol length is automatically determined from the dimension of the input data.

## Example 

```julia
using DelayEmbeddings, Entropies
D = Dataset([rand(7) for i = 1:1000])
symbolize(D, SymbolicPermutation())
```

[^Berger2019]: Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.
"""
function symbolize(x::Dataset{D, T}, est::PermutationProbabilityEstimator) where {D, T}
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
    length(s) == length(x) || throw(ArgumentError("Need length(h) == length(x), got `length(h)=$(length(h))` and `length(x)==$(length(x))`."))
    @inbounds for i = 1:length(x)
        s[i] = encode_motif(x[i], N)
    end
    _non0hist(s)
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

function entropy!(s::Vector{Int}, x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}
    p = probabilities!(s, x, est)
    return -sum(p .* log.(est.b, p))
end

"""
    entropy(x::Dataset, est::SymbolicPermutation)
    entropy!(s::Vector{Int}, x::Dataset, est::SymbolicPermutation)

Compute the permutation entropy of `x`. A pre-allocated symbol array `s`, 
where `length(x) = length(s)`, can be provided to save some memory 
allocations if the permutation entropy is to be computed for multiple data sets.
"""
function entropy(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}
    s = zeros(Int, length(x))
    entropy!(s, x, est)
end
