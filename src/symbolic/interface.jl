export symbolize

"""
# Permutation symbolization

    symbolize(x::AbstractVector{T}, est::SymbolicPermutation) where {T} → Vector{Int}
    symbolize!(s, x::AbstractVector{T}, est::SymbolicPermutation) where {T} → Vector{Int}

If `x` is a univariate time series, first `x` create a delay reconstruction of `x`
using embedding lag `est.τ` and embedding dimension `est.m`, then symbolizing the resulting
state vectors with [`encode_motif`](@ref).

Optionally, the in-place `symbolize!` can be used to put symbols in a pre-allocated
integer vector `s`, where `length(s) == length(x)-(est.m-1)*est.τ`.

    symbolize(x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T} → Vector{Int}
    symbolize!(s, x::AbstractDataset{m, T}, est::SymbolicPermutation) where {m, T} → Vector{Int}

If `x` is an `m`-dimensional dataset, then motif lengths are determined by the dimension of
the input data, and `x` is symbolized by converting each `m`-dimensional
state vector as a unique integer in the range ``1, 2, \\ldots, m-1``, using
[`encode_motif`](@ref).

Optionally, the in-place `symbolize!` can be used to put symbols in a pre-allocated
integer vector `s`, where `length(s) == length(x)`.

## Examples

Symbolize a 7-dimensional dataset. Motif lengths (or order of the permutations) are
inferred to be 7.

```julia
using DelayEmbeddings, Entropies
D = Dataset([rand(7) for i = 1:1000])
s = symbolize(D, SymbolicPermutation())
```

Symbolize a univariate time series by first embedding it in dimension 5 with embedding lag 2.
Motif lengths (or order of the permutations) are therefore 5.

```julia
using DelayEmbeddings, Entropies
n = 5000
x = rand(n)
s = symbolize(x, SymbolicPermutation(m = 5, τ = 2))
```

The integer vector `s` now has length `n-(m-1)*τ = 4992`, and each `s[i]` contains
the integer symbol for the ordinal pattern of state vector `x[i]`.

[^Berger2019]: Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.

# Gaussian symbolization

    symbolize(x::AbstractVector, s::GaussianSymbolization)

Map the elements of `x` to a symbol time series according to the Gaussian symbolization
scheme `s`.

## Examples

```jldoctest; setup = :(using Entropies)
julia> x = [0.1, 0.4, 0.7, -2.1, 8.0, 0.9, -5.2];

julia> Entropies.symbolize(x, GaussianSymbolization(5))
7-element Vector{Int64}:
 3
 3
 3
 2
 5
 3
 1
```

See also: [`GaussianSymbolization`](@ref).
"""
function symbolize end
