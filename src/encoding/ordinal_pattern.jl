using StaticArrays: MVector

export OrdinalPatternEncoding

"""
    OrdinalPatternEncoding <: Encoding
    OrdinalPatternEncoding(; m::Int, lt = Entropies.isless_rand)

An encoding scheme that [`encode`](@ref)s `m`-dimensional permutation/ordinal patterns to
integers and [`decode`](@ref)s these integers to permutation patterns based on the Lehmer
code.

## Usage

Used in [`outcomes`](@ref) with probabilities estimators such as
[`SymbolicPermutation`](@ref) to map state vectors `xᵢ ∈ x` into their
integer symbol representations `πᵢ`.

## Description

The Lehmer code, as implemented here, is a bijection between the set of `factorial(n)`
possible permutations for a length-`n` sequence, and the integers `1, 2, …, n`.

- *Encoding* converts an `m`-dimensional permutation pattern `p` into its unique integer
    symbol ``\\omega \\in \\{0, 1, \\ldots, m - 1 \\}``, using Algorithm 1 in Berger
    et al. (2019)[^Berger2019].
- *Decoding* converts an ``\\omega_i \\in \\Omega` ` to its original permutation pattern.

`OrdinalPatternEncoding` is thus meant to be applied on a *permutation*, i.e.
a vector of indices that would sort some vector in ascending order (in practice: the
result of calling `sortperm(x)` for some input vector `x`).

## Example

```jldoctest
julia> using Entropies

julia> x = [1.2, 5.4, 2.2, 1.1]; encoding = OrdinalPatternEncoding(m = length(x));

julia> xs = sortperm(x)
4-element Vector{Int64}:
 4
 1
 3
 2

julia> s = encode(encoding, xs)
20

julia> decode(encoding, s)
4-element SVector{4, Int64} with indices SOneTo(4):
 4
 1
 3
 2
```

[^Berger2019]:
    Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient
    Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.
"""
Base.@kwdef struct OrdinalPatternEncoding{M <: Integer} <: Encoding
    m::M = 3
    lt::Function = isless_rand
end

function encode(encoding::OrdinalPatternEncoding, perm)
    m = encoding.m
    n = 0
    for i = 1:m-1
        for j = i+1:m
            n += perm[i] > perm[j] ? 1 : 0
        end
        n = (m-i)*n
    end
    # The Lehmer code actually results in 0 being an encoded symbol. Shift by 1, so that
    # encodings are positive integers.
    return n + 1
end

# I couldn't find any efficient algorithm in the literature for converting
# between factorial number system representations and Lehmer codes, so we'll just have to
# use this naive approach for now. It is probably possible to do this in a faster way.
function decode(encoding::OrdinalPatternEncoding, s::Int)
    m = encoding.m
    # Convert integer to its factorial number representation. Each factorial number
    # corresponds to a unique permutation of the numbers `1, 2, ..., m`.
    f = base10_to_factorial(s - 1, m) # subtract 1 because we add 1 in `encode`

    # Reconstruct the permutation from the factorial representation
    xs = 1:m |> collect
    perm = zeros(MVector{m, Int})
    for i = 1:m
        perm[i] = popat!(xs, f[i] + 1)
    end

    return SVector{m, Int}(perm) # converting from SVector to MVector is essentially free
end

"""
    base10_to_factorial(s::Int,
        ndigits::Int = ndigits_in_factorial_base(s)) → f::SVector{ndigits, Int}

Convert a base-10 integer to its factorial number system representation. `f` is a
vector where `f[k]` is the multiplier of `factorial(k - 1)`.

For example, the base-10 integer `567`, in the factorial number system, is
``4\\cdot 5! + 3\\cdot 4! + 2\\cdot 3! + 1\\cdot 2! + 1\\cdot 1! + 0\\cdot 0!``.
For this example, `base10_to_factorial` would return the `SVector` `[4, 3, 2, 1, 1, 0]`.

`ndigits` fixes the number of digits in `f` (this just prepends a zero to `f` for each
extraneous radix/base). This is useful when using factorial number for decoding Lehmer codes
into permutations
"""
function base10_to_factorial(s::Int, ndigits::Int = ndigits_in_factorial_base(s))
    remainders = zeros(MVector{ndigits, Int})
    q = s ÷ 1
    r = s % 1
    remainders[end] = r
    for k = 2:ndigits
        r = q % k
        q = q ÷ k
        remainders[end - k + 1] = r
    end

    return SVector{ndigits, Int}(remainders)
end


""" Compute how many digits a base-10 integer needs in the factorial number system. """
function ndigits_in_factorial_base(n::Int)
    k = 1
    while factorial(k) < n
        k += 1
    end
    return k
end


function isless_rand(a, b)
    if a == b
        rand(Bool)
    elseif a < b
        true
    else
        false
    end
end
