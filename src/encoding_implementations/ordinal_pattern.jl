using StaticArrays: MVector
using Combinatorics: permutations

export OrdinalPatternEncoding

"""
    OrdinalPatternEncoding <: Encoding
    OrdinalPatternEncoding(m::Int, lt = ComplexityMeasures.isless_rand)

An encoding scheme that [`encode`](@ref)s length-`m` vectors into
their permutation/ordinal patterns and then into the integers based on the Lehmer
code. It is used by [`OrdinalPatterns`](@ref) and similar estimators, see that for
a description of the outcome space.

The ordinal/permutation pattern of a vector `χ` is simply `sortperm(χ)`, which gives the
indices that would sort `χ` in ascending order.

## Description

The Lehmer code, as implemented here, is a bijection between the set of `factorial(m)`
possible permutations for a length-`m` sequence, and the integers `1, 2, …, factorial(m)`.
The encoding step uses algorithm 1 in [Berger2019](@citet), which is
highly optimized.
The decoding step is much slower due to missing optimizations (pull requests welcomed!).

## Example

```jldoctest
julia> using ComplexityMeasures

julia> χ = [4.0, 1.0, 9.0];

julia> c = OrdinalPatternEncoding(3);

julia> i = encode(c, χ)
3

julia> decode(c, i)
3-element SVector{3, Int64} with indices SOneTo(3):
 2
 1
 3
```

If you want to encode something that is already a permutation pattern, then you
can use the non-exported `permutation_to_integer` function.
"""
struct OrdinalPatternEncoding{M, F} <: Encoding
    perm::MVector{M, Int}
    lt::F
end
function OrdinalPatternEncoding{m}(lt::F) where {m,F}
    OrdinalPatternEncoding{m, F}(zero(MVector{m, Int}), lt)
end
function OrdinalPatternEncoding(m = 3, lt::F = isless_rand) where {F}
    return OrdinalPatternEncoding{m, F}(zero(MVector{m, Int}), lt)
end

# So that SymbolicPerm stuff fallback here
total_outcomes(::OrdinalPatternEncoding{m}) where {m} = factorial(m)
function outcome_space(::OrdinalPatternEncoding{m}) where {m}
    collect(SVector{m}(p) for p in permutations(1:m))
end

# Notice that `χ` is an element of a `StateSpaceSet`, so most definitely a static vector in
# our code. However we allow `AbstractVector` if a user wanna use `encode` directly.
function encode(encoding::OrdinalPatternEncoding{m}, χ::AbstractVector) where {m}
    if m != length(χ)
        throw(ArgumentError("Permutation order and length of input must match!"))
    end
    perm = sortperm!(encoding.perm, χ)
    return permutation_to_integer(perm)
end

# The algorithm from Berger (2019). Use this directly if encoding *permutations* instead
# of input vectors that are to be permuted.
function permutation_to_integer(perm)
    m = length(perm)
    n = 0
    for i = 1:m-1
        for j = i+1:m
            n += perm[i] > perm[j] ? 1 : 0
        end
        n = (m-i)*n
    end
    # The Lehmer code actually results in 0 being an encoded symbol. Shift by 1, so that
    # encodings are the positive integers.
    return n + 1
end

# I couldn't find any efficient algorithm in the literature for converting
# between factorial number system representations and Lehmer codes, so we'll just have to
# use this naive approach for now. It is probably possible to do this in a faster way.
function decode(::OrdinalPatternEncoding{m}, s::Int) where {m}
    # Convert integer to its factorial number representation. Each factorial number
    # corresponds to a unique permutation of the numbers `1, 2, ..., m`.
    f::SVector{m, Int} = base10_to_factorial(s - 1, m) # subtract 1 because we add 1 above

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
    if  a < b
        true
    elseif a > b
        false
    else
        rand(Bool)
    end
end
