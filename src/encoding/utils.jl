"""
    encode_motif(x, m::Int = length(x)) → s::Int

Encode the length-`m` motif `x` (a vector of indices that would sort some vector `v`
in ascending order) into its unique integer symbol ``s \\in \\{0, 1, \\ldots, m - 1 \\}``,
using Algorithm 1 in Berger et al. (2019)[^Berger2019].

## Example

```julia
v = rand(5)

# The indices that would sort `v` in ascending order. This is now a permutation
# of the index permutation (1, 2, ..., 5)
x = sortperm(v)

# Encode this permutation as an integer.
encode_motif(x)
```
[^Berger2019]: Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.
"""
function encode_motif(x, m::Int = length(x))
    n = 0
    for i = 1:m-1
        for j = i+1:m
            n += x[i] > x[j] ? 1 : 0
        end
        n = (m-i)*n
    end

    return n
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

# I couldn't find any efficient algorithm in the literature for converting
# between factorial number system representations and Lehmer codes, so we'll just have to
# use this naive approach for now. It is probably possible to do this in a faster way.
"""
    decode_motif(s::Int, m::Int) → SVector{m, Int}

Given the integer `s`, which is an encoding of the `m`-element ordinal permutation `π`
based on Lehmer codes, computed using [`encode_motif`](@ref),
compute the original permutation (a permutation of the numbers `1, 2, …, m`).

## Example

```jldoctest
julia> using Entropies

julia> x = [1.2, 5.4, 2.2, 1.1]; m = length(x);

julia> xs = sortperm(x) # [4, 1, 3, 2]
4-element Vector{Int64}:
 4
 1
 3
 2

julia> s = encode_motif(xs, m)
19

julia> decode_motif(19, m)
4-element SVector{4, Int64} with indices SOneTo(4):
 4
 1
 3
 2
```
"""
function decode_motif(s::Int, m::Int)
    # Convert integer to its factorial number representation. Each factorial number
    # corresponds to a unique permutation of the numbers `1, 2, ..., m`.
    f = base10_to_factorial(s, m)

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
