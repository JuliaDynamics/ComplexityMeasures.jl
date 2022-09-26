export entropy_tsallis
export maxentropy_tsallis

"""
    entropy_tsallis(p::Probabilities; k = 1, q = 0, base = MathConstants.e)

Compute the Tsallis entropy of `p` (Tsallis, 1998)[^Tsallis1988].

`base` only applies in the limiting case `q == 1`, in which the Tsallis entropy reduces
to Shannon entropy.

    entropy_tsallis(x::Array_or_Dataset, est; k = 1, q = 0, base = MathConstants.e)

A convenience syntax, which calls first `probabilities(x, est)`
and then calculates the Tsallis entropy of the result (and thus `est` can be anything
the [`probabilities`](@ref) function accepts).

## Description

The Tsallis entropy is a generalization of the Boltzmann-Gibbs entropy,
with `k` standing for the Boltzmann constant. It is defined as

```math
S_q(p) = \\frac{k}{q - 1}\\left(1 - \\sum_{i} p[i]^q\\right)
```

[^Tsallis1988]:
    Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics.
    Journal of statistical physics, 52(1), 479-487.
"""
function entropy_tsallis(prob::Probabilities; k = 1, q = 0, base = MathConstants.e)
    # As for entropy_renyi, we copy because if someone initialized Probabilities with 0s,
    # they'd probably want to index the zeros as well.
    probs = Iterators.filter(!iszero, prob.p)
    if q ≈ 1
        return -sum(p * log(base, p) for p in probs) # Shannon entropy
    else
        return k/(q-1)*(1 - sum(p^q for p in probs))
    end
end

function entropy_tsallis(x::Array_or_Dataset, est; kwargs...)
    p = probabilities(x, est)
    entropy_tsallis(p; kwargs...)
end

# Normalization of Tsallis entropies is also well-defined. See Zhang, Z. (2007). Uniform
# estimates on the Tsallis entropies. Letters in Mathematical Physics, 80(2), 171-181.
"""
    entropy_tsallis_norm(prob::Probabilities, est; k = 1, q = 0)

Computed the normalized Tsallis entropy

```math
S_q(p) = \\dfrac{\\frac{k}{q - 1} \\left(1 - \\sum_{i} p[i]^q \\right)}{\\dfrac{ N^{1 - q} }{1 - q}}

```

where `N` is the alphabet length, or total
number of states, as determined by [`alphabet_length`](@ref). Normalization is only
well-defined for estimators for which the alphabet length is known.
The resulting value is restricted to the interval `[0, 1]`.

    entropy_tsallis_norm(x::Array_or_Dataset, est; q = 1.0, base)

The same as above, but first calls `probabilities(x, est)` first and then calculates the
normalized Tsallis entropy of the result.

[^Zhang2007]: Zhang, Z. (2007). Uniform estimates on the Tsallis entropies. Letters in
    Mathematical Physics, 80(2), 171-181.
"""
function entropy_tsallis_norm(prob::Probabilities, est; k = 1, q = 0)

    # Maximum value is obtained when `pᵢ = 1/N ∀ pᵢ : i ∈ [1, 2, …, N]`, where `N` is the
    # alphabet length, or total number of states (e.g Zhang, Z. (2007). Uniform estimates
    # on the Tsallis entropies. Letters in Mathematical Physics, 80(2), 171-181)
    f = (alphabet_length(est)^(1 - q) - 1) / (1 - q)
    return entropy_tsallis(prob; k = k, q = q) / f
end

function entropy_tsallis_norm(x::Array_or_Dataset, est; kwargs...)
    p = entropy_tsallis(x, est; kwargs...)
    return entropy_tsallis_norm(p; k = k, q = q) / f
end
