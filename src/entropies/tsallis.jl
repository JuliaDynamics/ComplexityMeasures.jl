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


"""
    maxentropy_tsallis(N::Int, q; base = MathConstants.e)

Convenience function that computes the maximum value of the generalized Tsallis
entropy with parameter `q` for an `N`-element probability distribution, i.e.
``\\dfrac{N^{1 - q} - 1}{(1 - q)}``, which is useful for normalization when `N` and `q`
is known.

If `q == 1`, then `log(base, N)` is returned.

See also [`entropy_tsallis`](@ref), [`entropy_normalized`](@ref),
[`maxentropy_tsallis`](@ref).
"""
function maxentropy_tsallis(N::Int, q; base = MathConstants.e)
    if q ≈ 1.0
        return log(base, N)
    else
        return (N^(1 - q) - 1) / (1 - q)
    end
end
