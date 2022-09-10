export entropy_tsallis

"""
    entropy_tsallis(p::Probabilities; k = 1, q = 0)


Compute the Tsallis entropy of `x` (Tsallis, 1998)[^Tsallis1988].

    entropy_tsallis(x::Array_or_Dataset, est; k = 1, q = 0)

A convenience syntax, which calls first `probabilities(x, est)`
and then calculates the entropy of the result (and thus `est` can be anything
the [`probabilities`](@ref) function accepts).

## Description
The Tsallis entropy is a generalization of the Boltzmann–Gibbs entropy,
with `k` standing for the Boltzmann constant. It is defined as
```math
S_q(p) = \\frac{k}{q - 1}\\left(1 - \\sum_{i} p[i]^q\\right)
```

[^Tsallis1988]:
    Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics.
    Journal of statistical physics, 52(1), 479-487.
"""
function entropy_tsallis(prob::Probabilities; k = 1, q = 0)
    # As for entropy_renyi, we copy because if someone initialized Probabilities with 0s,
    # they'd probably want to index the zeros as well.
    probs = Iterators.filter(!iszero, prob.p)
    return k/(q-1)*(1 - sum(p^q for p in probs))
end

function entropy_tsallis(x, est; kwargs...)
    p = probabilities(x, est)
    entropy_tsallis(p; kwargs...)
end