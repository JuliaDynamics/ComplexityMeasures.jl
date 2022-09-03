export disequilibrium

"""
    disequilibrium(x, est)

Compute the disequilibrium ``Q_J`` of `x` using the given estimator `est`.

# Definition

```math
Q_J[P, P_e] =\\dfrac{ S[\\dfrac{(P + P_e)}{2}] - \\dfrac{S[P]}{2}] -  \\dfrac{S[P_e]}{2}] }{Q_{max}},
```

where

```math
Q_{max} = - \\dfrac{1}{2} \\left{ \\dfrac{(d_x d_y)! + 1}{d_x d_y} \\log [(d_x d_y)! + 1] - 2 \\log [2(d_x d_y)!] + \\log[(d_x d_y)!] \\right}
````

# Meaning

The disequilibrium quantifies the information contained in correlational structures beyond
that which is captured by permutation entropy.


"""
function disequilibrium end

# For disequilibrium, we are taking the Shannon entropy of distributions which don't sum
# to 1 (𝐏 .+ 𝐏ₑ). Since `genentropy` is only defined for `Probabilities` instances, we
# define this (only internally used) function.
_shannon_ent(p; base = 2) = -sum(x * log(base, x) for x in p)

function _compute_q(𝐏::AbstractVector{T}, 𝐏ₑ::AbstractVector{T};
        base = MathConstants.e) where T <: Real
    _shannon_ent((𝐏 .+ 𝐏ₑ) ./ 2, base = base) -
    _shannon_ent(𝐏, base = base) / 2 -
    _shannon_ent(𝐏ₑ, base = base) / 2
end

function _compute_q(𝐏::AbstractVector{T}; base = MathConstants.e) where T <: Real
    Pₑ = repeat([1/length(P)], length(P))
    _shannon_ent((𝐏 .+ 𝐏ₑ) ./ 2, base = base) -
    _shannon_ent(𝐏, base = base) / 2 -
    _shannon_ent(𝐏ₑ, base = base) / 2
end

include("generator.jl")
