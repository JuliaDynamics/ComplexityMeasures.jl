export entropy_shannon
export maxentropy_shannon

"""
    entropy_shannon(args...; base = MathConstants.e)

Equivalent to `entropy_renyi(args...; base = base, q = 1)` and provided solely for convenience.
Computes the Shannon entropy, given by
```math
H(p) = - \\sum_i p[i] \\log(p[i])
```

See also: [`maxentropy_shannon`](@ref).
"""
entropy_shannon(args...; base = MathConstants.e) =
    entropy_renyi(args...; base = base, q = 1)

"""
    maxentropy_shannon(N::Int, q; base = MathConstants.e)

Convenience function that computes the maximum value of the Shannon entropy, i.e.
``log_{base}(N)``, which is useful for normalization when `N` is known.

See also [`entropy_shannon`](@ref), [`entropy_normalized`](@ref).
"""
maxentropy_shannon(N::Int; base = MathConstants.e) = maxentropy_renyi(N, base = base)
