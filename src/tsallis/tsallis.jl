export tsallisentropy

"""
    tsallisentropy(x::Probabilities; k = 1, q = 0)

Compute the Tsallis entropy of `x` (Tsallis, 1998)[^Tsallis1988].

```math
S_q(\\bf p) = \\dfrac{k}{q - 1}\\left(1 - \\sum_{p_i \\in \\bf p} \\right)
```

[^Tsallis1988]: Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics. Journal of statistical physics, 52(1), 479-487.
"""
function tsallisentropy(prob::Probabilities; k = 1, q = 0)
    haszero = any(iszero, prob)
    p = if haszero
        i0 = findall(iszero, prob.p)
        # As for genentropy, we copy because if someone initialized Probabilities with 0s,
        # they'd probably want to index the zeros as well.
        deleteat!(copy(prob.p), i0)
    else
        prob.p
    end

    return k/(q-1)*(1 - sum(prob .^ q))
end
