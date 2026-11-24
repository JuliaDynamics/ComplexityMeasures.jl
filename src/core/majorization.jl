export lorenz_curve, majorization_gaps, majorizes

"""
    lorenz_curve(p::AbstractVector{<:Real}) → S

Return the partial sums ``S_k(p) = \\sum_{i=1}^k p_{[i]}``, ``k = 1, \\ldots, N``, of the
decreasing rearrangement ``p_{[1]} \\geq p_{[2]} \\geq \\cdots \\geq p_{[N]}`` of `p`.

`p` is typically a [`Probabilities`](@ref) instance. The last element of `S` equals
`sum(p)`, i.e. `1` for normalized probabilities.

See also: [`majorizes`](@ref), [`majorization_gaps`](@ref).
"""
lorenz_curve(p::AbstractVector{<:Real}) = cumsum(sort(p; rev = true))

"""
    majorization_gaps(p, q) → g

Return the element-wise difference `g = lorenz_curve(p) .- lorenz_curve(q)`
between the [`lorenz_curve`](@ref)s of `p` and `q`, which must have the same length.

`p` [`majorizes`](@ref) `q` if and only if all entries of `g` are non-negative. For
normalized probabilities, the last entry is zero up to floating point error.

See also: [`majorizes`](@ref), [`lorenz_curve`](@ref).
"""
function majorization_gaps(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    if length(p) != length(q)
        msg = "`p` and `q` must have the same length. " *
            "Got length(p) = $(length(p)) and length(q) = $(length(q))."
        throw(ArgumentError(msg))
    end
    return lorenz_curve(p) .- lorenz_curve(q)
end

"""
    majorizes(p, q; atol = 0.0, checknorm = true) → Bool

Return `true` if `p` majorizes `q`, written ``p \\succ q``, i.e. if

```math
\\sum_{i=1}^k p_{[i]} \\geq \\sum_{i=1}^k q_{[i]} \\quad \\text{for all } k = 1, \\ldots, N,
```

where ``p_{[1]} \\geq \\cdots \\geq p_{[N]}`` and ``q_{[1]} \\geq \\cdots \\geq q_{[N]}`` are
the decreasing rearrangements of `p` and `q` [marshall1979inequalities](@cite). Each
inequality is tested with tolerance `atol`, i.e. the partial sums of `p` may fall below
those of `q` by at most `atol`.

`p` and `q` must have the same length. Majorization is only meaningful for vectors with
equal total mass, such as two [`Probabilities`](@ref) instances over the same number of
outcomes. By default (`checknorm = true`), an `ArgumentError` is thrown unless both
`p` and `q` sum to `1`. Set `checknorm = false` to skip this check, e.g. to compare
vectors with an equal but non-unit total mass. No normalization is performed.

## Description

Majorization is a partial order on the probability simplex that formalizes "more
concentrated than": ``p \\succ q`` means that `p` is at least as concentrated as `q`.
A vector with all its mass on one outcome majorizes every other vector of the same
length, and the uniform vector is majorized by every other vector. Not all pairs of
vectors are comparable.

Most of [`InformationMeasure`](@ref) in this package are Schur-concave, which means that
it respects this order in reverse: if `majorizes(p, q)`, then
`information(m, p) ≤ information(m, q)` for any discrete information measure `m`.

## Example

```julia
p = Probabilities([0.7, 0.2, 0.1])
q = Probabilities([0.4, 0.4, 0.2])
majorizes(p, q)                                       # true
information(Shannon(), p) ≤ information(Shannon(), q) # true
```

See also: [`lorenz_curve`](@ref), [`majorization_gaps`](@ref).
"""
function majorizes(p::AbstractVector{<:Real}, q::AbstractVector{<:Real};
        atol::Real = 0.0, checknorm::Bool = true)
    if checknorm && !(sum(p) ≈ 1 && sum(q) ≈ 1)
        msg = "`p` and `q` must both sum to 1 (got sum(p) = $(sum(p)) and " *
            "sum(q) = $(sum(q))). Pass `Probabilities` instances, or set " *
            "`checknorm = false` to compare vectors with equal non-unit total mass."
        throw(ArgumentError(msg))
    end
    return all(g -> g ≥ -atol, majorization_gaps(p, q))
end
