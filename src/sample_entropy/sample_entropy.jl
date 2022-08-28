using DelayEmbeddings
using Neighborhood
using StatsBase

export sample_entropy

"""
    sample_entropy(xm = 3, r = 0.5 * StatsBase.std(x), base = MathConstants.e) → SampEn

The sample entropy (Richman & Moorman, 2000)[^Richman2000] is defined as

```math
SampEn(m, r) = \\lim_{N \\to \\infty} [-\\ln \\dfrac{A^{m+1}(r)}{B^m(r)}].
```

## Estimation

To estimate ``SampEn(m,r)``, first construct from `x` the `N-m-1` possible `m`-dimensional
embedding vectors ``{\\bf x}_i^m = (x(i), x(i+1), \\ldots, x(i+m-1))``. Next, compute
``B^{m}(r) = \\sum_{i = 1}^{N-m} B_i^{m}(r)``, where ``B_i^{m}(r)`` is the
number of vectors within radius `r` of ``{\\bf x}_i`` (without self-inclusion).

Finally, repeat the procedure, but with embedding vectors of dimension `m + 1`, and
compute ``A^{m+1}(r) = \\sum_{i = 1}^{N-m} A_i^{m+1}(r)``, where ``A_i^{m+1}(r)`` is the
number of vectors within radius `r` of ``{\\bf x}_i^{m+1}``.

Sample entropy, to the given `base`, is then estimated as

```math
SampEn(m,r) = -\\log_{base}{\\dfrac{A^{m+1}(r)}{B^{m}(r)}}.
```

## Data requirements

If the radius `r` is too small relative to the magnitudes of the `xᵢ ∈ x`, or if `x` is
too short, it is possible that no radius-`r` neighbors are found, so that
`SampEn(m,r) = log(0)`. If logarithms of zeros are encountered, `0.0` is
returned.

## Examples

```jldoctest; setup = :(using Entropies)
julia> x = repeat([0.84, 0.52, 0.46, 0.6], 3);

julia> y = [0.84, 0.52, 0.46, 0.6, 0.47, 0.18, 0.47, 0.88, 0.73, 0.75, 0.3, 0.95];

julia> hx = sample_entropy(x, m = 3, r = 0.5);

julia> hy = sample_entropy(y, m = 3, r = 0.5);

julia> hx, hy
(0.4054651081081643, 1.1451323043030026)
```

[^Richman2000]: Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
"""
function sample_entropy(x; m = 3, r = StatsBase.std(x), base = MathConstants.e)
    N = length(x)
    pts_m = genembed(x, 0:m-1)
    pts_m₊₁ = genembed(x, 0:m)

    metric = Euclidean()
    tree_m = KDTree(pts_m, metric)
    tree_m₊₁ = KDTree(pts_m₊₁, metric)

    theiler = Theiler(0) # w = 0 in the Theiler window excludes the point itself
    idxs_m = bulkisearch(tree_m, pts_m, WithinRange(r), theiler)
    idxs_m₊₁ = bulkisearch(tree_m₊₁, pts_m₊₁, WithinRange(r), theiler)

    # The probability that two sequences will match for m points
    Bᵐ = 0.0

    for idxs in idxs_m
        for idx in idxs
            Bᵐ += 1.0
        end
    end
    Bᵐ /= (N - m - 1) * (N - m)

    # The probability that two sequences will match for m + 1 points
    Aᵐ⁺¹ = 0.0
    for idxs in idxs_m₊₁
        for idx in idxs
            Aᵐ⁺¹ += 1.0
        end
    end
    Aᵐ⁺¹ /= (N - m - 1) * (N - m)

    if (Aᵐ⁺¹ == 0.0 || Bᵐ == 0.0)
        return 0.0
    else
        return -log(base, Aᵐ⁺¹/Bᵐ)
    end
end
