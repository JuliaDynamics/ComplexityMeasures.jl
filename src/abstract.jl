export ProbabilitiesEstimator, entropy, entropy!, genentropy

"""
An abstract type for probabilities estimators.
"""
abstract type ProbabilitiesEstimator end 


function entropy end
function entropy! end

function probabilities end
function probabilities! end

"""
    genentropy(α::Real, p::AbstractArray; base = Base.MathConstants.e)

Compute the entropy of an array of probabilities `p`, assuming that `p` is
sum-normalized.

## Description

Let ``p`` be an array of probabilities (summing to 1). Then the Rényi entropy is

```math
H_\\alpha(p) = \\frac{1}{1-\\alpha} \\log \\left(\\sum_i p[i]^\\alpha\\right)
```

and generalizes other known entropies,
like e.g. the information entropy
(``\\alpha = 1``, see [^Shannon1948]), the maximum entropy (``\\alpha=0``,
also known as Hartley entropy), or the correlation entropy
(``\\alpha = 2``, also known as collision entropy).

[^Rényi1960]: A. Rényi, *Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability*, pp 547 (1960)
[^Shannon1948]: C. E. Shannon, Bell Systems Technical Journal **27**, pp 379 (1948)
"""
function genentropy end 