export ElectronicEntropy

"""
    ElectronicEntropy <: InformationMeasure
    ElectronicEntropy(; h = Shannon(; base = 2), j = ShannonExtropy(; base = 2))

The ["electronic entropy"](https://en.wikipedia.org/wiki/Electronic_entropy) measure is
defined in discrete form in Lad et al. (2015)[^Lad2015] as

```math
H{EL}(p) = H_S(p) + J_S(P),
```

where ``H_S(p)`` is the [`Shannon`](@ref) entropy and ``J_S(p)`` is the [`ShannonExtropy`](@ref)
extropy of the probability vector ``p``.

[^Lad2015]:
    Lad, F., Sanfilippo, G., & Agro, G. (2015). Extropy: Complementary dual of entropy.
"""
struct ElectronicEntropy <: InformationMeasure
    h::Shannon
    j::ShannonExtropy

    function ElectronicEntropy(; h = Shannon(; base = 2), j = ShannonExtropy(; base = 2))
        verify_equal_bases(h, j)
        new(h, j)
    end
end

function verify_equal_bases(h::Shannon, j::ShannonExtropy)
    if h.base != j.base
        s = "The logarithm base must be the same for both the entropy and extropy measure." *
        "Got bases $(h.base) (entropy) and $(j.base) (extropy)."
        throw(ArgumentError(s))
    end
end

function information(e::ElectronicEntropy, probs::Probabilities)
    return information(e.h, probs) + information(e.j, probs)
end

# The electronic entropy is defined as the sum of two quantities with well-defined
# maxima. Its maximum value is therefore the sum of the maximum of each term.
function information_maximum(e::ElectronicEntropy, L::Int)
    return information_maximum(e.h, L) + information_maximum(e.j, L)
end
