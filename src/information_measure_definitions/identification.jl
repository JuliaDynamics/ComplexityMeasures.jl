export Identification

"""
    Identification <: InformationMeasure
    Identification()

Identification entropy (Ahlswede et al., 2006)[^Ahlswede2006].

## Description

The identification entropy is the functional

```math
H_I(p) = 2\\left( 1 - \\sum_{i=1}^N p_i^2 \\right).
```

Details about this entropy definition can be found in Ahlswede et al. (2021)[^Ahlswede2021].

[^Ahlswede2006]:
    Ahlswede, R., Bäumer, L., Cai, N., Aydinian, H., Blinovsky, V., Deppe, C., &
    Mashurian, H. (Eds.). (2006). General theory of information transfer and
    combinatorics (Vol. 68). Berlin/Heidelberg, Germany: Springer.

[^Ahlswede2021]:
    Ahlswede, R., Ahlswede, A., Althöfer, I., Deppe, C., & Tamm, U. (2021). Identification
    and Other Probabilistic Models. Springer International Publishing.
"""
struct Identification <: InformationMeasure end

# Page 375 in Ahlswede et al. (2021)
function information(e::Identification, probs::Probabilities)
    return 2 * (1 - sum(probs .^ 2))
end

# Page 378 in Ahlswede et al. (2021)
function information_maximum(e::Identification, L::Int)
    return 2 * (1 - 1 / L)
end
