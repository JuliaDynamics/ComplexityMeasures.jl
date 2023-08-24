export Identification

"""
    Identification <: InformationMeasure
    Identification()

Identification entropy (Ahlswede et al., 2006)[Ahlswede2006](@cite).

## Description

The identification entropy is the functional

```math
H_I(p) = 2\\left( 1 - \\sum_{i=1}^N p_i^2 \\right).
```

Details about this entropy definition can be found in Ahlswede et al.
(2021)[Ahlswede2021](@cite).
"""
struct Identification <: Entropy end

# Page 375 in Ahlswede et al. (2021)
function information(e::Identification, probs::Probabilities)
    return 2 * (1 - sum(probs .^ 2))
end

# Page 378 in Ahlswede et al. (2021)
function information_maximum(e::Identification, L::Int)
    return 2 * (1 - 1 / L)
end
