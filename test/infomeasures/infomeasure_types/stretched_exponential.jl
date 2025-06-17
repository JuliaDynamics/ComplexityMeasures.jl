# Analytical test cases from Anteneodo & Plastino (1999)
# -----------------------------------------------------

# Minimum value 0 occurs when there is complete certainty (exactly one outcome with
#  probability 1)
p = Probabilities([1.0])
@test information(StretchedExponential(), p) ≈ 0.0

# Maximal value occurs for uniform distribution
N = 10
up = Probabilities(repeat([1/N], N))
b = 2
η = 2.0

@test information(StretchedExponential(η = η, base = b), up) ≈
    information_maximum(StretchedExponential(η = η, base = b), N)

# An experimental time series and probabilities estimator that gives a uniform
# probability distribution.
x = [repeat([0, 1], 5); 0]
est = OrdinalPatterns(m = 2)
@test information(StretchedExponential(η = η, base = b), est, x) ≈
    information_maximum(StretchedExponential(η = η, base = b), total_outcomes(est))


# ---------------------------------------------------------------------------------------------------------
# Self-information tests
# ---------------------------------------------------------------------------------------------------------
# Check experimentally that the self-information expressions are correct by comparing to the 
# regular computation of the measure from a set of probabilities.
function information_from_selfinfo(e::StretchedExponential, probs::Probabilities)
    non0_probs = collect(Iterators.filter(!iszero, vec(probs)))
    return sum(pᵢ * self_information(e, pᵢ) for pᵢ in non0_probs)
end
η = 2
base = 2
p = Probabilities([1//10, 1//5, 1//7, 1//5, 0])
H_AP = StretchedExponential(η = η, base = base)
@test round(information_from_selfinfo(H_AP, p), digits = 5) ≈ round(information(H_AP, p), digits = 5)
