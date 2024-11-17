
# Analytical tests
# -----------------------------------
# Minimal and equal to zero when probability distribution has only one element...
@test information(Identification(), Probabilities([1.0])) ≈ 0.0

# or minimal when only one probability is nonzero and equal to 1.0
@test information(Identification(), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0

# Maximal for a uniform distribution
x = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4]
@test information_normalized(Identification(), UniqueElements(), x) ≈ 1.0

# Submaximal for a non-uniform distribution
y = [0.1, 0.2, 0.2, 0.25, 0.2, 0.351, 0.312, 0.3, 0.3, 0.4, 0.4]
@test information_normalized(Identification(), UniqueElements(), y) < 1.0


# ---------------------------------------------------------------------------------------------------------
# Self-information tests
# ---------------------------------------------------------------------------------------------------------
# Check experimentally that the self-information expressions are correct by comparing to the 
# regular computation of the measure from a set of probabilities.
function information_from_selfinfo(e::Identification, probs::Probabilities)
    non0_probs = collect(Iterators.filter(!iszero, vec(probs)))
    return sum(pᵢ * self_information(e, pᵢ) for pᵢ in non0_probs)
end
p = Probabilities([1//5, 1//5, 1//5, 0, 1//5])
H_I = Identification()
@test round(information_from_selfinfo(H_I, p), digits = 5) ≈ round(information(H_I, p), digits = 5)
