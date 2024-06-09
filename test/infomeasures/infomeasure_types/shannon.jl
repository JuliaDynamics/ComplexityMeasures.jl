using Random
rng = MersenneTwister(1234)

x = rand(1000)
xp = Probabilities(x)
@test_throws MethodError information(Shannon(), x) isa Real
@test information(Shannon(base = 10), xp) isa Real
@test information(Shannon(base = 2), xp) isa Real
@test information_maximum(Shannon(), 2) == 1

# Analytical tests
# -----------------------------------
# Minimal and equal to zero when probability distribution has only one element...
@test information(Shannon(), Probabilities([1.0])) ≈ 0.0

# or minimal when only one probability is nonzero and equal to 1.0
@test information(Shannon(), Probabilities([1.0, 0.0, 0.0, 0.0])) ≈ 0.0



# ---------------------------------------------------------------------------------------------------------
# Self-information tests
# ---------------------------------------------------------------------------------------------------------
# Check experimentally that the self-information expressions are correct by comparing to the 
# regular computation of the measure from a set of probabilities.
function information_from_selfinfo(e::Shannon, probs::Probabilities)
    non0_probs = collect(Iterators.filter(!iszero, vec(probs)))
    return sum(pᵢ * selfinformation(e, pᵢ) for pᵢ in non0_probs)
end
p = Probabilities([1//10, 1//5, 1//7, 1//5, 0])
Hs = Shannon()
@test round(information_from_selfinfo(Hs, p), digits = 5) ≈ round(information(Hs, p), digits = 5)
