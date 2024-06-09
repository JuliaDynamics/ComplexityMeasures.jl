N = 10
κs = [0.5, 1.0, 2.0, 5.0, 10.0]

probs = Probabilities([1, 3, 6,2, 9])
@test information(Kaniadakis(), probs) isa Real
@test information(Kaniadakis(), probs) >= 0
@test information(Kaniadakis(κ = 0, base = 2), probs) == information(Shannon(base = 2), probs)

p = Probabilities([0, 1])
@test information(Kaniadakis(), p) == 0

# Kaniadakis does not state for which distribution for which Kaniadakis entropy is maximised
@test_throws ErrorException information_maximum(Kaniadakis(), 2)


# ---------------------------------------------------------------------------------------------------------
# Self-information tests
# ---------------------------------------------------------------------------------------------------------
# Check experimentally that the self-information expressions are correct by comparing to the 
# regular computation of the measure from a set of probabilities.
function information_from_selfinfo(e::Kaniadakis, probs::Probabilities)
    e.κ ≈ 1.0 && return information_wm(Shannon(; base = e.base ), probs)
    non0_probs = collect(Iterators.filter(!iszero, vec(probs)))
    return sum(pᵢ * selfinformation(e, pᵢ) for pᵢ in non0_probs)
end
p = Probabilities([1//5, 1//5, 1//5, 0, 1//5])
Hk = Kaniadakis(κ = 2)
@test round(information_from_selfinfo(Hk, p), digits = 5) ≈ round(information(Hk, p), digits = 5)
