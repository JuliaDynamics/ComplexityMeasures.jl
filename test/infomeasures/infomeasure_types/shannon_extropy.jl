

# Minimized for one-element distributions, where there is total order.
x = [0.1, 0.1, 0.1]
@test information(ShannonExtropy(), UniqueElements(), x) == 0.0
@test information_normalized(ShannonExtropy(), UniqueElements(), x) == 0.0

# Normalized Shannon extropy should be maximized (i.e. be equal to 1) for a
# uniform distribution.
x = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6]
js = information_normalized(ShannonExtropy(), UniqueElements(), x)
@test js ≈ 1

# It should be less than 1 for non-uniform distributions
x = [0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6]
js = information_normalized(ShannonExtropy(), UniqueElements(), x)
@test js < 1

# Correctness of maximum
x = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5]
js = information(ShannonExtropy(base = 2), UniqueElements(), x)
@test js ≈ (3 - 1)*log(2, (3 / (3 - 1)))

# ---------------------------------------------------------------------------------------------------------
# Self-information tests
# ---------------------------------------------------------------------------------------------------------
# Check experimentally that the self-information expressions are correct by comparing to the 
# regular computation of the measure from a set of probabilities.
function information_from_selfinfo(e::ShannonExtropy, probs::Probabilities)
    non0_probs = collect(Iterators.filter(!iszero, vec(probs)))
    return sum((1 - pᵢ) * self_information(e, pᵢ) for pᵢ in non0_probs)
end
p = Probabilities([1//5, 1//5, 1//5, 1//2, 0])
Js = ShannonExtropy()
@test round(information_from_selfinfo(Js, p), digits = 5) ≈ round(information(Js, p), digits = 5)