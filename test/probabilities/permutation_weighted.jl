@test SymbolicWeightedPermutation() isa SymbolicWeightedPermutation
@test SymbolicWeightedPermutation(lt = Base.isless) isa SymbolicWeightedPermutation
@test SymbolicWeightedPermutation(lt = Entropies.isless_rand) isa SymbolicWeightedPermutation

@test total_outcomes(SymbolicWeightedPermutation(m = 3)) == factorial(3)

m = 4
τ = 1
τs = tuple([τ*i for i = 0:m-1]...)
x = rand(100)
D = genembed(x, τs)

# Probability distributions
est = SymbolicWeightedPermutation(m = m, τ = τ)
p1 = probabilities(x, est)
p2 = probabilities(D, est)
@test sum(p1) ≈ 1.0
@test sum(p2) ≈ 1.0
@test all(p1.p .≈ p2.p)

# Entropy
e1 = entropy(Renyi(), est, D)
e2 = entropy(Renyi(), est, x)
@test e1 ≈ e2


@testset "Custom sorting" begin
    m = 4
    τ = 1
    τs = tuple([τ*i for i = 0:m-1]...)
    ts = rand(1:3, 100)
    D = genembed(ts, τs)


    est_isless = SymbolicWeightedPermutation(m = 5, τ = 1, lt = Base.isless)
    est_isless_rand = SymbolicWeightedPermutation(m = 5, τ = 1, lt = Entropies.isless_rand)
    @test probabilities(ts, est_isless) isa Probabilities
    @test probabilities(D, est_isless) isa Probabilities
end

# With m = 2, ordinal patterns and frequencies are:
# [1, 2] => 3
# [2, 1] => 2
x = [1, 2, 1, 2, 1, 2]
# don't randomize in the case of equal values, so use Base.isless
est = SymbolicWeightedPermutation(m = 2, lt = Base.isless)
probs, πs = probabilities_and_outcomes(x, est)
@test πs == [0, 1]
# TODO: probabilities should be explicitly tested too.
