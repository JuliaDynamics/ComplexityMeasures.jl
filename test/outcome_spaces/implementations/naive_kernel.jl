using ComplexityMeasures.DelayEmbeddings.Neighborhood: KDTree

@test NaiveKernel(0.1; method = KDTree) isa OutcomeSpace

N = 1000
pts = StateSpaceSet([rand(2) for i = 1:N]);
ϵ = 0.3
o_direct = NaiveKernel(ϵ, method = KDTree)
o_tree = NaiveKernel(ϵ, method = BruteForce)

@test probabilities(o_tree, pts) isa Probabilities
@test probabilities(o_direct, pts) isa Probabilities
p_tree = probabilities(o_tree, pts)
p_direct = probabilities(o_direct, pts)
@test all(p_tree .== p_direct) == true

@test issorted(outcome_space(o_tree, pts))
@test issorted(outcomes(o_tree, pts))

@test information(Renyi(), o_direct, pts) isa Real
@test information(Renyi(), o_tree, pts) isa Real

probs, z = probabilities_and_outcomes(o_tree, pts)
@test z == 1:length(pts)
@test outcome_space(o_tree, z) == 1:length(pts)
