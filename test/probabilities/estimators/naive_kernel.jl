using ComplexityMeasures.DelayEmbeddings.Neighborhood: KDTree

@test NaiveKernel(0.1; method = KDTree) isa ProbabilitiesEstimator

N = 1000
pts = StateSpaceSet([rand(2) for i = 1:N]);
ϵ = 0.3
est_direct = NaiveKernel(ϵ, method = KDTree)
est_tree = NaiveKernel(ϵ, method = BruteForce)

@test probabilities(est_tree, pts) isa Probabilities
@test probabilities(est_direct, pts) isa Probabilities
p_tree = probabilities(est_tree, pts)
p_direct = probabilities(est_direct, pts)
@test all(p_tree .== p_direct) == true

@test issorted(outcome_space(est_tree, pts))
@test issorted(outcomes(est_tree, pts))

@test entropy(Renyi(), est_direct, pts) isa Real
@test entropy(Renyi(), est_tree, pts) isa Real

probs, z = probabilities_and_outcomes(est_tree, pts)
@test z == 1:length(pts)
@test outcome_space(est_tree, z) == 1:length(pts)
