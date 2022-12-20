using Entropies.DelayEmbeddings.Neighborhood: KDTree

@test_throws ArgumentError NaiveKernel(0.1) isa NaiveKernel
@test_throws ArgumentError NaiveKernel(0.1, method = KDTree) isa NaiveKernel

N = 1000
pts = Dataset([rand(2) for i = 1:N]);
ϵ = 0.3
est_direct = NaiveKernel(pts, ϵ, method = KDTree)
est_tree = NaiveKernel(pts, ϵ, method = BruteForce)

@test probabilities(est_tree, pts) isa Probabilities
@test probabilities(est_direct, pts) isa Probabilities
p_tree = probabilities(est_tree, pts)
p_direct = probabilities(est_direct, pts)
@test all(p_tree .== p_direct) == true

@test entropy(Renyi(), est_direct, pts) isa Real
@test entropy(Renyi(), est_tree, pts) isa Real

probs, z = probabilities_and_outcomes(est_tree, pts)
@test z == 1:length(pts)
@test outcome_space(est_tree) == 1:length(pts)
