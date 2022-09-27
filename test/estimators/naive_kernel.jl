using Neighborhood: KDTree, BruteForce

@test NaiveKernel(0.1) isa NaiveKernel
@test NaiveKernel(0.1, KDTree) isa NaiveKernel
@test NaiveKernel(0.1, BruteForce) isa NaiveKernel

N = 1000
pts = Dataset([rand(2) for i = 1:N]);
ϵ = 0.3
est_direct = NaiveKernel(ϵ, KDTree)
est_tree = NaiveKernel(ϵ, BruteForce)

@test probabilities(pts, est_tree) isa Probabilities
@test probabilities(pts, est_direct) isa Probabilities
p_tree = probabilities(pts, est_tree)
p_direct = probabilities(pts, est_direct)
@test all(p_tree .== p_direct) == true

@test entropy(Renyi(), pts, est_direct) isa Real
@test entropy(Renyi(), pts, est_tree) isa Real
