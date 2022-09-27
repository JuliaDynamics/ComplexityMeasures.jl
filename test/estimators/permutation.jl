@test SymbolicPermutation() isa SymbolicPermutation
@test SymbolicPermutation(lt = Base.isless) isa SymbolicPermutation
@test SymbolicPermutation(lt = Entropies.isless_rand) isa SymbolicPermutation

@testset "Pre-allocated" begin
    est = SymbolicPermutation(m = 5, τ = 1)
    N = 500
    s = zeros(Int, N);
    x = Dataset(repeat([1.1 2.2 3.3], N))
    y = Dataset(rand(N, 5))
    z = rand(N)

    # Probability distributions
    p1 = probabilities!(s, x, est)
    p2 = probabilities!(s, y, est)
    @test sum(p1) ≈ 1.0
    @test sum(p2) ≈ 1.0

    # Entropies
    @test Entropies.entropy!(Renyi(q = 1), s, x, est) ≈ 0  # Regular order-1 entropy
    @test Entropies.entropy!(Renyi(q = 1), s, y, est) >= 0 # Regular order-1 entropy
    @test Entropies.entropy!(Renyi(q = 2), s, x, est) ≈ 0  # Higher-order entropy
    @test Entropies.entropy!(Renyi(q = 2), s, y, est) >= 0 # Higher-order entropy

    # For a time series
    sz = zeros(Int, N - (est.m-1)*est.τ)
    @test probabilities!(sz, z, est) isa Probabilities
    @test probabilities(z, est) isa Probabilities
    @test Entropies.entropy!(Renyi(), sz, z, est) isa Real
    @test entropy(Renyi(), z, est) isa Real
end

@testset "Not pre-allocated" begin
    est = SymbolicPermutation(m = 3, τ = 1)
    N = 500
    x = Dataset(repeat([1.1 2.2 3.3], N))
    y = Dataset(rand(N, 5))

    # Probability distributions
    p1 = probabilities(x, est)
    p2 = probabilities(y, est)
    @test sum(p1) ≈ 1.0
    @test sum(p2) ≈ 1.0

    # Entropy
    @test entropy(Renyi(q = 1), x, est) ≈ 0  # Regular order-1 entropy
    @test entropy(Renyi(q = 2), y, est) >= 0 # Higher-order entropy
end

@testset "Custom sorting" begin
    m = 4
    τ = 1
    τs = tuple([τ*i for i = 0:m-1]...)
    ts = rand(1:3, 100)
    D = genembed(ts, τs)

    est_isless = SymbolicPermutation(m = 5, τ = 1, lt = Base.isless)
    est_isless_rand = SymbolicPermutation(m = 5, τ = 1, lt = Entropies.isless_rand)
    @test probabilities(D, est_isless) isa Probabilities
    @test probabilities(D, est_isless_rand) isa Probabilities
end
