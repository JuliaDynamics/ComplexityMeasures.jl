@test SymbolicPermutation() isa SymbolicPermutation
@test SymbolicPermutation(lt = Base.isless) isa SymbolicPermutation
@test SymbolicPermutation(lt = Entropies.isless_rand) isa SymbolicPermutation

@testset "Pre-allocated" begin
    @testset "Probabilities" begin
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
    end


    @testset "In-place permutation entropy" begin
        m, τ = 2, 1
        est = SymbolicPermutation(; m, τ)

        # For these two inputs, with m = 2, τ = 1, there should be two symbols (0 and 1)
        # with equal probabilities, so base-2 Shannon entropy should be
        # -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0
        x_timeseries = [repeat([1, 2], 5); 1]
        x_dataset = Dataset(repeat([1 2; 2 1], 3))

        # Pre-allocated integer vectors
        s_timeseries = zeros(Int, length(x_timeseries) - (m - 1)*τ)
        s_dataset = zeros(Int, length(x_dataset))

        @test entropy!(s_timeseries, Shannon(base = 2), x_timeseries, est) ≈ 1.0
        @test entropy!(s_dataset, Shannon(base = 2), x_dataset, est) ≈ 1.0

        # Should default to Shannon base 2
        @test entropy!(s_timeseries, x_timeseries, est) ≈ 1.0
        @test entropy!(s_dataset, x_dataset, est) ≈ 1.0
    end
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
