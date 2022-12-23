using Entropies, Test

@testset "API" begin
    @test SymbolicPermutation() isa SymbolicPermutation
    @test SymbolicPermutation(lt = Base.isless) isa SymbolicPermutation
    @test SymbolicPermutation(lt = Entropies.isless_rand) isa SymbolicPermutation
    @test total_outcomes(SymbolicPermutation(m = 3)) == factorial(3)
end

@testset "Analytic" begin
    est = SymbolicPermutation(m = 3, τ = 1)
    N = 500
    x = Dataset(repeat([1.1 2.2 3.3], N))
    y = Dataset(rand(N, 3))

    @testset "direct" begin
        p1 = probabilities(est, x)
        p2 = probabilities(est, y)
        @test sum(p1) ≈ 1.0
        @test sum(p2) ≈ 1.0
        @test entropy(Renyi(q = 1), est, x) ≈ 0  # Regular order-1 entropy
        @test entropy(Renyi(q = 2), est, y) >= 0 # Higher-order entropy
    end

    @testset "pre-allocated" begin
        s = zeros(Int, N);
        p1 = probabilities!(s, est, x)
        p2 = probabilities!(s, est, y)
        @test sum(p1) ≈ 1.0
        @test sum(p2) ≈ 1.0
    end

end

@testset "Custom sorting" begin
    m = 4
    τ = 1
    τs = tuple([τ*i for i = 0:m-1]...)
    ts = rand(1:3, (100, m))
    D = Dataset(ts)

    est_isless = SymbolicPermutation(; m, τ = 1, lt = Base.isless)
    est_isless_rand = SymbolicPermutation(; m, τ = 1, lt = Entropies.isless_rand)
    @test probabilities(est_isless, D) isa Probabilities
    @test probabilities(est_isless_rand, D) isa Probabilities
end

@testset "outcomes" begin
    # With m = 2, ordinal patterns and frequencies are:
    # [1, 2] => 3
    # [2, 1] => 2
    x = [1, 2, 1, 2, 1, 2]
    # don't randomize in the case of equal values, so use Base.isless
    est = SymbolicPermutation(m = 2, lt = Base.isless)
    probs, πs = probabilities_and_outcomes(est, x)
    @test πs == SVector{2, Int}.([[1, 2], [2, 1]])
    @test probs == [3/5, 2/5]

    est3 = SymbolicPermutation(m = 3)
    @test outcome_space(est3) == [
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1],
    ]
    @test total_outcomes(est3) == factorial(3)
end