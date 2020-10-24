using Test
using Entropies
using DelayEmbeddings 

@testset "Permutation entropy" begin
    est = SymbolicPermutation(m=2)
    N = 100
    x = Dataset(repeat([1.1 2.2 3.3], N))
    y = Dataset(rand(N, 5))
    
    @testset "Pre-allocated" begin
        s = zeros(Int, N);
        @test entropy!(s, x, est) ≈ 0
        @test entropy!(s, y, est) >= 0
    end
    
    @testset "Not pre-allocated" begin
        @test entropy(x, est) ≈ 0
        @test entropy(y, est) >= 0
    end
end