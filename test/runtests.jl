using Test
using Entropies
using DelayEmbeddings 


@testset "Estimators" begin
    @test SymbolicPermutation(2) isa SymbolicPermutation
    @test VisitationFrequency(RectangularBinning(3)) isa VisitationFrequency
end

@testset "Permutation entropy" begin
    est = SymbolicPermutation(2)
    N = 100
    x = Dataset(repeat([1.1 2.2 3.3], N))
    y = Dataset(rand(N, 5))
    
    @testset "Pre-allocated" begin
        s = zeros(Int, N);
        @test entropy!(s, x, est, 1) ≈ 0  # Regular order-1 entropy
        @test entropy!(s, y, est, 1) >= 0 # Regular order-1 entropy
        @test entropy!(s, x, est, 2) ≈ 0  # Higher-order entropy
        @test entropy!(s, y, est, 2) >= 0 # Higher-order entropy
    end
    
    @testset "Not pre-allocated" begin
        @test entropy(x, est, 1) ≈ 0  # Regular order-1 entropy
        @test entropy(y, est, 2) >= 0 # Higher-order entropy
    end
end

@testset "VisitationFrequency" begin
    D = Dataset(rand(100, 3))
    
    binnings = [
        RectangularBinning(3),
        RectangularBinning(0.2),
        RectangularBinning([2, 2, 3]),
        RectangularBinning([0.2, 0.3, 0.3])
    ]

    @testset "Binning test $i" for i in 1:length(binnings)
        est = VisitationFrequency(binnings[i])
        @test probabilities(D, est) isa Vector{T} where T <: Real
        @test entropy(D, est, 1) isa Real # Regular order-1 entropy
        @test entropy(D, est, 3) isa Real # Higher-order entropy

    end
end

@testset "Generalized entropy" begin 
    x = rand(1000)
    xn = x ./ sum(x)
    @test genentropy(2, xn) isa Real
    @test genentropy(1, xn) isa Real
end