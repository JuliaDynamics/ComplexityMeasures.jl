using Test
using Entropies
using DelayEmbeddings 



@testset "Histogram estimation" begin 
    x = rand(1:10, 100)
    D = Dataset([rand(1:10, 3) for i = 1:100])
    @test non0hist(x) isa AbstractVector{T} where T<:Real
    @test non0hist(D) isa AbstractVector{T} where T<:Real
end

@testset "Generalized entropy" begin 
    x = rand(1000)
    xn = x ./ sum(x)
    @test genentropy(2, xn) isa Real
    @test genentropy(1, xn) isa Real
end

@testset "Probability/entropy estimators" begin
    @test SymbolicPermutation() isa SymbolicPermutation
    @test SymbolicWeightedPermutation() isa SymbolicWeightedPermutation
    @test VisitationFrequency(RectangularBinning(3)) isa VisitationFrequency

    @testset "Permutation entropy" begin
        est = SymbolicPermutation()
        N = 100
        x = Dataset(repeat([1.1 2.2 3.3], N))
        y = Dataset(rand(N, 5))
        
        @testset "Pre-allocated" begin
            s = zeros(Int, N);

            # Probability distributions
            p1 = probabilities!(s, x, est)
            p2 = probabilities!(s, y, est)
            @test sum(p1) ≈ 1.0
            @test sum(p2) ≈ 1.0

            # Entropies
            @test entropy!(s, x, est, 1) ≈ 0  # Regular order-1 entropy
            @test entropy!(s, y, est, 1) >= 0 # Regular order-1 entropy
            @test entropy!(s, x, est, 2) ≈ 0  # Higher-order entropy
            @test entropy!(s, y, est, 2) >= 0 # Higher-order entropy
        end
        
        @testset "Not pre-allocated" begin

            # Probability distributions
            p1 = probabilities(x, est)
            p2 = probabilities(y, est)
            @test sum(p1) ≈ 1.0
            @test sum(p2) ≈ 1.0

            # Entropy
            @test entropy(x, est, 1) ≈ 0  # Regular order-1 entropy
            @test entropy(y, est, 2) >= 0 # Higher-order entropy
        end
    end



    @testset "Weighted permutation entropy" begin 
        m = 4
        τ = 1
        τs = tuple([τ*i for i = 0:m-1]...)
        x = rand(25)
        D = genembed(x, τs)

        # Probability distributions
        p1 = probabilities(x, SymbolicWeightedPermutation(), m = m, τ = τ)
        p2 = probabilities(D, SymbolicWeightedPermutation())
        @test sum(p1) ≈ 1.0
        @test sum(p2) ≈ 1.0
        @test all(p1 .≈ p2)

        # Entropy
        e1 = entropy(D, SymbolicWeightedPermutation())
        e2 = entropy(x, SymbolicWeightedPermutation(), m = m, τ = τ)
        @test e1 ≈ e2
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
end
