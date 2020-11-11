using Test
using Entropies
using DelayEmbeddings 
using Wavelets
using StaticArrays 

@testset "Histogram estimation" begin 
    x = rand(1:10, 100)
    D = Dataset([rand(1:10, 3) for i = 1:100])
    D2 = [(rand(1:10), rand(1:10, rand(1:10)) for i = 1:100)]
    @test Entropies._non0hist(x) isa Probabilities
    @test Entropies._non0hist(D) isa Probabilities
    @test Entropies._non0hist(D2) isa Probabilities

    @test Entropies._non0hist(x) |> sum ≈ 1.0
    @test Entropies._non0hist(D) |> sum ≈ 1.0
    @test Entropies._non0hist(D2)|> sum ≈ 1.0
end

@testset "Shorthand" begin
    D = Dataset([rand(1:10, 5) for i = 1:100])
    ps, bins = Entropies.binhist(D, 0.2)
    @test Entropies.binhist(D, 0.2) isa Tuple{Probabilities, Vector{<:SVector}}
    @test Entropies.binhist(D, RectangularBinning(0.2)) isa Tuple{Probabilities, Vector{<:SVector}}
    @test Entropies.binhist(D, RectangularBinning(5)) isa Tuple{Probabilities, Vector{<:SVector}}
    @test Entropies.binhist(D, RectangularBinning([5, 3, 4, 2, 2])) isa Tuple{Probabilities, Vector{<:SVector}}
    @test Entropies.binhist(D, RectangularBinning([0.5, 0.3, 0.4, 0.2, 0.2])) isa Tuple{Probabilities, Vector{<:SVector}}

end

@testset "Generalized entropy" begin 
    x = rand(1000)
    xn = x ./ sum(x)
    xp = Probabilities(xn)
    @test genentropy(xp, α = 2) isa Real
    @test genentropy(xp, α = 1) isa Real
    @test_throws MethodError genentropy(xn, α = 2) isa Real
end

@testset "Probability/entropy estimators" begin
    @test CountOccurrences() isa CountOccurrences
    @test SymbolicPermutation() isa SymbolicPermutation
    @test SymbolicWeightedPermutation() isa SymbolicWeightedPermutation
    @test SymbolicAmplitudeAwarePermutation() isa SymbolicAmplitudeAwarePermutation

    @test VisitationFrequency(RectangularBinning(3)) isa VisitationFrequency
    @test TimeScaleMODWT() isa TimeScaleMODWT
    @test TimeScaleMODWT(Wavelets.WT.Daubechies{8}()) isa TimeScaleMODWT

    @testset "Counting based" begin
        D = Dataset(rand(1:3, 5000, 3))
        ts = [(rand(1:4), rand(1:4), rand(1:4)) for i = 1:3000]
        @test Entropies.genentropy(D, CountOccurrences(), α = 2, base = 2) isa Real
    end

    @testset "Permutation entropy" begin
        est = SymbolicPermutation()
        N = 100
        x = Dataset(repeat([1.1 2.2 3.3], N))
        y = Dataset(rand(N, 5))
        z = rand(N)

        @testset "Encoding" begin
            @test encode_motif([2, 3, 1]) isa Int
        end
        
        @testset "Pre-allocated" begin
            s = zeros(Int, N);


            # Probability distributions
            p1 = probabilities!(s, x, est)
            p2 = probabilities!(s, y, est)
            @test sum(p1) ≈ 1.0
            @test sum(p2) ≈ 1.0

            # Entropies
            @test genentropy!(s, x, est, α = 1) ≈ 0  # Regular order-1 entropy
            @test genentropy!(s, y, est, α = 1) >= 0 # Regular order-1 entropy
            @test genentropy!(s, x, est, α = 2) ≈ 0  # Higher-order entropy
            @test genentropy!(s, y, est, α = 2) >= 0 # Higher-order entropy

            # For a time series
            m, τ = 3, 2
            sz = zeros(Int, N - (m-1)*τ)
            @test probabilities!(sz, z, est; m = m, τ = τ) isa Probabilities
            @test probabilities(z, est; m = m, τ = τ) isa Probabilities
            @test genentropy!(sz, z, est; m = m, τ = τ) isa Real
            @test genentropy(z, est; m = m, τ = τ) isa Real
        end
        
        @testset "Not pre-allocated" begin

            # Probability distributions
            p1 = probabilities(x, est)
            p2 = probabilities(y, est)
            @test sum(p1) ≈ 1.0
            @test sum(p2) ≈ 1.0

            # Entropy
            @test genentropy(x, est, α = 1) ≈ 0  # Regular order-1 entropy
            @test genentropy(y, est, α = 2) >= 0 # Higher-order entropy
        end
    end



    @testset "Weighted permutation entropy" begin 
        m = 4
        τ = 1
        τs = tuple([τ*i for i = 0:m-1]...)
        x = rand(100)
        D = genembed(x, τs)

        # Probability distributions
        p1 = probabilities(x, SymbolicWeightedPermutation(), m = m, τ = τ)
        p2 = probabilities(D, SymbolicWeightedPermutation())
        @test sum(p1) ≈ 1.0
        @test sum(p2) ≈ 1.0
        @test all(p1.p .≈ p2.p)

        # Entropy
        e1 = genentropy(D, SymbolicWeightedPermutation())
        e2 = genentropy(x, SymbolicWeightedPermutation(), m = m, τ = τ)
        @test e1 ≈ e2
    end

    @testset "Amplitude-aware permutation entropy" begin 
        m = 4
        τ = 1
        τs = tuple([τ*i for i = 0:m-1]...)
        x = rand(25)
        D = genembed(x, τs)

        # Probability distributions
        p1 = probabilities(x, SymbolicAmplitudeAwarePermutation(), m = m, τ = τ)
        p2 = probabilities(D, SymbolicAmplitudeAwarePermutation())
        @test sum(p1) ≈ 1.0
        @test sum(p2) ≈ 1.0
        @test all(p1.p .≈ p2.p)

        # Entropy
        e1 = genentropy(D, SymbolicAmplitudeAwarePermutation())
        e2 = genentropy(x, SymbolicAmplitudeAwarePermutation(), m = m, τ = τ)
        @test e1 ≈ e2
    end


    @testset "VisitationFrequency" begin
        D = Dataset(rand(100, 3))

        @testset "Counting visits" begin 
            @test marginal_visits(D, RectangularBinning(0.2), 1:2) isa Vector{Vector{Int}}
            @test joint_visits(D, RectangularBinning(0.2)) isa Vector{Vector{Int}}
        end
        
        binnings = [
            RectangularBinning(3),
            RectangularBinning(0.2),
            RectangularBinning([2, 2, 3]),
            RectangularBinning([0.2, 0.3, 0.3])
        ]

        @testset "Binning test $i" for i in 1:length(binnings)
            est = VisitationFrequency(binnings[i])
            @test probabilities(D, est) isa Probabilities
            @test genentropy(D, est, α=1, base = 3) isa Real # Regular order-1 entropy
            @test genentropy(D, est, α=3, base = 2) isa Real # Higher-order entropy
            @test genentropy(D, est, α=3, base = 1) isa Real # Higher-order entropy

        end
    end

    @testset "Wavelet" begin
        N = 200
        a = 10
        t = LinRange(0, 2*a*π, N)
        x = sin.(t .+  cos.(t/0.1)) .- 0.1;

        @testset "TimeScaleMODWT" begin
            wl = WT.Daubechies{4}()
            est = TimeScaleMODWT(wl)

            @test Entropies.get_modwt(x) isa AbstractArray{<:Real, 2}
            @test Entropies.get_modwt(x, wl) isa AbstractArray{<:Real, 2}

            W = Entropies.get_modwt(x)
            Nlevels = maxmodwttransformlevels(x)
            @test Entropies.energy_at_scale(W, 1) isa Real
            @test Entropies.energy_at_time(W, 1) isa Real
            
            @test_throws ErrorException Entropies.energy_at_scale(W, 0)
            @test_throws ErrorException Entropies.energy_at_scale(W, Nlevels + 2)
            @test_throws ErrorException Entropies.energy_at_time(W, 0)
            @test_throws ErrorException Entropies.energy_at_time(W, N+1)

            @test Entropies.relative_wavelet_energy(W, 1) isa Real 
            @test Entropies.relative_wavelet_energies(W, 1:2) isa AbstractVector{<:Real}

            @test Entropies.time_scale_density(x, wl) isa AbstractVector{<:Real}
            @test probabilities(x, TimeScaleMODWT()) isa Probabilities
            @test genentropy(x, TimeScaleMODWT()) isa Real
        end
    end
end
