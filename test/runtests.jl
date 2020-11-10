using Test
using Entropies
using DelayEmbeddings 
using Wavelets


@testset "Histogram estimation" begin 
    x = rand(1:10, 100)
    D = Dataset([rand(1:10, 3) for i = 1:100])
    D2 = [(rand(1:10), rand(1:10, rand(1:10)) for i = 1:100)]
    @test non0hist(x) isa AbstractVector{T} where T<:Real
    @test non0hist(D) isa AbstractVector{T} where T<:Real
    @test non0hist(D2) isa AbstractVector{T} where T<:Real

    @test non0hist(x, normalize = true) |> sum ≈ 1.0
    @test non0hist(D, normalize = true) |> sum ≈ 1.0
    @test non0hist(D2, normalize = true)|> sum ≈ 1.0
end

@testset "Generalized entropy" begin 
    x = rand(1000)
    xn = x ./ sum(x)
    @test genentropy(2, xn) isa Real
    @test genentropy(1, xn) isa Real
end

@testset "Probability/entropy estimators" begin
    @test CountOccurrences() isa CountOccurrences
    @test SymbolicPermutation() isa SymbolicPermutation
    @test SymbolicWeightedPermutation() isa SymbolicWeightedPermutation
    @test SymbolicAmplitudeAwarePermutation() isa SymbolicAmplitudeAwarePermutation
    @test VisitationFrequency(RectangularBinning(3)) isa VisitationFrequency
    @test TimeScaleMODWT() isa TimeScaleMODWT
    @test TimeScaleMODWT(Wavelets.WT.Daubechies{8}()) isa TimeScaleMODWT
    @test Kraskov(k = 2, w = 1) isa Kraskov
    @test Kraskov() isa Kraskov
    @test KozachenkoLeonenko() isa KozachenkoLeonenko
    @test KozachenkoLeonenko(w = 5) isa KozachenkoLeonenko

    @testset "Counting based" begin
        D = Dataset(rand(1:3, 5000, 3))
        ts = [(rand(1:4), rand(1:4), rand(1:4)) for i = 1:3000]
        @test Entropies.genentropy(D, CountOccurrences(), 2, base = 2) isa Real
        @test Entropies.genentropy(ts, CountOccurrences()) isa Real
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
            @test genentropy!(s, x, est, 1) ≈ 0  # Regular order-1 entropy
            @test genentropy!(s, y, est, 1) >= 0 # Regular order-1 entropy
            @test genentropy!(s, x, est, 2) ≈ 0  # Higher-order entropy
            @test genentropy!(s, y, est, 2) >= 0 # Higher-order entropy

            # For a time series
            m, τ = 3, 2
            sz = zeros(Int, N - (m-1)*τ)
            @test probabilities!(sz, z, est; m = m, τ = τ) isa Vector{<:Real}
            @test probabilities(z, est; m = m, τ = τ) isa Vector{<:Real}
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
            @test genentropy(x, est, 1) ≈ 0  # Regular order-1 entropy
            @test genentropy(y, est, 2) >= 0 # Higher-order entropy
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
        @test all(p1 .≈ p2)

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
        @test all(p1 .≈ p2)

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
            @test probabilities(D, est) isa Vector{T} where T <: Real
            @test genentropy(D, est, 1, base = 3) isa Real # Regular order-1 entropy
            @test genentropy(D, est, 3, base = 2) isa Real # Higher-order entropy
            @test genentropy(D, est, 3, base = 0) isa Real # Higher-order entropy

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
            @test probabilities(x, TimeScaleMODWT()) isa AbstractVector{<:Real}
            @test genentropy(x, TimeScaleMODWT(), 1) isa Real
        end
    end

    @testset "Nearest neighbor based" begin 
        m = 4
        τ = 1
        τs = tuple([τ*i for i = 0:m-1]...)
        x = rand(250)
        D = genembed(x, τs)

        est_nn = KozachenkoLeonenko(w = 5)
        est_knn = Kraskov(k = 2, w = 1)

        @test entropy(D, est_nn) isa Real
        @test entropy(D, est_knn) isa Real
    end
end
