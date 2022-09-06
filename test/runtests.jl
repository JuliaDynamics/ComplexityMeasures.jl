using Test
using Entropies
using DelayEmbeddings
using Wavelets
using StaticArrays
using Neighborhood: KDTree, BruteForce

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
    x = rand(100)
    @test genentropy(x, 100) ≠ NaN
    @test genentropy(x, 0.1) ≠ NaN
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
    @test genentropy(xp, q = 2) isa Real
    @test genentropy(xp, q = 1) isa Real
    @test_throws MethodError genentropy(xn, q = 2) isa Real
end

@testset "Probability/entropy estimators" begin
    @test CountOccurrences() isa CountOccurrences

    @test SymbolicPermutation() isa SymbolicPermutation
    @test SymbolicPermutation(lt = Base.isless) isa SymbolicPermutation
    @test SymbolicPermutation(lt = Entropies.isless_rand) isa SymbolicPermutation
    @test SymbolicWeightedPermutation() isa SymbolicWeightedPermutation
    @test SymbolicWeightedPermutation(lt = Base.isless) isa SymbolicWeightedPermutation
    @test SymbolicWeightedPermutation(lt = Entropies.isless_rand) isa SymbolicWeightedPermutation
    @test SymbolicAmplitudeAwarePermutation() isa SymbolicAmplitudeAwarePermutation
    @test SymbolicAmplitudeAwarePermutation(lt = Base.isless) isa SymbolicAmplitudeAwarePermutation
    @test SymbolicAmplitudeAwarePermutation(lt = Entropies.isless_rand) isa SymbolicAmplitudeAwarePermutation

    @test VisitationFrequency(RectangularBinning(3)) isa VisitationFrequency
    @test TransferOperator(RectangularBinning(3)) isa TransferOperator
    @test TimeScaleMODWT() isa TimeScaleMODWT
    @test TimeScaleMODWT(Wavelets.WT.Daubechies{8}()) isa TimeScaleMODWT
    @test Kraskov(k = 2, w = 1) isa Kraskov
    @test Kraskov() isa Kraskov
    @test KozachenkoLeonenko() isa KozachenkoLeonenko
    @test KozachenkoLeonenko(w = 5) isa KozachenkoLeonenko
    @test NaiveKernel(0.1) isa NaiveKernel

    @testset "Counting based" begin
        D = Dataset(rand(1:3, 1000, 3))
        ts = [(rand(1:4), rand(1:4), rand(1:4)) for i = 1:3000]
        @test Entropies.genentropy(D, CountOccurrences(), q = 2, base = 2) isa Real
    end

    @testset "NaiveKernel" begin
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

        @test Entropies.genentropy(pts, est_direct, base = 2) isa Real
        @test Entropies.genentropy(pts, est_tree, base = 2) isa Real
    end

    @testset "Permutation entropy" begin


        @testset "Encoding and symbolization" begin
            @test Entropies.encode_motif([2, 3, 1]) isa Int
            @test 0 <= Entropies.encode_motif([2, 3, 1]) <= factorial(3) - 1

            est = SymbolicPermutation(m = 5, τ = 1)
            N = 100
            x = Dataset(repeat([1.1 2.2 3.3], N))
            y = Dataset(rand(N, 5))
            z = rand(N)

            # Without pre-allocation
            D = genembed(z, [0, -1, -2])
            est = SymbolicPermutation(m = 5, τ = 2)

            @test Entropies.symbolize(z, est) isa Vector{<:Int}
            @test Entropies.symbolize(D, est) isa Vector{<:Int}


            # With pre-allocation
            N = 100
            x = rand(N)
            est = SymbolicPermutation(m = 5, τ = 2)
            s = fill(-1, N-(est.m-1)*est.τ)

            # if symbolization has occurred, s must have been filled with integers in
            # the range 0:(m!-1)
            @test all(Entropies.symbolize!(s, x, est) .>= 0)
            @test all(0 .<= Entropies.symbolize!(s, x, est) .< factorial(est.m))

            m = 4
            D = Dataset(rand(N, m))
            s = fill(-1, length(D))
            @test all(0 .<= Entropies.symbolize!(s, D, est) .< factorial(m))
        end

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
            @test genentropy!(s, x, est, q = 1) ≈ 0  # Regular order-1 entropy
            @test genentropy!(s, y, est, q = 1) >= 0 # Regular order-1 entropy
            @test genentropy!(s, x, est, q = 2) ≈ 0  # Higher-order entropy
            @test genentropy!(s, y, est, q = 2) >= 0 # Higher-order entropy

            # For a time series
            sz = zeros(Int, N - (est.m-1)*est.τ)
            @test probabilities!(sz, z, est) isa Probabilities
            @test probabilities(z, est) isa Probabilities
            @test genentropy!(sz, z, est) isa Real
            @test genentropy(z, est) isa Real
        end

        @testset "Not pre-allocated" begin
            est = SymbolicPermutation(m = 5, τ = 1)
            N = 500
            x = Dataset(repeat([1.1 2.2 3.3], N))
            y = Dataset(rand(N, 5))

            # Probability distributions
            p1 = probabilities(x, est)
            p2 = probabilities(y, est)
            @test sum(p1) ≈ 1.0
            @test sum(p2) ≈ 1.0

            # Entropy
            @test genentropy(x, est, q = 1) ≈ 0  # Regular order-1 entropy
            @test genentropy(y, est, q = 2) >= 0 # Higher-order entropy
        end
    end



    @testset "Weighted permutation entropy" begin
        m = 4
        τ = 1
        τs = tuple([τ*i for i = 0:m-1]...)
        x = rand(100)
        D = genembed(x, τs)

        # Probability distributions
        est = SymbolicWeightedPermutation(m = m, τ = τ)
        p1 = probabilities(x, est)
        p2 = probabilities(D, est)
        @test sum(p1) ≈ 1.0
        @test sum(p2) ≈ 1.0
        @test all(p1.p .≈ p2.p)

        # Entropy
        e1 = genentropy(D, est)
        e2 = genentropy(x, est)
        @test e1 ≈ e2
    end

    @testset "Amplitude-aware permutation entropy" begin
        m = 4
        τ = 1
        τs = tuple([τ*i for i = 0:m-1]...)
        x = rand(25)
        D = genembed(x, τs)

        est = SymbolicAmplitudeAwarePermutation(m = m, τ = τ)
        # Probability distributions
        p1 = probabilities(x, est)
        p2 = probabilities(D, est)
        @test sum(p1) ≈ 1.0
        @test sum(p2) ≈ 1.0
        @test all(p1.p .≈ p2.p)

        # Entropy
        e1 = genentropy(D, est)
        e2 = genentropy(x, est)
        @test e1 ≈ e2
    end

    @testset "Permutation, custom sorting" begin

        @testset "isless_rand" begin
            # because permutations are partially random, we sort many times and check that
            # we get *a* (not *the one*) correct answer every time
            for i = 1:50
                s = sortperm([1, 2, 3, 2], lt = Entropies.isless_rand)
                @test s == [1, 2, 4, 3] || s == [1, 4, 2, 3]
            end
        end


        m = 4
        τ = 1
        τs = tuple([τ*i for i = 0:m-1]...)
        ts = rand(1:3, 100)
        D = genembed(ts, τs)


        @testset "SymbolicPermutation" begin
            est_isless = SymbolicPermutation(m = 5, τ = 1, lt = Base.isless)
            est_isless_rand = SymbolicPermutation(m = 5, τ = 1, lt = Entropies.isless_rand)
            @test Entropies.symbolize(ts, est_isless) isa Vector{<:Int}
            @test Entropies.symbolize(D, est_isless_rand) isa Vector{<:Int}
            @test probabilities(D, est_isless) isa Probabilities
            @test probabilities(D, est_isless_rand) isa Probabilities
        end

        @testset "SymbolicWeightedPermutation" begin
            est_isless = SymbolicWeightedPermutation(m = 5, τ = 1, lt = Base.isless)
            est_isless_rand = SymbolicWeightedPermutation(m = 5, τ = 1, lt = Entropies.isless_rand)
            @test probabilities(ts, est_isless) isa Probabilities
            @test probabilities(D, est_isless) isa Probabilities
        end

        @testset "SymbolicAmplitudeAwarePermutation" begin
            est_isless = SymbolicAmplitudeAwarePermutation(m = 5, τ = 1, lt = Base.isless)
            est_isless_rand = SymbolicAmplitudeAwarePermutation(m = 5, τ = 1, lt = Entropies.isless_rand)
            @test probabilities(ts, est_isless) isa Probabilities
            @test probabilities(D, est_isless) isa Probabilities
        end
    end


    @testset "VisitationFrequency" begin
        D = Dataset(rand(100, 3))

        @testset "Counting visits" begin
            @test Entropies.marginal_visits(D, RectangularBinning(0.2), 1:2) isa Vector{<:AbstractVector{Int}}
            @test Entropies.joint_visits(D, RectangularBinning(0.2)) isa Vector{<:AbstractVector{Int}}
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
            @test genentropy(D, est, q=1, base = 3) isa Real # Regular order-1 entropy
            @test genentropy(D, est, q=3, base = 2) isa Real # Higher-order entropy
            @test genentropy(D, est, q=3, base = 1) isa Real # Higher-order entropy

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
            @test genentropy(x, TimeScaleMODWT(), q = 1, base = 2) isa Real
            @test probabilities(x, TimeScaleMODWT()) isa Probabilities
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

        @test genentropy(D, est_nn) isa Real
        @test genentropy(D, est_knn) isa Real
    end

    @testset "TransferOperator" begin
        D = Dataset(rand(1000, 3))

        binnings = [
            RectangularBinning(3),
            RectangularBinning(0.2),
            RectangularBinning([2, 2, 3]),
            RectangularBinning([0.2, 0.3, 0.3])
        ]

        @testset "Binning test $i" for i in 1:length(binnings)
            to = Entropies.transferoperator(D, binnings[i])
            @test to isa Entropies.TransferOperatorApproximationRectangular

            iv = invariantmeasure(to)
            @test iv isa InvariantMeasure

            p, bins = invariantmeasure(iv)
            @test p isa Probabilities
            @test bins isa Vector{<:SVector}

            @test probabilities(D, TransferOperator(binnings[i])) isa Probabilities
        end
    end

    @testset "Dispersion entropy" begin
        # Li et al. (2018) recommends using at least 1000 data points when estimating
        # dispersion entropy.
        x = rand(1000)
        n_categories = 4
        m = 4
        τ = 1
        s = GaussianSymbolization(n_categories = n_categories)

        # Symbols should be in the set [1, 2, …, n_categories].
        symbols = Entropies.symbolize(x, s)
        @test all([s ∈ collect(1:n_categories) for s in symbols])

        # Dispersion patterns should have a normalized histogram that sums to 1.0.
        dispersion_patterns = DelayEmbeddings.embed(symbols, m, τ)
        hist = Entropies.dispersion_histogram(dispersion_patterns, length(x), m, τ)
        @test sum(hist) ≈ 1.0

        de = dispersion_entropy(x, s, m = 4, τ = 1)
        @test typeof(de) <: Real
        @test de >= 0.0
    end

    @testset "Tsallis" begin
        p = Probabilities(repeat([1/5], 5))
        @assert round(tsallisentropy(p, q = -1/2, k = 1), digits = 2) ≈ 6.79
    end

    @testset "Reverse dispersion entropy" begin
        est = ReverseDispersion()
        @test Probabilities(est) isa Probabilities

        # RDE is minimal when all probabilities are equal. Normalized RDE should then → 0.
        m, n_categories = 3, 5
        ps = Probabilities(repeat([1/n_categories^m], n_categories^m))
        rde_eq = Entropies.distance_to_whitenoise(ps, n_categories, m)
        @test round(rde_eq, digits = 10) ≈ 0.0

        # RDE measures deviation from white noise, so for long enough
        # time series, normalized values should approach zero.
        rde = entropy_reverse_dispersion(rand(100000), m = 5, normalize = true)
        @test round(rde, digits = 3) ≈ 0.0

        # RDE is minimal when all symbol *embedding vectors*are equal.
        # Normalized RDE should then → 1. This situtation arises
        # when the input only has one unique element.
        # Note: the input repeat([1, 2], 10), for example, would *not* give equal
        # probabilities,because symbolization occurs *before* the embedding, slightly
        # skewing the probabilities due to data entries lost during embedding.
        x = repeat([1.0], 100)
        rde_max = entropy_reverse_dispersion(x, normalize = true)
        @test rde_max ≈ 1.0


        # In all situations except those above, RDE ∈ (0.0, 1.0)
        x = repeat([1, 2, 3, 4, 5, 4, 3, 2, 1, 0], 100)
        res = entropy_reverse_dispersion(x, normalize = true)
        @test 0.0 < res < 1.0
    end
end
