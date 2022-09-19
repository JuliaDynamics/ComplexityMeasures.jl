using Test
using Entropies
using DelayEmbeddings
using Wavelets
using StaticArrays
using Neighborhood: KDTree, BruteForce

# TODO: This is how the tests should look like in the end:
defaultname(file) = splitext(basename(file))[1]
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
@testset "Entropies.jl" begin
    testfile("timescales.jl")
    testfile("dispersion.jl")
    testfile("complexity_measures.jl")
end

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
    @test entropy_renyi(x, 100) ≠ NaN
    @test entropy_renyi(x, 0.1) ≠ NaN
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
    @test entropy_renyi(xp, q = 2) isa Real
    @test entropy_renyi(xp, q = 1) isa Real
    @test_throws MethodError entropy_renyi(xn, q = 2) isa Real
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
    @test NaiveKernel(0.1) isa NaiveKernel

    @testset "Counting based" begin
        D = Dataset(rand(1:3, 1000, 3))
        ts = [(rand(1:4), rand(1:4), rand(1:4)) for i = 1:3000]
        @test Entropies.entropy_renyi(D, CountOccurrences(), q = 2, base = 2) isa Real
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

        @test Entropies.entropy_renyi(pts, est_direct, base = 2) isa Real
        @test Entropies.entropy_renyi(pts, est_tree, base = 2) isa Real
    end

    @testset "Symbolization" begin
        @testset "Ordinal patterns" begin
            @test Entropies.encode_motif([2, 3, 1]) isa Int
            @test 0 <= Entropies.encode_motif([2, 3, 1]) <= factorial(3) - 1

            scheme = OrdinalPattern(m = 5, τ = 1)
            N = 100
            x = Dataset(repeat([1.1 2.2 3.3], N))
            y = Dataset(rand(N, 5))
            z = rand(N)

            # Without pre-allocation
            D = genembed(z, [0, -1, -2])
            scheme = OrdinalPattern(m = 5, τ = 2)

            @test Entropies.symbolize(z, scheme) isa Vector{<:Int}
            @test Entropies.symbolize(D, scheme) isa Vector{<:Int}


            # With pre-allocation
            N = 100
            x = rand(N)
            scheme = OrdinalPattern(m = 5, τ = 2)
            s = fill(-1, N-(scheme.m-1)*scheme.τ)

            # if symbolization has occurred, s must have been filled with integers in
            # the range 0:(m!-1)
            @test all(Entropies.symbolize!(s, x, scheme) .>= 0)
            @test all(0 .<= Entropies.symbolize!(s, x, scheme) .< factorial(scheme.m))

            m = 4
            D = Dataset(rand(N, m))
            s = fill(-1, length(D))
            @test all(0 .<= Entropies.symbolize!(s, D, scheme) .< factorial(m))
        end

        @testset "Gaussian symbolization" begin
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
        end

    end

    @testset "Permutation entropy" begin



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
            @test Entropies.entropy_renyi!(s, x, est, q = 1) ≈ 0  # Regular order-1 entropy
            @test Entropies.entropy_renyi!(s, y, est, q = 1) >= 0 # Regular order-1 entropy
            @test Entropies.entropy_renyi!(s, x, est, q = 2) ≈ 0  # Higher-order entropy
            @test Entropies.entropy_renyi!(s, y, est, q = 2) >= 0 # Higher-order entropy

            # For a time series
            sz = zeros(Int, N - (est.m-1)*est.τ)
            @test probabilities!(sz, z, est) isa Probabilities
            @test probabilities(z, est) isa Probabilities
            @test Entropies.entropy_renyi!(sz, z, est) isa Real
            @test entropy_renyi(z, est) isa Real
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
            @test entropy_renyi(x, est, q = 1) ≈ 0  # Regular order-1 entropy
            @test entropy_renyi(y, est, q = 2) >= 0 # Higher-order entropy
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
        e1 = entropy_renyi(D, est)
        e2 = entropy_renyi(x, est)
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
        e1 = entropy_renyi(D, est)
        e2 = entropy_renyi(x, est)
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

        @testset "Binning test $i" for i in eachindex(binnings)
            est = VisitationFrequency(binnings[i])
            @test probabilities(D, est) isa Probabilities
            @test entropy_renyi(D, est, q=1, base = 3) isa Real # Regular order-1 entropy
            @test entropy_renyi(D, est, q=3, base = 2) isa Real # Higher-order entropy
            @test entropy_renyi(D, est, q=3, base = 1) isa Real # Higher-order entropy

        end
    end

    @testset "TransferOperator" begin
        D = Dataset(rand(1000, 3))

        binnings = [
            RectangularBinning(3),
            RectangularBinning(0.2),
            RectangularBinning([2, 2, 3]),
            RectangularBinning([0.2, 0.3, 0.3])
        ]

        @testset "Binning test $i" for i in eachindex(binnings)
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

    @testset "Tsallis" begin
        p = Probabilities(repeat([1/5], 5))
        @assert round(entropy_tsallis(p, q = -1/2, k = 1), digits = 2) ≈ 6.79
    end
end

include("spatial_permutation_tests.jl")
include("nn_tests.jl")
