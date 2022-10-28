using DelayEmbeddings: genembed, Dataset
using StaticArrays: SVector

@testset "Ordinal patterns" begin
    @test Entropies.encode_motif([2, 3, 1]) isa Int
    @test 0 <= Entropies.encode_motif([2, 3, 1]) <= factorial(3) - 1

    scheme = OrdinalMapping(m = 5, τ = 1)
    N = 100
    x = Dataset(repeat([1.1 2.2 3.3], N))
    y = Dataset(rand(N, 5))
    z = rand(N)

    # Without pre-allocation
    D = genembed(z, [0, -1, -2])
    scheme = OrdinalMapping(m = 5, τ = 2)

    @test Entropies.outcomes(z, scheme) isa Vector{<:Int}
    @test Entropies.outcomes(D, scheme) isa Vector{<:Int}


    # With pre-allocation
    N = 100
    x = rand(N)
    scheme = OrdinalMapping(m = 5, τ = 2)
    s = fill(-1, N-(scheme.m-1)*scheme.τ)

    # if symbolization has occurred, s must have been filled with integers in
    # the range 0:(m!-1)
    @test all(Entropies.outcomes!(s, x, scheme) .>= 0)
    @test all(0 .<= Entropies.outcomes!(s, x, scheme) .< factorial(scheme.m))

    m = 4
    D = Dataset(rand(N, m))
    s = fill(-1, length(D))
    @test all(0 .<= Entropies.outcomes!(s, D, scheme) .< factorial(m))
end

@testset "Gaussian symbolization" begin
     # Li et al. (2018) recommends using at least 1000 data points when estimating
    # dispersion entropy.
    x = rand(1000)
    c = 4
    m = 4
    τ = 1
    s = GaussianMapping(c = c)

    # Symbols should be in the set [1, 2, …, c].
    symbols = Entropies.outcomes(x, s)
    @test all([s ∈ collect(1:c) for s in symbols])

    # Test case from Rostaghi & Azami (2016)'s dispersion entropy paper.
    y = [9.0, 8.0, 1.0, 12.0, 5.0, -3.0, 1.5, 8.01, 2.99, 4.0, -1.0, 10.0]
    scheme = GaussianMapping(3)
    s = outcomes(y, scheme)
    @test s == [3, 3, 1, 3, 2, 1, 1, 3, 2, 2, 1, 3]
end

@testset "isless_rand" begin
    # because permutations are partially random, we sort many times and check that
    # we get *a* (not *the one*) correct answer every time
    for i = 1:50
        s = sortperm([1, 2, 3, 2], lt = Entropies.isless_rand)
        @test s == [1, 2, 4, 3] || s == [1, 4, 2, 3]
    end
end

@testset "Equal-width intervals (rectangular binning)" begin
    N = 10 # each dimension is divides [min, max] into N chunks
    @testset "User-defined grid" begin
        # For datasets
        # --------------------------------
        D = Dataset([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
        r = (1 - (-1)) / N

        # (-1, 1) range for all dims.
        binning_same = FixedRectangularBinning(-1.0, 1.0, N)

        # (axismins[1], axismaxs[1]) gives the range for the 1st dim.
        axismins, axismaxs = (-1.0, -1.0), (1.0, 1.0)
        binning_diff = FixedRectangularBinning(axismins, axismaxs, N)

        # Verify that grids are as expected.
        symb_diff = RectangularBinEncoder(D, binning_diff)
        symb_same = RectangularBinEncoder(D, binning_same)
        @test all(@. symb_diff.edgelengths ≈ r)
        @test all(@. symb_same.edgelengths ≈ r)
        @test all(@. symb_diff.mini ≈ -1.0)
        @test all(@. symb_same.mini ≈ -1.0)

        # bins are indexed from 0, so should get the following symbols
        expected = [SVector(0,0), SVector(4, 4), SVector(9, 9)]
        @test symbolize(D, symb_diff) ==
            symbolize(D, symb_same) ==
            expected

        @test alphabet_length(symb_diff) == N^2
        @test alphabet_length(symb_same) == N^2
        @test alphabet_length(D, symb_diff) == N^2
        @test alphabet_length(D, symb_same) == N^2
        # For univariate timeseries
        # --------------------------------
        # bins are indexed from 0, so should get [0, 4, 9] with N = 10
        x = [-1.0, 0.0, 1.0]
        symb_1D = RectangularBinEncoder(x, binning_same)

        @test alphabet_length(symb_1D) == N
        @test alphabet_length(x, symb_1D) == N
        @test symb_1D.mini ≈ -1.0
        @test symb_1D.edgelengths ≈ (1 - (-1)) / N
        @test symbolize(x, symb_1D) == [0, 4, 9]
    end

    @testset "Grid defined by data" begin
        D = Dataset([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
        r = nextfloat((1.0 - (-1.0)) / N)
        binning_int = RectangularBinning(N)
        binning_intvec = RectangularBinning([N, N])
        binning_float = RectangularBinning(r)
        binning_floatvec = RectangularBinning([r, r])

        symb_int = RectangularBinEncoder(D, binning_int)
        symb_float = RectangularBinEncoder(D, binning_float)
        symb_intvec = RectangularBinEncoder(D, binning_intvec)
        symb_floatvec = RectangularBinEncoder(D, binning_floatvec)

        @test round.(symb_int.edgelengths, digits = 15) ==
            round.(symb_float.edgelengths, digits = 15) ==
            round.(symb_intvec.edgelengths, digits = 15) ==
            round.(symb_floatvec.edgelengths, digits = 15)
        @test all(@. symb_int.edgelengths ≈ r)
        @test all(@. symb_float.edgelengths ≈ r)
        @test all(@. symb_intvec.edgelengths ≈ r)
        @test all(@. symb_floatvec.edgelengths ≈ r)
        @test all(@. symb_int.mini ≈ -1.0)
        @test all(@. symb_float.mini ≈ -1.0)
        @test all(@. symb_intvec.mini ≈ -1.0)
        @test all(@. symb_floatvec.mini ≈ -1.0)

        @test symbolize(D, symb_int) ==
            symbolize(D, symb_intvec) ==
            symbolize(D, symb_float) ==
            symbolize(D, symb_floatvec) ==
            [SVector(0,0), SVector(4,4), SVector(9,9)]

        x = [-1.0, 0.0, 1.0]
        r = (1.0 - (-1.0)) / N
        binning_int = RectangularBinning(N)
        binning_intvec = RectangularBinning([N])
        symb_int = RectangularBinEncoder(x, binning_int)
        symb_float = RectangularBinEncoder(x, binning_float)

        @test all(@. symb_int.edgelengths ≈ r)
        @test all(@. symb_float.edgelengths ≈ r)
        @test all(@. symb_int.mini ≈ -1.0)
        @test all(@. symb_float.mini ≈ -1.0)

        @test symbolize(x, symb_int) ==
            symbolize(x, symb_float) ==
            [0, 4, 9]
    end

    @testset "Alphabet length" begin
        X = Dataset(rand(10, 3))
        x = rand(10)
        rbN = RectangularBinning(5)
        rbNs = RectangularBinning([5, 3, 4])
        rbF = RectangularBinning(0.5)
        rbFs = RectangularBinning([0.5, 0.4, 0.3])

        symbolization_xN = RectangularBinEncoder(x, rbN)
        symbolization_xF = RectangularBinEncoder(x, rbF)
        symbolization_XN = RectangularBinEncoder(X, rbN)
        symbolization_XNs = RectangularBinEncoder(X, rbNs)
        symbolization_XF = RectangularBinEncoder(X, rbF)
        symbolization_XFs = RectangularBinEncoder(X, rbFs)

        @test alphabet_length(x, symbolization_xN) == 5
        @test_throws ArgumentError alphabet_length(x, symbolization_xF)
        @test_throws ArgumentError alphabet_length(x, symbolization_xF)
        @test_throws ArgumentError alphabet_length(x, symbolization_XNs)

        @test alphabet_length(X, symbolization_XN) == 5^3
        @test alphabet_length(X, symbolization_XNs) == 5*3*4
        @test_throws ArgumentError alphabet_length(X, symbolization_XF)
        @test_throws ArgumentError alphabet_length(X, symbolization_XFs)
    end
end

@testset "Missing symbols" begin
    x = [1, 2, 3, 4, 2, 1, 0]
    m, τ = 3, 1
    # With these parameters, embedding vectors and ordinal patterns are
    #    (1, 2, 3) -> (1, 2, 3)
    #    (2, 3, 4) -> (1, 2, 3)
    #    (3, 4, 2) -> (3, 1, 2)
    #    (4, 2, 1) -> (3, 2, 1)
    #    (2, 1, 0) -> (3, 2, 1),
    # so there are three occurring patterns and m! - 3 = 3*2*1 - 3 = 3 missing patterns
    @test missing_outcomes(x, SymbolicPermutation(; m, τ)) == 3

    m, τ = 2, 1
    y = [1, 2, 1, 2] # only two patterns, none missing
    @test missing_outcomes(x, SymbolicPermutation(; m, τ)) == 0
end
