using StateSpaceSets: Dataset
using DelayEmbeddings: genembed
using StaticArrays: SVector
using ComplexityMeasures: encode, decode
using Statistics: mean, std

@testset "Ordinal patterns" begin
    o = OrdinalPatternEncoding(5)
    @test o isa OrdinalPatternEncoding{5}

    # This is not part of the public API, but this is crucial to test directly to
    # ensure its correctness. It makes no sense to test it though "end-product" code,
    # because there would be no obvious way of debugging the forward/inverse Lehmer-code
    # code from end-product code. We therefore test the internals here.
    @testset "Encoding/decoding" begin
        # Simple example
        enc = OrdinalPatternEncoding(3)
        @test decode(enc, encode(enc, [10, -2, 1])) == SVector{3, Int}(2, 3, 1)
        enc = OrdinalPatternEncoding(4)
        @test decode(enc, encode(enc, [-5, 4, -3, 5])) == SVector{4, Int}(1, 3, 2, 4)

        m = 4
        # All possible permutations for length-4 vectors.
        # Table 1 in Berger et al. These permutations should, in the given order,
        # map onto integers 0, 1, ..., factorial(4) - 1.
        permutations = [[1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4], [1, 3, 4, 2], [1, 4, 2, 3], [1, 4, 3, 2], [2, 1, 3, 4], [2, 1, 4, 3], [2, 3, 1, 4], [2, 3, 4, 1], [2, 4, 1, 3], [2, 4, 3, 1], [3, 1, 2, 4], [3, 1, 4, 2], [3, 2, 1, 4], [3, 2, 4, 1], [3, 4, 1, 2], [3, 4, 2, 1], [4, 1, 2, 3], [4, 1, 3, 2], [4, 2, 1, 3], [4, 2, 3, 1], [4, 3, 1, 2], [4, 3, 2, 1]]
        # Just add some noise to the data that doesn't alter the order.
        xs = [sortperm(p .+ rand(4) .* 0.1) for p in permutations]

        # When using raw input vectors
        encoder = OrdinalPatternEncoding(m)
        encoded_πs = [encode(encoder, xi) for xi in xs]
        decoded_πs = decode.(Ref(encoder), encoded_πs)
        @test all(sort(decoded_πs) .== sort(permutations))
        @test all(encoded_πs .== 1:factorial(m))
        @test all(decoded_πs .== permutations)

        # When input vectors are already permutations
        encoded_πs = [ComplexityMeasures.permutation_to_integer(p) for p in permutations]
        decoded_πs = decode.(Ref(encoder), encoded_πs)
        @test all(encoded_πs .== 1:factorial(m))
        @test all(decoded_πs .== permutations)
    end
end

@testset "Gaussian symbolization" begin
    # Li et al. (2018) recommends using at least 1000 data points when estimating
    # dispersion entropy.
    x = rand(1000)
    μ = mean(x)
    σ = std(x)
    c = 4
    m = 4
    τ = 1
    s = GaussianCDFEncoding(c = c; μ, σ)

    # Symbols should be in the set [1, 2, …, c].
    symbols = encode.(Ref(s), x)
    @test all([s ∈ collect(1:c) for s in symbols])

    # Test case from Rostaghi & Azami (2016)'s dispersion entropy paper.
    y = [9.0, 8.0, 1.0, 12.0, 5.0, -3.0, 1.5, 8.01, 2.99, 4.0, -1.0, 10.0]
    μ = mean(y)
    σ = std(y)
    encoding = GaussianCDFEncoding( c = 3; μ, σ)
    s = encode.(Ref(encoding), y)
    @test s == [3, 3, 1, 3, 2, 1, 1, 3, 2, 2, 1, 3]
end

@testset "isless_rand" begin
    # because permutations are partially random, we sort many times and check that
    # we get *a* (not *the one*) correct answer every time
    for i = 1:50
        s = sortperm([1, 2, 3, 2], lt = ComplexityMeasures.isless_rand)
        @test s == [1, 2, 4, 3] || s == [1, 4, 2, 3]
    end
end

# @testset "Equal-width intervals (rectangular binning)" begin
#     N = 10 # each dimension is divides [min, max] into N chunks
#     @testset "User-defined grid" begin
#         # For datasets
#         # --------------------------------
#         D = Dataset([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
#         r = (1 - (-1)) / N

#         # (-1, 1) range for all dims.
#         binning_same = FixedRectangularBinning(-1.0, 1.0, N)

#         # (axismins[1], axismaxs[1]) gives the range for the 1st dim.
#         axismins, axismaxs = (-1.0, -1.0), (1.0, 1.0)
#         binning_diff = FixedRectangularBinning(axismins, axismaxs, N)

#         # Verify that grids are as expected.
#         symb_diff = RectangularBinEncoding(D, binning_diff)
#         symb_same = RectangularBinEncoding(D, binning_same)
#         @test all(@. symb_diff.edgelengths ≈ r)
#         @test all(@. symb_same.edgelengths ≈ r)
#         @test all(@. symb_diff.mini ≈ -1.0)
#         @test all(@. symb_same.mini ≈ -1.0)

#         # bins are indexed from 0, so should get the following symbols
#         expected = [SVector(0,0), SVector(4, 4), SVector(9, 9)]
#         @test outcomes(D, symb_diff) ==
#             outcomes(D, symb_same) ==
#             expected

#         @test total_outcomes(symb_diff) == N^2
#         @test total_outcomes(symb_same) == N^2
#         @test total_outcomes(D, symb_diff) == N^2
#         @test total_outcomes(D, symb_same) == N^2

#         # all possible outcomes
#         o = outcome_space(symb_diff)
#         @test length(o) == N^2
#         @test eltype(o) == SVector{2, Int}
#         for i in 1:N
#             for j in 1:N
#                 @test SVector(i, j) ∈ o
#             end
#         end

#         # For univariate timeseries
#         # --------------------------------
#         # bins are indexed from 0, so should get [0, 4, 9] with N = 10
#         x = [-1.0, 0.0, 1.0]
#         symb_1D = RectangularBinEncoding(x, binning_same)

#         @test total_outcomes(symb_1D) == N
#         @test total_outcomes(x, symb_1D) == N
#         @test symb_1D.mini ≈ -1.0
#         @test symb_1D.edgelengths ≈ (1 - (-1)) / N
#         @test outcomes(x, symb_1D) == [0, 4, 9]
#     end

#     @testset "Grid defined by data" begin
#         D = Dataset([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
#         r = nextfloat((1.0 - (-1.0)) / N)
#         binning_int = RectangularBinning(N)
#         binning_intvec = RectangularBinning([N, N])
#         binning_float = RectangularBinning(r)
#         binning_floatvec = RectangularBinning([r, r])

#         symb_int = RectangularBinEncoding(D, binning_int)
#         symb_float = RectangularBinEncoding(D, binning_float)
#         symb_intvec = RectangularBinEncoding(D, binning_intvec)
#         symb_floatvec = RectangularBinEncoding(D, binning_floatvec)

#         @test round.(symb_int.edgelengths, digits = 15) ==
#             round.(symb_float.edgelengths, digits = 15) ==
#             round.(symb_intvec.edgelengths, digits = 15) ==
#             round.(symb_floatvec.edgelengths, digits = 15)
#         @test all(@. symb_int.edgelengths ≈ r)
#         @test all(@. symb_float.edgelengths ≈ r)
#         @test all(@. symb_intvec.edgelengths ≈ r)
#         @test all(@. symb_floatvec.edgelengths ≈ r)
#         @test all(@. symb_int.mini ≈ -1.0)
#         @test all(@. symb_float.mini ≈ -1.0)
#         @test all(@. symb_intvec.mini ≈ -1.0)
#         @test all(@. symb_floatvec.mini ≈ -1.0)

#         @test outcomes(D, symb_int) ==
#             outcomes(D, symb_intvec) ==
#             outcomes(D, symb_float) ==
#             outcomes(D, symb_floatvec) ==
#             [SVector(0,0), SVector(4,4), SVector(9,9)]

#         x = [-1.0, 0.0, 1.0]
#         r = (1.0 - (-1.0)) / N
#         binning_int = RectangularBinning(N)
#         binning_intvec = RectangularBinning([N])
#         symb_int = RectangularBinEncoding(x, binning_int)
#         symb_float = RectangularBinEncoding(x, binning_float)

#         @test all(@. symb_int.edgelengths ≈ r)
#         @test all(@. symb_float.edgelengths ≈ r)
#         @test all(@. symb_int.mini ≈ -1.0)
#         @test all(@. symb_float.mini ≈ -1.0)

#         @test outcomes(x, symb_int) ==
#             outcomes(x, symb_float) ==
#             [0, 4, 9]
#     end

#     @testset "outcome length" begin
#         X = Dataset(rand(10, 3))
#         x = rand(10)
#         rbN = RectangularBinning(5)
#         rbNs = RectangularBinning([5, 3, 4])
#         rbF = RectangularBinning(0.5)
#         rbFs = RectangularBinning([0.5, 0.4, 0.3])

#         symbolization_xN = RectangularBinEncoding(x, rbN)
#         symbolization_xF = RectangularBinEncoding(x, rbF)
#         symbolization_XN = RectangularBinEncoding(X, rbN)
#         symbolization_XNs = RectangularBinEncoding(X, rbNs)
#         symbolization_XF = RectangularBinEncoding(X, rbF)
#         symbolization_XFs = RectangularBinEncoding(X, rbFs)

#         @test total_outcomes(x, symbolization_xN) == 5
#         @test_throws ArgumentError total_outcomes(x, symbolization_xF)
#         @test_throws ArgumentError total_outcomes(x, symbolization_xF)
#         @test_throws ArgumentError total_outcomes(x, symbolization_XNs)

#         @test total_outcomes(X, symbolization_XN) == 5^3
#         @test total_outcomes(X, symbolization_XNs) == 5*3*4
#         @test_throws ArgumentError total_outcomes(X, symbolization_XF)
#         @test_throws ArgumentError total_outcomes(X, symbolization_XFs)
#     end
# end

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
    @test missing_outcomes(SymbolicPermutation(; m, τ), x) == 3

    m, τ = 2, 1
    y = [1, 2, 1, 2] # only two patterns, none missing
    @test missing_outcomes(SymbolicPermutation(; m, τ), x) == 0
end
