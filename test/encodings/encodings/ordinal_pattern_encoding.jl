using Test
using ComplexityMeasures
using StaticArrays: SVector

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
    @test missing_outcomes(OrdinalPatterns(; m, τ), x) == 3

    m, τ = 2, 1
    y = [1, 2, 1, 2] # only two patterns, none missing
    @test missing_outcomes(OrdinalPatterns(; m, τ), x) == 0
end

@testset "Pretty printing" begin 
    o = OrdinalPatterns{3}()
    @test occursin("OrdinalPatterns{3}", repr(o))
    @test occursin("encoding = OrdinalPatternEncoding(perm = [0, 0, 0], lt = isless_rand), τ = 1", repr(o))

    os = [o, o, o]
    s = "[OrdinalPatterns{3}(encoding = OrdinalPatternEncoding(perm = [0, 0, 0], lt = isless_rand), τ = 1), OrdinalPatterns{3}(encoding = OrdinalPatternEncoding(perm = [0, 0, 0], lt = isless_rand), τ = 1), OrdinalPatterns{3}(encoding = OrdinalPatternEncoding(perm = [0, 0, 0], lt = isless_rand), τ = 1)]"
    @test occursin(s, repr(os))
end
