using ComplexityMeasures
using Test
using Random

Random.seed!(1234)

@testset "SequentialSlopes" begin

    @testset "Constructors" begin
        # Unsorted thresholds must error
        @test_throws ArgumentError SequentialSlopes(2, [1.0, -1.0, 0.0])
        @test_throws ArgumentError SequentialSlopes(2, [0.0, 0.0, 1.0])  # not strictly sorted either
        # γ smaller than δ is an invalid (unsorted) threshold configuration
        @test_throws ArgumentError SequentialSlopes(2; γ = 0.0001, δ = 1.0)
    end

    @testset "outcome_space / total_outcomes" begin
        o1 = SequentialSlopes(1)
        @test outcome_space(o1) == [(s,) for s in -2:2]
        @test total_outcomes(o1) == 5
        @test total_outcomes(o1) == length(outcome_space(o1))

        o2 = SequentialSlopes(2)
        Ω = outcome_space(o2)
        @test length(Ω) == 25 == total_outcomes(o2)
        @test issorted(Ω)
        @test allunique(Ω)
        @test all(w -> all(s -> s in -2:2, w), Ω)

        # total_outcomes scales as (nsymbols)^m
        o3 = SequentialSlopes(4)
        @test total_outcomes(o3) == 5^4 == length(outcome_space(o3))

        # General (6-symbol) alphabet built from a 5-element threshold vector
        o_gen = SequentialSlopes(1, [-2.0, -1.0, 0.0, 1.0, 2.0])
        @test outcome_space(o_gen) == [(s,) for s in -2:3]
        @test total_outcomes(o_gen) == 6
    end

    @testset "codify" begin
        o = SequentialSlopes(2; γ = 0.5, δ = 0.001)

        # Constant series -> all differences are 0 -> all symbols 0
        xconst = fill(3.0, 10)
        @test codify(o, xconst) == zeros(Int, 9)

        # length(codify(o, x)) == length(x) - 1, for any x
        x = cumsum(randn(50))
        @test length(codify(o, x)) == length(x) - 1

        # Independently hand-checked example
        # diffs = [1.0, 0.4, -1.001, -2.399]
        xhand = [0.0, 1.0, 1.4, 0.399, -2.0]
        @test codify(o, xhand) == [2, 1, -2, -2]
    end

    @testset "analytic outcomes" begin
        # x alternates +1/-1 differences forever, so with γ=0.5 (δ default 0.001)
        # the symbol sequence alternates 2, -2, 2, -2, ... and m=2 words alternate
        # between exactly two words: (2, -2) and (-2, 2).
        x = repeat([1, 2], 1000)
        o = SequentialSlopes(2; γ = 0.5)

        outs = outcomes(o, x)
        @test length(outs) == 2
        @test Set(Tuple.(outs)) == Set([(2, -2), (-2, 2)])

        cts, outs2 = counts_and_outcomes(o, x)
        @test Tuple.(outs2) == Tuple.(outs)
        @test sum(cts) == length(x) - o.m          # 1998 total words
        @test all(==(999), collect(cts))            # perfectly balanced alternation

        # Two perfectly balanced outcomes => exactly 1 bit of Shannon entropy
        @test entropy(Shannon(; base = 2), o, x) ≈ 1.0
    end

    @testset "counts_and_outcomes matches a manual tally" begin
        x = [1.0, 1.2, 1.19, 3.5, 3.6, -10.0, -9.9, -9.89, -9.89, 0.0]
        o = SequentialSlopes(2; γ = 0.5, δ = 0.01)

        symbols = codify(o, x)
        m = o.m
        manual = Dict{NTuple{m,Int},Int}()
        for i in 1:(length(symbols) - m + 1)
            w = ntuple(j -> symbols[i + j - 1], m)
            manual[w] = get(manual, w, 0) + 1
        end

        cts, outs = counts_and_outcomes(o, x)
        computed = Dict(zip(Tuple.(outs), collect(cts)))

        @test computed == manual
        @test sum(values(manual)) == length(x) - m
    end

    @testset "edge cases" begin
        o = SequentialSlopes(2)

        # Minimal length: exactly m+1 points -> exactly one word
        xmin = [1.0, 2.0, 3.0]
        @test length(codify(o, xmin)) == 2
        _, outs = counts_and_outcomes(o, xmin)
        @test length(outs) == 1

        # Too short to form even one word must error
        @test_throws Exception counts_and_outcomes(o, [1.0])
        @test_throws Exception counts_and_outcomes(o, [1.0, 2.0])

        # m = 1: words are just the symbols themselves
        o1 = SequentialSlopes(1; γ = 0.5, δ = 0.001)
        x = [0.0, 1.0, 1.0, -1.0, 0.0]  # diffs: 1, 0, -2, 1 -> symbols: 2, 0, -2, 2
        @test codify(o1, x) == [2, 0, -2, 2]
        cts, outs = counts_and_outcomes(o1, x)
        computed = Dict(zip(Tuple.(outs), collect(cts)))
        @test computed == Dict((2,) => 2, (0,) => 1, (-2,) => 1)
    end

end