using ComplexityMeasures, Test
using Statistics: mean

@testset "interface" begin
    x = ones(3)
    p = probabilities(x)
    @test p isa Probabilities
    @test p == [1]
    @test mean(p) == 1
end

@testset "Histogram" begin
    # Okay, with these inputs we know exactly what the final scenario
    # should be. Bins 1 2 and 3 have one entry, bin 4 has zero, bin 5
    # has two entries, bins 6 7 8 have zero, bin 9 has 1 and bin 10 zero
    x = [0.05, 0.15, 0.25, 0.45, 0.46, 0.85]
    est = ValueHistogram(FixedRectangularBinning((0:0.1:1,)))
    correct = [1, 1, 1, 0, 2, 0, 0, 0, 1, 0]
    correct = correct ./ sum(correct)

    miss = missing_outcomes(est, x)
    @test miss == 5

    p = allprobabilities(est, x)

    @test p == correct
end

@testset "Ordinal" begin
    # here it is guaranteed that all outcomes are the 123 permutation
    # except the final one that is the 312. That permutation is
    # fourth in the order of the ordinal patterns for m=3
    x = [1, 2, 3, 4, 5, 6, 1]
    est = SymbolicPermutation(m = 3)
    probs, outs = probabilities_and_outcomes(est, x)
    correct = [4, 0, 0, 0, 1, 0]
    correct = correct ./ sum(correct)
    p = allprobabilities(est, x)
    @test p == correct

    miss = missing_outcomes(est, x)
    @test miss == 4

    # Make sure that different outcome would get different allprobs
    x = [1, 2, 3, 4, 5, 6, 5.5]
    probs, outs = probabilities_and_outcomes(est, x)
    correct = [4, 1, 0, 0, 0, 0]
    correct = correct ./ sum(correct)
    p = allprobabilities(est, x)
    @test p == correct
end