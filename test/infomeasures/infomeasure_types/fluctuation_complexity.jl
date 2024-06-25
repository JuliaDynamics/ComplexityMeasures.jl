# Examples from https://en.wikipedia.org/wiki/Information_fluctuation_complexity
# for the Shannon fluctuation complexity.
@testset "fluctuation_complexity" begin
    probs = Probabilities([2//17, 2//17, 1//34, 5//34, 2//17, 2//17, 2//17, 4//17])
    def = Shannon(base = 2)
    c = FluctuationComplexity(definition = def, base = 2)
    @test round(information(c, probs), digits = 2) ≈ 0.56 
    # Zero both for uniform and single-element PMFs.
    @test information(c, Probabilities([0.2, 0.2, 0.2, 0.2, 0.2])) == 0.0
    @test information(c, Probabilities([1.0, 0.0])) == 0.0
end