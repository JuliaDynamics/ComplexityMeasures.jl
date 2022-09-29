@testset "Histogram estimation" begin
    x = rand(1:10, 100)
    D = Dataset([rand(1:10, 3) for i = 1:100])
    D2 = [(rand(1:10), rand(1:10, rand(1:10)) for i = 1:100)]
    @test Entropies.fasthist(x) isa Probabilities
    @test Entropies.fasthist(D) isa Probabilities
    @test Entropies.fasthist(D2) isa Probabilities

    @test Entropies.fasthist(x) |> sum ≈ 1.0
    @test Entropies.fasthist(D) |> sum ≈ 1.0
    @test Entropies.fasthist(D2)|> sum ≈ 1.0
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
