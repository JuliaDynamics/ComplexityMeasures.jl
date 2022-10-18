using Entropies, Test

@testset "Complexity measures" begin
    testfile("./reverse_dispersion.jl")
    testfile("./statistical_complexity.jl")
end
