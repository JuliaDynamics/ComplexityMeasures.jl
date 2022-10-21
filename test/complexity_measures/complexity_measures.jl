using Entropies, Test

@testset "Complexity measures" begin
    testfile("./reverse_dispersion.jl")
    testfile("./fuzzy_entropy.jl")
end
