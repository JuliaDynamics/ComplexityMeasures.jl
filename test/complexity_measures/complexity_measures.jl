using Entropies, Test

@testset "Complexity measures" begin
    testfile("./reverse_dispersion.jl")
    testfile("./sample_entropy.jl")
end
