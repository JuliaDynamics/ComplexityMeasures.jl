using Entropies, Test

@testset "Complexity measures" begin
    testfile("./reverse_dispersion.jl")
    testfile("./missing_dispersion.jl")
    testfile("./approx_entropy.jl")
    testfile("./sample_entropy.jl")
end
