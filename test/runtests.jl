using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
@testset "Entropies.jl" begin
    # Probability estimators
    testfile("estimators/count_occurrences.jl")
    testfile("estimators/visitation_frequency.jl")
    testfile("estimators/transfer_operator.jl")
    testfile("estimators/naive_kernel.jl")
    testfile("estimators/permutation.jl")
    testfile("estimators/permutation_weighted.jl")
    testfile("estimators/permutation_amplitude_aware.jl")
    testfile("estimators/timescales.jl")
    testfile("estimators/dispersion.jl")
    testfile("estimators/diversity.jl")

    testfile("estimators/spatial/permutation_spatial.jl")
    testfile("estimators/spatial/dispersion_spatial.jl")

    # Different entropies
    testfile("entropies/renyi.jl")
    testfile("entropies/shannon.jl")
    testfile("entropies/tsallis.jl")
    testfile("entropies/curado.jl")
    testfile("entropies/stretched_exponential.jl")

    testfile("entropies/nearest_neighbors_direct.jl")

    # Multiscale analysis
    testfile("multiscale/downsampling.jl")
    testfile("multiscale/multiscale.jl")

    # Various
    testfile("utils/utils.jl")
    testfile("entropies/convenience.jl")
end
