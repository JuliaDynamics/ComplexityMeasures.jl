using Test
# TODO: None of these packages should be used here. Instead, the files they are needed
using Entropies
using Entropies.DelayEmbeddings
using Entropies.DelayEmbeddings.StaticArrays

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
    testfile("estimators/permutation_spatial.jl")
    testfile("estimators/timescales.jl")
    testfile("estimators/dispersion.jl")
    # Different entropies
    testfile("entropies/renyi.jl")
    testfile("entropies/shannon.jl")
    testfile("entropies/tsallis.jl")
    testfile("entropies/curado.jl")
    testfile("entropies/stretched_exponential.jl")

    testfile("entropies/nearest_neighbors_direct.jl")

    # Multiscale analysis
    testfile("multiscale/multiscale_entropy.jl")

    # Various
    testfile("complexity_measures/complexity_measures.jl")
    testfile("utils/utils.jl")
    testfile("entropies/convenience.jl")
end
