using Test
using Entropies
# TODO: None of these packages should be used here. Instead, the files they are needed
using Entropies.DelayEmbeddings
using Entropies.DelayEmbeddings.StaticArrays

defaultname(file) = splitext(basename(file))[1]
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
@testset "Entropies.jl" begin
    # Different generalized entropies
    testfile("entropies/renyi.jl")
    testfile("entropies/shannon.jl")
    testfile("entropies/tsallis.jl")

    # Probability and entropy estimators
    testfile("estimators/counting_based.jl")
    testfile("estimators/timescales.jl")
    testfile("estimators/dispersion.jl")
    testfile("estimators/naive_kernel.jl")
    testfile("estimators/permutation.jl")
    testfile("estimators/permutation_weighted.jl")
    testfile("estimators/permutation_amplitude_aware.jl")
    testfile("estimators/permutation_spatial.jl")
    testfile("estimators/visitation_frequency.jl")
    testfile("estimators/transfer_operator.jl")
    testfile("estimators/nn.jl", "nearest neighbors direct")

    # Various
    testfile("complexity_measures/complexity_measures.jl")
    testfile("utils/utils.jl")
    testfile("entropies/convenience.jl")
end
