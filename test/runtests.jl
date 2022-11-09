using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
@testset "Entropies.jl" begin
    # Probability estimators
    testfile("probabilities/count_occurrences.jl")
    testfile("probabilities/visitation_frequency.jl")
    testfile("probabilities/transfer_operator.jl")
    testfile("probabilities/naive_kernel.jl")
    testfile("probabilities/permutation.jl")
    testfile("probabilities/permutation_weighted.jl")
    testfile("probabilities/permutation_amplitude_aware.jl")
    testfile("probabilities/permutation_spatial.jl")
    testfile("probabilities/timescales.jl")
    testfile("probabilities/dispersion.jl")
    testfile("probabilities/diversity.jl")

    # Different entropies
    testfile("entropies/renyi.jl")
    testfile("entropies/shannon.jl")
    testfile("entropies/tsallis.jl")
    testfile("entropies/curado.jl")
    testfile("entropies/stretched_exponential.jl")

    # Entropy estimators
    testfile("entropies/estimators/estimators.jl")

    # Various
    testfile("utils/utils.jl")
    testfile("entropies/convenience.jl")
end
