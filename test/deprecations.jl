using ComplexityMeasures, Test
using Random
rng = Xoshiro(1234)

@testset "2.0 deprecations" begin
    x = randn(rng, 1000)

    # Convenience
    @test permentropy(x) == entropy_permutation(x; base=MathConstants.e)
    msg ="`permentropy(x; τ, m, base)` is deprecated.\nUse instead: `entropy_permutation(x; τ, m, base)`, or even better, use the\ndirect syntax discussed in the docstring of `entropy_permutation`.\n"
    @test_logs (:warn, msg) permentropy(x)

    # Generalized entropies
    @test genentropy(x, 0.1) == information(Shannon(MathConstants.e), ValueBinning(0.1), x)
    msg = "`genentropy(probs::Probabilities; q, base)` deprecated.\nUse instead: `information(Renyi(q, base), probs)`.\n"
    @test_logs (:warn, msg) genentropy(Probabilities(rand(rng, 3)))

    msg = "`genentropy(x::Array_or_SSSet, o::OutcomeSpace; q, base)` is deprecated.\nUse instead: `information(Renyi(q, base), est, x)`.\n"
    @test_logs (:warn, msg) genentropy(x, ValueBinning(0.1))

    @test probabilities(x, 0.1) == probabilities(ValueBinning(0.1), x)
    msg = "`probabilities(x, est::OutcomeSpace)`\nis deprecated, use `probabilities(est::OutcomeSpace, x) instead`.\n"
    @test_logs (:warn, msg) probabilities(x, ValueBinning(0.1))

    x = StateSpaceSet(rand(rng, 100, 3))
    @test genentropy(x, 4) == information(Shannon(MathConstants.e), ValueBinning(4), x)
end

@testset "deprecations: binning" begin
    @test FixedRectangularBinning((0, 1), (0, 1), 2) isa FixedRectangularBinning
    @test FixedRectangularBinning(0.1, 0.9, 2) isa FixedRectangularBinning
    @test FixedRectangularBinning(0.1, 0.9, 0.2) isa FixedRectangularBinning
end


@testset "3.0 deprecations" begin
    x = StateSpaceSet(rand(rng, 100, 3))

    @test entropy_maximum(Shannon(MathConstants.e), ValueBinning(4), x) ==
        information_maximum(Shannon(MathConstants.e), ValueBinning(4), x)

    @test entropy_normalized(Shannon(MathConstants.e), ValueBinning(4), x) ==
        information_normalized(Shannon(MathConstants.e), ValueBinning(4), x)
    
    @testset "Ordinal pattern constructiors" begin
        @test SymbolicPermutation() isa OrdinalPatterns
        @test SymbolicWeightedPermutation() isa WeightedOrdinalPatterns
        @test SymbolicAmplitudeAwarePermutation() isa AmplitudeAwareOrdinalPatterns

        @testset "OrdinalPatterns" begin 
            msg = "Keyword argument `τ` to `OrdinalPatterns` is deprecated. " *
            "The signature is now " * 
            "`OrdinalPatterns{m}(τ = 1, lt::Function = ComplexityMeasures.isless_rand)`" * 
            ", so provide `τ` as a positional argument instead. " * 
            "In this call, the given keyword `τ` is used instead of the positional `τ`."
            τ = 1; 
            @test_logs (:warn, msg) OrdinalPatterns{3}(τ + 1; τ)

            msg = "Keyword argument `lt` to `OrdinalPatterns` is deprecated. " *
            "The signature is now " * 
            "`OrdinalPatterns{m}(τ = 1, lt::Function = ComplexityMeasures.isless_rand)`" * 
            ", so provide `lt` as a positional argument instead. "  * 
            "In this call, the given keyword `lt` is used instead of the positional `lt`."
            lt = Base.isless;
            @test_logs (:warn, msg) OrdinalPatterns{3}(τ, lt; lt)

            # Test that keyword argument is preferred over positional argument
            o = OrdinalPatterns{3}(5; τ = 2)
            @test o.τ == 2
            lt = Base.isless;
            ltr = ComplexityMeasures.isless_rand
            o = OrdinalPatterns{3}(2, lt; lt = ltr)
            @test o.encoding.lt == ltr
        end

        @testset "WeightedOrdinalPatterns" begin 
            msg = "Keyword argument `τ` to `WeightedOrdinalPatterns` is deprecated. " *
            "The signature is now " * 
            "`WeightedOrdinalPatterns{m}(τ::Int = 1, lt::F=ComplexityMeasures.isless_rand)`" * 
            ", so provide `τ` as a positional argument instead. "  * 
            "In this call, the given keyword `τ` is used instead of the positional `τ`."
            τ = 1; 
            @test_logs (:warn, msg) WeightedOrdinalPatterns{3}(τ + 1; τ)

            msg = "Keyword argument `lt` to `WeightedOrdinalPatterns` is deprecated. " *
            "The signature is now " * 
            "`WeightedOrdinalPatterns{m}(τ = 1, lt::Function = ComplexityMeasures.isless_rand)`" * 
            ", so provide `lt` as a positional argument instead. "  * 
            "In this call, the given keyword `lt` is used instead of the positional `lt`."
            lt = Base.isless;
            @test_logs (:warn, msg) WeightedOrdinalPatterns{3}(τ, lt; lt)

            # Test that keyword argument is preferred over positional argument
            o = WeightedOrdinalPatterns{3}(2; τ)
            @test o.τ == WeightedOrdinalPatterns{3}(τ).τ
            lt = Base.isless;
            ltr = ComplexityMeasures.isless_rand
            @test WeightedOrdinalPatterns{3}(2, lt; lt = ltr).encoding.lt == ltr
        end

        @testset "AmplitudeAwareOrdinalPatterns" begin 
            msg = "Keyword argument `τ` to `AmplitudeAwareOrdinalPatterns` is deprecated. " *
            "The signature is now " * 
            "`AmplitudeAwareOrdinalPatterns{m}(τ::Int = 1, A = 0.5, lt::F=isless_rand)`" * 
            ", so provide `τ` as a positional argument instead. "  * 
            "In this call, the given keyword `τ` is used instead of the positional `τ`."
            τ = 1; 
            @test_logs (:warn, msg) AmplitudeAwareOrdinalPatterns{3}(τ + 1; τ)

            msg = "Keyword argument `lt` to `AmplitudeAwareOrdinalPatterns` is deprecated. " *
            "The signature is now " * 
            "`AmplitudeAwareOrdinalPatterns{m}(τ::Int = 1, A = 0.5, lt::F=isless_rand)`" * 
            ", so provide `lt` as a positional argument instead. "  * 
            "In this call, the given keyword `lt` is used instead of the positional `lt`."
            lt = Base.isless;
            @test_logs (:warn, msg) AmplitudeAwareOrdinalPatterns{3}(τ, 0.5, lt; lt)
            @test_throws ArgumentError AmplitudeAwareOrdinalPatterns{3}(τ, lt; lt)

            msg = "Keyword argument `A` to `AmplitudeAwareOrdinalPatterns` is deprecated. " *
            "The signature is now " * 
            "`AmplitudeAwareOrdinalPatterns{m}(τ::Int = 1, A = 0.5, lt::F=isless_rand)`" * 
            ", so provide `A` as a positional argument instead. "  * 
            "In this call, the given keyword `A` is used instead of the positional `A`."
            A = 0.5;
            @test_logs (:warn, msg) AmplitudeAwareOrdinalPatterns{3}(τ, A; A)

            # Test that keyword argument is preferred over positional argument
            o = AmplitudeAwareOrdinalPatterns{3}(2; τ = 5)
            @test o.τ == 5
            lt = Base.isless;
            ltr = ComplexityMeasures.isless_rand
            o = AmplitudeAwareOrdinalPatterns{3}(2, 0.5, Base.isless; lt = ltr)
            @test o.encoding.lt == ltr
            o = AmplitudeAwareOrdinalPatterns{3}(2; τ = 5)
            @test o.τ == 5
            o = AmplitudeAwareOrdinalPatterns{3}(2, 0.5; A = 0.9)
            @test o.A == 0.9
        end
    end

    for f in (OrdinalPatterns, WeightedOrdinalPatterns, AmplitudeAwareOrdinalPatterns)
        a = f(; m = 3, τ = 2)
        b = f{3}(2)
        @test entropy(a, x) == entropy(b, x)
    end

    # `allcounts` and `allprobabilities` are redundant, because they always need to 
    # compute all outcomes anyway.
    x = rand(rng, 1:20, 19)
    o = UniqueElements()

    sc = "`allcounts` is deprecated. Use `allcounts_and_outcomes` instead."
    @test_logs (:warn, sc) allcounts(o, x)
    @test allcounts(o, x) == first(allcounts_and_outcomes(o, x))

    est = AddConstant()
    sp = "`allprobabilities` is deprecated. Use `allprobabilities_and_outcomes` instead."
    @test_logs (:warn, sp) allprobabilities(o, x)
    @test allprobabilities(o, x) == first(allprobabilities_and_outcomes(o, x))
    @test allprobabilities(est, o, x) == first(allprobabilities_and_outcomes(est, o, x))
end
