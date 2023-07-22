struct SomeNewComplexityEstimator <: ComplexityEstimator end
x = rand(100)
@test_throws ArgumentError complexity(SomeNewComplexityEstimator(), x)
