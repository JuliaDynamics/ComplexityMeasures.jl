using ComplexityMeasures, Test
x = rand(1000)
@test_throws MethodError entropy(x, 0.1)
est = AlizadehArghami() # the AlizadehArghami estimator only works for Shannon entropy
@test_throws MethodError entropy(Tsallis(), AlizadehArghami(), x)
