x = ones(3)
@test_throws MethodError entropy(x, 0.1)

@testset "Non-default entropy types" begin
    x = rand(1000)
    est = AlizadehArghami() # the AlizadehArghami estimator only works for Shannon entropy
    @test_throws ArgumentError entropy(Tsallis(), AlizadehArghami(), x)
end
