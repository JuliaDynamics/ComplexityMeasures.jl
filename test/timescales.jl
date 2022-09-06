using Entropies, Test

@testset "Timescales" begin
    N = 200
    a = 10
    t = LinRange(0, 2*a*π, N)
    x = sin.(t .+  cos.(t/0.1)) .- 0.1;

    @testset "WaveletOverlap" begin
        wl = Entropies.Wavelets.WT.Daubechies{4}()
        est = WaveletOverlap(wl)
        ps = probabilities(x, est)
        @test length(ps) == 8
        @test ps isa Probabilities
        @test genentropy(x, WaveletOverlap(), q = 1, base = 2) isa Real
    end
end