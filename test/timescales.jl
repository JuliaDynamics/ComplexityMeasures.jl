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
        @test entropy_renyi(x, WaveletOverlap(), q = 1, base = 2) isa Real
    end

    @testet "Fourier Spectrum" begin
        N = 1000
        t = range(0, 10π, N)
        x = sin.(t)
        y = @. sin(t) + sin(sqrt(3)*t)
        z = randn(N)
        est = PowerSpectrum()
        ents = [entropy_renyi(w, est) for w in (x,y,z)]
        @test ents[1] < ents[2] < ents[3]
    end
end
