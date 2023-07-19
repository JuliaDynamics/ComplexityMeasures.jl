using ComplexityMeasures, Test

@testset "Timescales" begin
    N = 200
    a = 10
    t = LinRange(0, 2*a*π, N)
    x = sin.(t .+  cos.(t/0.1)) .- 0.1;

    @testset "WaveletOverlap" begin
        wl = ComplexityMeasures.Wavelets.WT.Daubechies{4}()
        est = WaveletOverlap(wl)
        ps = probabilities(est, x)
        @test length(ps) == 8
        @test ps isa Probabilities
        @test information(Renyi(q = 1, base = 2), WaveletOverlap(), x) isa Real
        @test issorted(outcome_space(WaveletOverlap(), x))
    end

    @testset "Fourier Spectrum" begin
        N = 1000
        t = range(0, 10π, N)
        x = sin.(t)
        y = @. sin(t) + sin(sqrt(3)*t)
        z = randn(N)
        est = PowerSpectrum()
        ents = [information(Renyi(), est, w) for w in (x,y,z)]
        @test ents[1] < ents[2] < ents[3]
        # Test event stuff (analytically, using sine wave)
        probs, outs = probabilities_and_outcomes(est, x)
        @test length(outs) == length(probs) == 501
        @test outs[1] ≈ 0 atol=1e-16 # 0 frequency, i.e., mean value
        @test probs[1] ≈ 0 atol=1e-16  # sine wave has 0 mean value
        @test outs[end] == 0.5 # Nyquist frequency, 1/2 the sampling rate (Which is 1)
        @test issorted(outcome_space(est, x))
    end
end
