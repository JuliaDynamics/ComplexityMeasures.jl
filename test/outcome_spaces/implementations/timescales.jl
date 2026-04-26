using ComplexityMeasures, Test
using Random
rng = Random.MersenneTwister(1234)

@testset "Timescales" begin
    N = 200
    a = 10
    t = LinRange(0, 2 * a * π, N)
    x = sin.(t .+ cos.(t / 0.1)) .- 0.1

    @testset "WaveletOverlap" begin
        # Only works for timeseries inputs
        @test_throws ArgumentError probabilities(WaveletOverlap(), 2)

        wl = ComplexityMeasures.Wavelets.WT.Daubechies{4}()
        o = WaveletOverlap(wl)
        ps = probabilities(o, x)
        @test length(ps) == 7
        @test ps isa Probabilities
        @test information(Renyi(q = 1, base = 2), WaveletOverlap(), x) isa Real
        @test issorted(outcome_space(WaveletOverlap(), x))
    end

    @testset "Fourier Spectrum" begin
        # Only works for timeseries inputs
        @test_throws ArgumentError probabilities(PowerSpectrum(), 2)

        N = 1000
        t = range(0, 10π, N)
        x = sin.(t)
        y = @. sin(t) + sin(sqrt(3) * t)
        z = randn(N)
        o = PowerSpectrum()
        ents = [information(Renyi(), o, w) for w in (x, y, z)]
        @test ents[1] < ents[2] < ents[3]
        # Test event stuff (analytically, using sine wave)
        probs, outs = probabilities_and_outcomes(o, x)
        @test length(outs) == length(probs) == 501
        @test outs[1] ≈ 0 atol = 1.0e-16 # 0 frequency, i.e., mean value
        @test probs[1] ≈ 0 atol = 1.0e-16  # sine wave has 0 mean value
        @test outs[end] == 0.5 # Nyquist frequency, 1/2 the sampling rate (Which is 1)
        @test issorted(outcome_space(o, x))

        x = cos.(range(0, 2π; length = 10000)) .+ 1.0e-2 .* randn(rng, 10000)
        o = PowerSpectrum(δ = 0.1)
        p, outs = probabilities_and_outcomes(o, x)
        @test sum(p .> 0.0) == 1
        @test length(outs) == length(p) == 5001
        @test outs[1] ≈ 0 atol = 1.0e-16
        @test p[1] ≈ 0 atol = 1.0e-16
        o = PowerSpectrum(10.5, true)
        p, ~ = probabilities_and_outcomes(o, x)
        @test sum(p .> 0.0) == 1
        @test length(outs) == length(p) == 5001
        @test outs[1] ≈ 0 atol = 1.0e-16
        @test p[1] ≈ 0 atol = 1.0e-16
    end
end
