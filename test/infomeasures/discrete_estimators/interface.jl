using Random
rng = Random.MersenneTwister(1234)

probests = [
    Dispersion(),
    OrdinalPatterns(),
    AmplitudeAwareOrdinalPatterns(),
    WeightedOrdinalPatterns(),
    WaveletOverlap(),
    PowerSpectrum(),
    ValueHistogram(RectangularBinning(3)),
    Diversity()
]
h = Shannon()
hests = [
    ChaoShen(h),
    HorvitzThompson(h),
    MillerMadow(h),
    Schürmann(h),
    GeneralizedSchürmann(h),
    Jackknife(h),
]

x = rand(rng, 300)
for pest in probests
    hplugin = information(PlugIn(Shannon()), pest, x)

    for hest in hests
        h = information(hest, pest, x)
        @test h isa Real
        @test h >= 0
        @test h > hplugin || abs(h - hplugin) < 1e-5
    end
end
