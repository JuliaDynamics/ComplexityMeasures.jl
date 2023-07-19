using ComplexityMeasures

est = Dispersion()
x = rand(100)
maxscale = 5
e = Shannon()

# Generic tests is all we need here. The tests that make sure that the entropies are
# computed for the correctly sampled timeseries are in `/test/multiscale/downsampling.jl`
mr = ComplexityMeasures.multiscale(Regular(), est, x; maxscale)
mrn = ComplexityMeasures.multiscale_normalized(Regular(), est, x; maxscale)
@test mr isa Vector{T} where T <: Real
@test mrn isa Vector{T} where T <: Real
@test length(mr) == 5
@test length(mrn) == 5

# `DiffInfoMeasureEst`s` should work for `multiscale`, but not `multiscale_normalized`
@test ComplexityMeasures.multiscale(Regular(), e, Kraskov(), x) isa Vector{T} where T <: Real
@test_throws ErrorException ComplexityMeasures.multiscale_normalized(Regular(), e, Kraskov(), x)
