using ComplexityMeasures

est = Dispersion()
x = rand(100)
maxscale = 5
e = Shannon()

# Generic tests is all we need here. The tests that make sure that the entropies are
# computed for the correctly sampled timeseries are in `/test/multiscale/downsampling.jl`
mc = ComplexityMeasures.multiscale(Composite(), e, est, x; maxscale)
mcn = ComplexityMeasures.multiscale_normalized(Composite(), e, est, x; maxscale)
@test mc isa Vector{T} where T <: Real
@test mcn isa Vector{T} where T <: Real
@test length(mc) == 5
@test length(mcn) == 5

# `DiffInfoMeasureEst`s` should work for `multiscale`, but not `multiscale_normalized`
@test ComplexityMeasures.multiscale(Composite(), e, Kraskov(), x) isa Vector{T} where T <: Real
@test_throws ErrorException ComplexityMeasures.multiscale_normalized(Composite(), e, Kraskov(), x)
