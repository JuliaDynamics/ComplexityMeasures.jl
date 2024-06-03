using ComplexityMeasures
using Statistics

########################################################################################
# Downsampling. If these are correct, then upstream results are also guaranteed to be
# correct (given the correctness of upstream methods).
########################################################################################
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

r = RegularDownsampling(f = mean)

# just returns original time series
@test ComplexityMeasures.downsample(r, 1, x) == x

# for scale 2, downsampling indices for `x` are 1:2, 3:4, 5:6, 7:8, 9:10, so we should get
# the means of consecutive running pairs
@test ComplexityMeasures.downsample(r, 2, x) == [1.5, 3.5, 5.5, 7.5, 9.5]

# for scale 3, downsampling indices for `x` are 1:3, 4:6, 7:9. Take the means:
@test ComplexityMeasures.downsample(r, 3, x) == [2.0, 5.0, 8.0]

# for scale 4, downsampling indices for `x` are 1:4, 5:8. Take the means:
@test ComplexityMeasures.downsample(r, 4, x) == [2.5, 6.5]

# for scale 5, downsampling indices for `x` are 1:5, 6:10. Take the means:
@test ComplexityMeasures.downsample(r, 5, x) == [3.0, 8.0]

@test_throws DomainError ComplexityMeasures.downsample(r, 6, x)


##############################################################
# API
##############################################################
x = rand(100)
r = RegularDownsampling()
hest = Shannon()
o = Dispersion()
maxscale = 5

# Discrete information measures
mr = multiscale(r, hest, o, x; maxscale)
mrn = multiscale_normalized(r, hest, o, x; maxscale)
@test mr isa Vector{T} where T <: Real
@test mrn isa Vector{T} where T <: Real
@test length(mr) == 5
@test length(mrn) == 5

# `DifferentialInfoEstimator`s` should work for `multiscale`, but not `multiscale_normalized`
@test multiscale(r, Kraskov(hest), x) isa Vector{T} where T <: Real
@test_throws MethodError multiscale_normalized(r, Kraskov(hest), x)

# `ComplexityEstimator`s
@test multiscale(r, SampleEntropy(x), x) isa Vector
@test multiscale_normalized(r, SampleEntropy(x), x) isa Vector
