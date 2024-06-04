using ComplexityMeasures
using Statistics

########################################################################################
# Constructors
########################################################################################
# It should be possible to provide arbitrary scales
r_arb = RegularDownsampling(; f = mean, scales = [2, 3, 6])
@test r_arb isa RegularDownsampling

# Providing integer scale `n` should be equivalent to scales 1:n
r_int = RegularDownsampling(; f = mean, scales = 5)
r_range = RegularDownsampling(; f = mean, scales = 1:5)
@test collect(r_int.scales) == collect(r_range.scales)


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

x = randn(500)
r = RegularDownsampling(; scales = 5)
hest = Shannon()
o = Dispersion()

# Discrete information measures
mr = multiscale(r, hest, o, x)
mrn = multiscale_normalized(r, hest, o, x)
@test mr isa Vector{T} where T <: Real
@test mrn isa Vector{T} where T <: Real
@test length(mr) == 5
@test length(mrn) == 5

# `DifferentialInfoEstimator`s` should work for `multiscale`, but not `multiscale_normalized`
@test multiscale(r, Kraskov(hest), x) isa Vector{T} where T <: Real
@test_throws MethodError multiscale_normalized(r, Kraskov(hest), x)

# `ComplexityEstimator`s
cest = SampleEntropy(x)
@test multiscale(r, cest, x) isa Vector
@test multiscale_normalized(r, cest, x) isa Vector

# any type of scale input should work
r_arb = RegularDownsampling(; f = mean, scales = [1, 2, 3, 4, 5])
r_int = RegularDownsampling(; f = mean, scales = 5)
r_range = RegularDownsampling(; f = mean, scales = 1:5)

@test multiscale(r_arb, cest, x) == multiscale(r_int, cest, x) == multiscale(r_range, cest, x)
