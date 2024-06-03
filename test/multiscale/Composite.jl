using ComplexityMeasures
using Statistics

########################################################################################
# Downsampling. If these are correct, then upstream results are also guaranteed to be
# correct (given the correctness of upstream methods).
########################################################################################
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
c = CompositeDownsampling(f = mean)

# for scale 2, downsampling indices for `x` are 1:2, 3:4, 5:6, 7:8, 9:10, so we should get
# the means of consecutive running pairs
@test ComplexityMeasures.downsample(c, 2, x) == [
    [1.5, 3.5, 5.5, 7.5], # indices 1:2, 3:4, 5:6, 7:8
    [2.5, 4.5, 6.5, 8.5], # indices 2:3, 4:5, 6:7, 8:9
]
@test ComplexityMeasures.downsample(c, 3, x) == [
    [2.0, 5.0], # indices 1:3, 4:6
    [3.0, 6.0], # indices 2:4, 5:7
    [4.0, 7.0], # indices 3:5, 6:8
]

@test ComplexityMeasures.downsample(c, 4, x) == [
    [2.5], # indices 1:4
    [3.5], # indices 2:5
    [4.5], # indices 3:6
    [5.5], # indices 4:7
]

@test ComplexityMeasures.downsample(c, 5, x) == [
    [3.0], # indices 1:5
    [4.0], # indices 2:6
    [5.0], # indices 3:7
    [6.0], # indices 4:8
    [7.0], # indices 5:9
]

@test_throws DomainError ComplexityMeasures.downsample(c, 6, x)


##############################################################
# API
##############################################################
x = rand(100)
maxscale = 5

# Discrete info measures
hest = Shannon()
o = Dispersion()
mc = multiscale(c, hest, o, x; maxscale)
mcn = multiscale_normalized(c, hest, o, x; maxscale)
@test mc isa Vector{T} where T <: Real
@test mcn isa Vector{T} where T <: Real
@test length(mc) == 5
@test length(mcn) == 5

# `DifferentialInfoEstimator`s` should work for `multiscale`, but not `multiscale_normalized`
@test multiscale(c, Kraskov(hest), x) isa Vector{T} where T <: Real
@test_throws MethodError multiscale_normalized(c, Kraskov(hest), x)

# `ComplexityEstimator`s
@test multiscale(c, SampleEntropy(x), x) isa Vector
@test multiscale_normalized(c, SampleEntropy(x), x) isa Vector
