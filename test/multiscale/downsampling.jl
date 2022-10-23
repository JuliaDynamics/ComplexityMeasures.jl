using Statistics
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

##############################################################
# Regular downsampling
##############################################################
r = Regular(f = mean)

# just returns original time series
@test downsample(r, x, 1) == x

# for scale 2, downsampling indices for `x` are 1:2, 3:4, 5:6, 7:8, 9:10, so we should get
# the means of consecutive running pairs
@test downsample(r, x, 2) == [1.5, 3.5, 5.5, 7.5, 9.5]

# for scale 3, downsampling indices for `x` are 1:3, 4:6, 7:9. Take the means:
@test downsample(r, x, 3) == [2.0, 5.0, 8.0]

# for scale 4, downsampling indices for `x` are 1:4, 5:8. Take the means:
@test downsample(r, x, 4) == [2.5, 6.5]

# for scale 5, downsampling indices for `x` are 1:5, 6:10. Take the means:
@test downsample(r, x, 5) == [3.0, 8.0]

# for scale 6, only a single set of sampling indices is possible - 1:6
@test downsample(r, x, 6) == [3.5]
@test downsample(r, x, 6) == [3.5]

# For scales larger than the number of elements, there should be no values in the
# downsampled time series.
@test isempty(downsample(r, x, length(x) + 1))

##############################################################
# Regular downsampling
##############################################################
c = Composite(f = mean)
