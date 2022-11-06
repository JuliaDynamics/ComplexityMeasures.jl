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

@test_throws DomainError downsample(r, x, 6)

##############################################################
# Regular downsampling
##############################################################
c = Composite(f = mean)
# for scale 2, downsampling indices for `x` are 1:2, 3:4, 5:6, 7:8, 9:10, so we should get
# the means of consecutive running pairs
@test downsample(c, x, 2) == [
    [1.5, 3.5, 5.5, 7.5], # indices 1:2, 3:4, 5:6, 7:8
    [2.5, 4.5, 6.5, 8.5], # indices 2:3, 4:5, 6:7, 8:9
]
@test downsample(c, x, 3) == [
    [2.0, 5.0], # indices 1:3, 4:6
    [3.0, 6.0], # indices 2:4, 5:7
    [4.0, 7.0], # indices 3:5, 6:8
]

@test downsample(c, x, 4) == [
    [2.5], # indices 1:4
    [3.5], # indices 2:5
    [4.5], # indices 3:6
    [5.5], # indices 4:7
]

@test downsample(c, x, 5) == [
    [3.0], # indices 1:5
    [4.0], # indices 2:6
    [5.0], # indices 3:7
    [6.0], # indices 4:8
    [7.0], # indices 5:9
]

@test_throws DomainError downsample(c, x, 6)
