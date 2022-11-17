x = ones(3)
p = probabilities(x, ValueHistogram(0.1))
@test p isa Probabilities
