x = ones(3)
p = probabilities(ValueHistogram(0.1), x)
@test p isa Probabilities
