using ComplexityMeasures, Test

x = ones(3)
p = probabilities(x)
@test p isa Probabilities
@test p == [1]

# Test all probabilities
# Okay, with these inputs we know exactly what the final scenario
# should be. Bins 1 2 and 3 have one entry, bin 4 has zero, bin 5
# has two entries, bins 6 7 8 have zero, bin 9 has 1 and bin 10 zero
x = [0.05, 0.15, 0.25, 0.45, 0.46, 0.85]
est = ValueHistogram(FixedRectangularBinning((0:0.1:1,)))


# %%






i = 1
consecutive_empty = 0
inequality_index = 1
where_to_fill = Int[]
how_many_to_fill = Int[]

space_index = 1
space_length = length(ospace)

# iterate over observed outcomes
j = 1
while j ≤ length(outs)
    outcome = outs[j]
    if outcome != ospace[space_index]
        # We've reached a point of inequality
        push!(where_to_fill, j)
        # Count how many we need to reach the next point of equality
        consecutive_empty = 1
        space_index += 1
        while outcome != ospace[space_index] && space_index < space_length
            space_index += 1
            consecutive_empty += 1
        if
        push!(how_many_to_fill, consecutive_empty)
        # increment space index to next element, as current is equal with current outcome
        space_index += 1
    else
        j += 1
        space_index += 1
    end
    # last check in case
end

@show ospace
@show outs
@show how_many_to_fill
@show where_to_fill

@show sum(how_many_to_fill) + length(probs)
@show space_length

