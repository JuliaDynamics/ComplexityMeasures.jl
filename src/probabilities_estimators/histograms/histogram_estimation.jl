# Fast histograms are internal functions, NOT exported in the public API,
# because `probabilities_and_outcomes` is all the user would need.
# The functions _are_ part of the DEV API, and could be used downstream.
# (documented and tested).
"""
    fasthist!(x) → c::Vector{Int}

Count the occurrences `c` of the unique data values in `x`,
so that `c[i]` is the number of times the value
`sort!(unique(x))[i]` occurs. Hence, this method is useful mostly when
`x` contains integer or categorical data.

Prior to counting, `x` is sorted, so this function also mutates `x`.
Therefore, it is called with `copy` in higher level API when necessary.
This function works for any `x` for which `sort!(x)` works.
"""
function fasthist!(x)
    L = length(x)
    hist = Vector{Int}()
    # Reserve enough space for histogram (Base suggests this improves performance):
    sizehint!(hist, L)
    # Fill the histogram by counting consecutive equal values:
    sort!(x; alg = QuickSort)
    prev_val, count = x[1], 0
    for val in x
        if val == prev_val
            count += 1
        else
            push!(hist, count)
            prev_val = val
            count = 1
        end
    end
    push!(hist, count)
    # Shrink histogram capacity to fit its size:
    sizehint!(hist, length(hist))
    return hist
end
