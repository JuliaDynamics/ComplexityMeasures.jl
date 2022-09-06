export binhist

probabilities(x) = _non0hist(x)

# Frequencies are needed elsewhere in the package too, so keep in its own method.
function _non0hist_frequencies(x)
    L = length(x)

    hist = Vector{Float64}()
    # Reserve enough space for histogram:
    sizehint!(hist, L)

    sx = sort(x, alg = QuickSort)

    # Fill the histogram by counting consecutive equal values:
    prev_val, count = sx[1], 0
    for val in sx
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

function _non0hist(x, N)
    hist = _non0hist_frequencies(x)
    return Probabilities(hist ./ N)
end

function _non0hist(x)
    return _non0hist(x, length(x))
end

_non0hist(x::AbstractDataset) = _non0hist(x.data)
_non0hist(x::AbstractDataset, N) = _non0hist(x.data, N)
probabilities(x::AbstractDataset) = _non0hist(x.data)
