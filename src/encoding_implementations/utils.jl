# Normalize `x::Real ∈ [minval, maxval]` to the interval [minval, maxval]. Assumes
# `minval < maxval`. This needs to be checked at a higher level if this function is called.
function norm_minmax(x, minval, maxval)
    return (x - minval) / (maxval - minval)
end
