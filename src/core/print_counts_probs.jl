# The contents of this file is modified from DimensionalData.jl, but modified a bit, 
# to reduce the amount of unnecessary printing. We simply want to display the outcomes
# in the marginals for both `Counts` and `Probabilities`.
import Base.print_array

const CountsOrProbs{T, N} = Union{Counts{T, N}, Probabilities{T, N}}

function Base.summary(io::IO, A::CountsOrProbs{T,1}) where {T}
    print(io, string(nameof(typeof(A)), "{$T,1}"))
    print(io, " over $(length(A)) outcomes")
end
function Base.summary(io::IO, A::CountsOrProbs{T,N}) where {T,N}
    print(io, join(size(A), "×"), " ")
    print(io, string(nameof(typeof(A)), "{$T,$N}"))
end

function Base.show(io::IO, mime::MIME"text/plain", x::CountsOrProbs)
    lines = 0
    if x isa Counts
        A = x.cts
    elseif x isa Probabilities
        A = x.p
    end
    print_name(io, x)
    println(io)

    ds = displaysize(io)
    print_array(IOContext(io, :displaysize => (ds[1] - lines, ds[2])), x)
    return nothing
end

function print_name(io, A::CountsOrProbs{T,1}) where {T}
    namestr = string(nameof(typeof(A)), "{$T,1}")
    outstr = "over $(length(A)) outcomes"
    r = string(" ", namestr * " " * outstr)
    printstyled(io, r; color=:yellow)
end

function print_name(io, A::CountsOrProbs{T,N}) where {T,N}
    container_size = join(size(A), "×")
    namestr = string(nameof(typeof(A)), "{$T,$N}")
    printstyled(io, string(" ", container_size * " " * namestr); color=:yellow)
end

function print_array(io::IO, x::CountsOrProbs{T, 1}) where T
    print_array_with_margins(io::IO, get_data(x), outcomes(x))
end

function print_array(io::IO, x::CountsOrProbs{T, 2}) where T
    print_array_with_margins(io::IO, get_data(x), outcomes(x))
end

function print_array(io::IO, x::CountsOrProbs{T, 3}) where T
    # We only view a 2D slice of the array.
    i3 = firstindex(x, 3)
    slice = get_data(x)[:, :, i3]
    println(io, "[:, :, $i3]")

    # Print compact representations.
    ctx = IOContext(io, :compact=>true, :limit=>true, :typeinfo=>T)
    print_array_with_margins(ctx, slice, outcomes(x, (1, 2)))
    nremaining = size(x, 3) - 1
    nremaining > 0 && printstyled(io, "\n[and $nremaining more slices...]"; color=:light_black)
end

function print_array(io::IO, x::CountsOrProbs{T, N}) where {T, N}
    # We only view a 2D slice of the array.
    o = ntuple(xᵢ -> firstindex(x, xᵢ + 2), N-2)
    slice = get_data(x)[:, :, o...]
    onestring = join(o, ", ")
    nremaining = size(x, 3) - 1
    println(io, "[:, :, $(onestring)]")

    # Print compact representations.
    ctx = IOContext(io, :compact=>true, :limit=>true, :typeinfo=>T)
    print_array_with_margins(ctx, slice, outcomes(x, (1, 2)))
    nremaining = prod(size(x, d) for d=3:N) - 1
    nremaining > 0 && printstyled(io, "\n[and $nremaining more slices...]"; color=:light_black)
end


function get_data(x::CountsOrProbs)
    if x isa Counts
        return x.cts
    elseif x isa Probabilities
        return x.p
    end
end

# 1D printing
"""
    print_array_with_margins(io::IO, x::AbstractArray{T, 1}, margin::AbstractVector) where T

Prints a length-`N` vector `x` alongside the `margin` elements, such that `x[i]` 
corresponds to the marginal element `margin[i]`.
"""
function print_array_with_margins(io::IO, x::AbstractArray{T, 1}, margin::AbstractVector) where T
    h, w = displaysize(io)
    # Outcomes along first dimension.
    outs_x1 = vectorized_outcomes(margin)

    wn = w ÷ 3 # integers take 3 columns each when printed, floats more
    f1, l1, s1 = firstindex(x, 1), lastindex(x, 1), size(x, 1)
    itop =    s1 < h ? (f1:l1) : (f1:f1 + (h ÷ 2) - 1)
    ibottom = s1 < h ? (1:0)   : (f1 + s1 - (h ÷ 2) - 1:f1 + s1 - 1)
    labels = vcat(map(showblack, parent(outs_x1)[itop]), map(showblack, parent(outs_x1))[ibottom])
    vals = map(showdefault, vcat(x[itop], x[ibottom]))
    A_dims = hcat(labels, vals)
    Base.print_matrix(io, A_dims)
    return nothing
end

# 2D matrix. This method is all we need, because for higher dimensional arrays, we 
# show 2D slices, not the entire ND arrays.
function print_array_with_margins(io::IO, x::AbstractArray{T, 2}, margins::Tuple{Vararg{AbstractVector, 2}}) where {T}
    h, w = displaysize(io)
    outs_x1, outs_x2 = vectorized_outcomes.(margins)
    h, w = displaysize(io)
    wn = w ÷ 3 # integers take 3 columns each when printed, floats more
    f1, f2 = firstindex(outs_x1), firstindex(outs_x2)
    l1, l2 = lastindex(outs_x1), lastindex(outs_x2)
    s1, s2 = size(x)
    itop    = s1 < h  ? (f1:l1)     : (f1:h ÷ 2 + f1 - 1)
    ibottom = s1 < h  ? (f1:f1 - 1) : (f1 + s1 - h ÷ 2 - 1:f1 + s1 - 1)
    ileft   = s2 < wn ? (f2:l2)     : (f2:f2 + wn ÷ 2 - 1)
    iright  = s2 < wn ? (f2:f2 - 1) : (f2 + s2 - wn ÷ 2:f2 + s2 - 1)

    topleft = map(showdefault, x[itop, ileft])
    bottomleft = x[ibottom, ileft]

    # Labels in the left margin are color black.
    topleft = hcat(map(showblack, outs_x1[itop]), topleft)
    bottomleft = hcat(map(showblack, outs_x1[ibottom]), bottomleft)

    # Combine left labels with the array.
    leftblock = vcat(topleft, bottomleft)
    rightblock = vcat(x[itop, iright], x[ibottom, iright])
    bottomblock = hcat(leftblock, rightblock)
    
    # Marginal labels at the top are colored black.
    toplabel_left = map(showblack, outs_x2[ileft])
    toplabel_right = map(showblack, outs_x2[iright])

    # The top left corner should be an empty cell, so we just hide it.
    placeholder_topleft_corner = showhide("0")

    toprow = vcat(placeholder_topleft_corner, toplabel_left, toplabel_right) |> permutedims
    bottom_content = map(showdefault, bottomblock)
    processed_matrix = vcat(toprow, bottom_content)
    Base.print_matrix(io, processed_matrix)
    return nothing
end

function vectorized_outcomes(outs_along_single_dim)
    if outs_along_single_dim isa AbstractRange
        return collect(outs_along_single_dim)
    else
        return outs_along_single_dim
    end
end

function vectorized_outcomes(x::CountsOrProbs{T, 1}) where T
    vectorized_outcomes(outcomes(x))
end

struct ShowWith <: AbstractString
    val::Any
    hide::Bool
    color::Symbol
end
ShowWith(val; hide=false, color=:light_black) = ShowWith(val; hide, color)
function Base.show(io::IO, mime::MIME"text/plain", x::ShowWith; kw...)
    s = sprint(show, mime, x.val; context=io, kw...)
    s1 = x.hide ? " "^length(s) : s
    printstyled(io, s1; color=x.color)
end
showdefault(x) = ShowWith(x, false, :default)
showblack(x) = ShowWith(x, false, :light_black)
showhide(x) = ShowWith(x, true, :nothing)

Base.alignment(io::IO, x::ShowWith) = Base.alignment(io, x.val)
Base.length(x::ShowWith) = length(string(x.val))
Base.ncodeunits(x::ShowWith) = ncodeunits(string(x.val))
function Base.print(io::IO, x::ShowWith)
    printstyled(io, string(x.val); color = x.color, hidden = x.hide)
end
function Base.show(io::IO, x::ShowWith)
    printstyled(io, string(x.val); color = x.color, hidden = x.hide)
end

Base.iterate(x::ShowWith) = iterate(string(x.val))
Base.iterate(x::ShowWith, i::Int) = iterate(string(x.val), i::Int)