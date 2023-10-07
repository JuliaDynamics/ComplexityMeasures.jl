# The contents of this file is directly copied from DimensionalData.jl v0.25.2,
# but modified slightly so that we can reduce the amount of unnecessary printing
# for `Counts` and `Probabilities`.
import DimensionalData: NoName

const CountsOrProbs{T, N} = Union{Counts{T, N}, Probabilities{T, N}}

function Base.summary(io::IO, A::CountsOrProbs{T,1}) where {T}
    print(io, size(A, 1), "-element ")
    print(io, string(nameof(typeof(A)), "{$T,1}"))
end
function Base.summary(io::IO, A::CountsOrProbs{T,N}) where {T,N}
    print(io, join(size(A), "×"), " ")
    print(io, string(nameof(typeof(A)), "{$T,$N}"))
end

function Base.show(io::IO, mime::MIME"text/plain", c::CountsOrProbs)
    lines = 0
    if c isa Counts
        A = c.cts
    elseif c isa Probabilities
        A = c.p
    end
    summary(io, c)
    print_name(io, name(c))
    print("\n")

    # Printing the array data is optional, subtypes can 
    # show other things here instead.
    ds = displaysize(io)
    ioctx = IOContext(io, :displaysize => (ds[1] - lines, ds[2]))
    DimensionalData.show_after(ioctx, mime, A)
    return nothing
end

# print a name of something, in yellow
function print_name(io::IO, name)
    if !(name == Symbol("") || name isa NoName)
        printstyled(io, string(" ", name); color=:yellow)
    end
end
