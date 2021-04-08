export MutableSimplex

abstract type AbstractMutableSimplex{D, T} <: AbstractSimplex{D, T} end

"""
    MutableSimplex{D, T}

Simplex where vertices are represented by some type of abstract mutable vector.
"""
mutable struct MutableSimplex{D, T} <: AbstractMutableSimplex{D, T}
    vertices::Vector{Vector{T}}

    function MutableSimplex(pts::Vector{<:AbstractVector{T}}) where {T}
        if !(length(pts) == length(pts[1]) + 1)
            err = """ The input cannot be converted to a simplex.
            Vertices need to have `dim` elements, and there needs to be `dim + 1` vertices.
            """
            throw(DomainError(pts, err))    end
        dim = length(pts[1])
        new{dim, T}([pts[i] for i = 1:length(pts)])
    end

    function MutableSimplex(pts::AbstractArray{T, 2}) where {T}
        s = size(pts)

        if (maximum(s) - minimum(s)) != 1
            err = """ The input cannot be converted to a simplex.
                    size(pts) must be (dim, dim + 1) or (dim + 1, dim).
                """
                throw(DomainError(pts, err))
        end

        if s[1] > s[2] # Rows are points
            dim = s[2]
            return new{dim, T}([pts[i, :] for i = 1:maximum(s)])
        end

        if s[2] > s[1] # Columns are points
            dim = s[1]
            return new{dim, T}([pts[:, i] for i = 1:maximum(s)])
        end
    end
end

# Overwriting the i-th vertex
function Base.setindex!(simplex::MutableSimplex, v, i)
    simplex[i] .= v
end

# Overwriting elements of the i-th vertex
function Base.setindex!(simplex::MutableSimplex, v, i::Int, j)
    if length(v) != length(j)
        err = """
            Trying to overwrite elements $j of vertex $i with $v, which does not
            have the same number of elements as the target.
        """
        throw(ArgumentError(err))
    end
    simplex[i][j] = v
end
