using RecipesBase

import StaticArrays:
    SArray,
    MArray

import ..Simplices:
    Simplex,
    SSimplex,
    MutableSSimplex,
    connectvertices,
    splitaxes

##################################
# Plotting single triangulations
##################################
@recipe function f(pts::Vector{Vector{T}}, DT::DelaunayTriangulation) where {T}

    for i = 1:length(DT)
        s = Simplex(pts[DT[i]])
        @series begin
            seriestype := :path
            splitaxes(connectvertices(s))
        end
    end
end

@recipe function f(pts::AbstractArray{T, 2}, DT::DelaunayTriangulation) where {T}

    for i = 1:length(DT)
        s = Simplex(pts[:, DT[i]])
        @series begin
            seriestype := :path
            splitaxes(connectvertices(s))
        end
    end
end

@recipe function f(pts::Vector{MArray{Tuple{D},T,1,D}}, DT::DelaunayTriangulation) where {D, T}

    for vertexinds in DT
        s = MutableSSimplex(pts[vertexinds])
        @series begin
            seriestype := :path
            splitaxes(connectvertices(s))
        end
    end
end


@recipe function f(pts::Vector{SArray{Tuple{D},T,1,D}}, DT::DelaunayTriangulation) where {D, T}

    for vertexinds in DT
        s = SSimplex(pts[vertexinds])
        @series begin
            seriestype := :path
            splitaxes(connectvertices(s))
        end
    end
end
