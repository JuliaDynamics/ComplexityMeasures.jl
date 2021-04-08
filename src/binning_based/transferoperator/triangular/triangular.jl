include("simplex_types/simplex_types.jl")
include("./delaunay_triangulations/DelaunayTriangulations.jl")
import .DelaunayTriangulations: DelaunayTriangulation

abstract type TriangularBinning end

include("exact/SimplexExact.jl")
include("point/SimplexPoint.jl")