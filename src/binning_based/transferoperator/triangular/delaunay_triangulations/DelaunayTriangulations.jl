module DelaunayTriangulations
    using Simplices
    include("AbstractDelaunayTriangulation.jl")
    include("DelaunayTriangulation.jl")
    #include("plot_recipes/plot_recipes.jl")
end

"""
    DelaunayTriangulations

A module handling the creation of Delaunay triangulations from data.
"""
DelaunayTriangulations
