cd(@__DIR__)

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/apply_style.jl",
    joinpath(@__DIR__, "apply_style.jl")
)
include("apply_style.jl")

using ComplexityMeasures
import ComplexityMeasures.Wavelets

ENTROPIES_PAGES = [
    "index.md",
    "probabilities.md",
    "encodings.md",
    "entropies.md",
    "complexity.md",
    "convenience.md",
    "examples.md",
    "devdocs.md",
]

makedocs(
    modules = [ComplexityMeasures, StateSpaceSets],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 3,
    ),
    sitename = "ComplexityMeasures.jl",
    authors = "Kristian Agasøster Haaga, George Datseris",
    pages = ENTROPIES_PAGES,
    doctest = false,
    draft = false,
)

if CI
    deploydocs(
        repo = "github.com/JuliaDynamics/ComplexityMeasures.jl.git",
        target = "build",
        push_preview = true
    )
end
