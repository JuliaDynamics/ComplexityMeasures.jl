cd(@__DIR__)
using ComplexityMeasures
# Temporarily needed until new minor version of DimensionalData.jl is available
# (current version is 0.24.14)
using Pkg; Pkg.add(url="https://github.com/rafaqz/DimensionalData.jl#main")

# Convert tutorial file to markdown
import Literate
Literate.markdown("src/tutorial.jl", "src")

pages = [
    "index.md",
    "tutorial.md",
    "probabilities.md",
    "information_measures.md",
    "complexity.md",
    "convenience.md",
    "examples.md",
    "devdocs.md",
    "references.md",
]


import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "refs.bib");
    style=:authoryear
)

build_docs_with_style(pages, ComplexityMeasures, StateSpaceSets;
    expandfirst = ["index.md"], bib,
)
