# ComplexityMeasures.jl

[![docsdev](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/dev/)
[![docsstable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/)
[![CI](https://github.com/juliadynamics/ComplexityMeasures.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/ComplexityMeasures.jl/actions)
[![codecov](https://codecov.io/gh/JuliaDynamics/ComplexityMeasures.jl/branch/main/graph/badge.svg?token=6XlPGg5nRG)](https://codecov.io/gh/JuliaDynamics/ComplexityMeasures.jl)
[![Package Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FComplexityMeasures&query=total_requests&label=Downloads)](http://juliapkgstats.com/pkg/ComplexityMeasures)
[![Package Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FEntropies&query=total_requests&label=Downloads%20(Entropies))](http://juliapkgstats.com/pkg/Entropies)
[![publication](https://img.shields.io/badge/publication-arXiv.2406.05011-blueviolet.svg)](https://doi.org/10.48550/arXiv.2406.05011)

ComplexityMeasures.jl is a Julia-based software for calculating 1000s of various kinds of
probabilities, entropies, and other so-called _complexity measures_ from a single-variable input datasets. For relational measures across many input datasets see its extension [Associations.jl](https://juliadynamics.github.io/Associations.jl/dev/).
If you are a user of other programming languages (Python, R, MATLAB, ...),
you can still use ComplexityMeasures.jl due to Julia's interoperability.
For example, for Python use [`juliacall`](https://pypi.org/project/juliacall/).

A careful comparison with alternative widely used software shows that ComplexityMeasures.jl outclasses the alternatives in several objective aspects of comparison, such as computational performance, overall amount of measures, reliability, and extendability. See the associated publication for more details.

The key features that ComplexityMeasures.jl provides can be summarized as:

- A rigorous framework for extracting probabilities from data, based on the mathematical formulation of [probability spaces](https://en.wikipedia.org/wiki/Probability_space).
- Several (12+) outcome spaces, i.e., ways to discretize data into probabilities.
- Several estimators for estimating probabilities given an outcome space, which correct theoretically known estimation biases.
- Several definitions of information measures, such as various flavours of entropies (Shannon, Tsallis, Curado...), extropies, and other complexity measures, that are used in the context of nonlinear dynamics, nonlinear timeseries analysis, and complex systems.
- Several discrete and continuous (differential) estimators for entropies, which correct theoretically known estimation biases.
- An extendable interface and well thought out API accompanied by dedicated developer documentation. This makes it trivial to define new outcome spaces, or new estimators for probabilities, information measures, or complexity measures and integrate them with everything else in the software without boilerplate code.

ComplexityMeasures.jl can be used as a standalone package, or as part of other projects in the JuliaDynamics organization, such as [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/dev/) or [Associations.jl](https://juliadynamics.github.io/Associations.jl/dev/).

To install it, run `import Pkg; Pkg.add("ComplexityMeasures")`.

All further information is provided in the documentation, which you can either find [online](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/) or build locally by running the `docs/make.jl` file.

_Previously, this package was called Entropies.jl._
