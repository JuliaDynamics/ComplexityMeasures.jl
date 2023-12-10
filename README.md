# ComplexityMeasures.jl

[![docsdev](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/dev/)
[![docsstable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/)
[![CI](https://github.com/juliadynamics/ComplexityMeasures.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/ComplexityMeasures.jl/actions)
[![codecov](https://codecov.io/gh/JuliaDynamics/ComplexityMeasures.jl/branch/main/graph/badge.svg?token=6XlPGg5nRG)](https://codecov.io/gh/JuliaDynamics/ComplexityMeasures.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/ComplexityMeasures)](https://pkgs.genieframework.com?packages=ComplexityMeasures)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Entropies)](https://pkgs.genieframework.com?packages=Entropies)
[![DOI](https://zenodo.org/badge/306859984.svg)](https://zenodo.org/badge/latestdoi/306859984)

A Julia package that provides:

- a rigorous framework for extracting probabilities from data, based on the mathematical formulation of [probability spaces](https://en.wikipedia.org/wiki/Probability_space)
- several (12+) outcome spaces, i.e., ways to discretize data into probabilities
- several estimators for estimating probabilities given an outcome space, which correct theoretically known estimation biases
- several definitions of information measures, such as various flavours of entropies (Shannon, Tsallis, Curado...), extropies, and probability-based complexity measures, that are used in the context of nonlinear dynamics, nonlinear timeseries analysis, and complex systems
- several discrete and continuous (differential) estimators for entropies, which correct theoretically known estimation biases
- estimators for other complexity measures that are not estimated based on probability functions
- an extendable interface and well thought out API that makes it trivial to define new outcome spaces, or new estimators for probabilities, information measures, or complexity measures

ComplexityMeasures.jl can be used as a standalone package, or as part of other projects in the JuliaDynamics organization, such as [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/dev/) or [CausalityTools.jl](https://juliadynamics.github.io/CausalityTools.jl/dev/).

To install it, run `import Pkg; Pkg.add("ComplexityMeasures")`.

All further information is provided in the documentation, which you can either find [online](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/) or build locally by running the `docs/make.jl` file.

_Previously, this package was called Entropies.jl._
