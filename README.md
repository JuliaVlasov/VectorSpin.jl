# VectorSpin.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaVlasov.github.io/VectorSpin.jl/dev/)
[![Build Status](https://github.com/JuliaVlasov/VectorSpin.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaVlasov/VectorSpin.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaVlasov/VectorSpin.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaVlasov/VectorSpin.jl)

## Cite

Nicolas Crouseilles, Paul-Antoine Hervieux, Xue Hong, Giovanni Manfredi. Vlasov-Maxwell equations with spin effects. 2023. [hal-03960201](https://hal.inria.fr/hal-03960201/)

## Installation

```bash
git clone https://github.com/JuliaVlasov/VectorSpin.jl.git
cd VectorSpin.jl
julia -O3 --check-bounds=no --project examples/inplace.jl
```
This first implementation is translated from a MATLAB version. 

There is two implementations of the same algorithm. If you want to test the other one

```bash
julia -O3 --check-bounds=no --project examples/operators.jl
```

In this version, we use Julia Structs to reduce memory print and improve the readibility.
