# VectorSpin.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaVlasov.github.io/VectorSpin.jl/dev/)
[![Build Status](https://github.com/JuliaVlasov/VectorSpin.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaVlasov/VectorSpin.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaVlasov/VectorSpin.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaVlasov/VectorSpin.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


## Cite

Crouseilles N, Hervieux P-A, Hong X, Manfredi G. Vlasov–Maxwell equations with spin effects. Journal of Plasma Physics. 2023;89(2):905890215. doi:10.1017/S0022377823000314

## Installation

```bash
git clone https://github.com/JuliaVlasov/VectorSpin.jl.git
cd VectorSpin.jl
julia -O3 --check-bounds=no --project examples/vlasov-maxwell.jl
```
