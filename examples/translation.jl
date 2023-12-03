# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Julia 1.9.3
#     language: julia
#     name: julia-1.9
# ---

# +
using FFTW
using Plots
using VectorSpin

const M = 100 # mesh number in x direction
const N = 200 # mesh number in v direction
const H = 10.0  # computational domain [-H/2,H/2] in v
const L = 10.0  # computational domain [0,L] in x
dt = 0.1

mesh = Mesh(0, L, M, -H, H, N)

f1 = zeros(N, M)

for i = 1:M, j = 1:N
    f1[j, i] = exp(-(x[i] - 5)^2) * exp(-(v[j] - 5)^2)
end
contour(x, v, f1; aspect_ratio = 1)
# -

# +
xmin, xmax, nx = 0, L, M
vmin, vmax, nv = -H, H, N
mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)
adv = PSMAdvection(mesh)
dt = 1.0

@gif for i = 1:10
    advection!(f1, adv, ones(M), dt)
end

contour(x, v, f1; aspect_ratio = 1, legend = :none)
