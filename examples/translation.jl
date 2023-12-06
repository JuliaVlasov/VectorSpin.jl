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
using BenchmarkTools
using VectorSpin

M = 100 # mesh number in x direction
N = 200 # mesh number in v direction
H = 10.0  # computational domain [-H/2,H/2] in v
L = 10.0  # computational domain [0,L] in x

xmin, xmax, nx = 0.0, L, M
vmin, vmax, nv = -H, H, N
mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)
adv = PSMAdvection(mesh)

dt = 0.1

f = zeros(N, M)

for i = 1:M, j = 1:N
    f[j, i] = exp(-(mesh.x[i] - 5)^2) * exp(-(mesh.v[j] - 5)^2)
end

v = ones(M)

advection!(f, adv, v, dt)

@time advection!(f, adv, v, dt)

