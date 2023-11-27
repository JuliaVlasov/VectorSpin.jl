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

delta = ones(M) .* dt

x = collect(0:(M-1)) .* L ./ M # mesh in x
v = collect(1:N) .* 2H ./ N .- H # mesh in v

f1 = zeros(N, M)

for i in 1:M, j in 1:N
    f1[j, i] = exp(-(x[i]-5)^2) * exp(-(v[j]-5)^2)
end
contour(x, v, f1; aspect_ratio=1)
# -

@gif for i in 1:10
    translation!(f1, delta, mesh)
    contour(x, v, f1; aspect_ratio=1, legend = :none)
end

# +
f2 = similar(f1)
for i in 1:M, j in 1:N
    f2[j, i] = exp(-(x[i]-5)^2) * exp(-(v[j]+5)^2)
end

maximum(abs.(f1 .- f2))

# +
xmin, xmax, nx = 0, L, M
vmin, vmax, nv = -H, H, N
mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)
adv = PSMAdvection(mesh)
dt = 1.0

@gif for i in 1:10
    advection!(f1, adv, ones(M), dt)
end

contour(x, v, f1; aspect_ratio=1, legend = :none)


# +
f2 = similar(f1)
for i in 1:M, j in 1:N
    f2[j, i] = exp(-(x[i]-5)^2) * exp(-(v[j]-5)^2)
end

maximum(abs.(f1 .- f2))
# -


