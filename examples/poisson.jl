# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
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


M = 119      # mesh number in x direction
N = 129      # mesh number in v direction
H = 10.0 / 2 # computational domain [-H/2,H/2] in v
k = 0.5      # wave number/frequency
L = 2π / k   # computational domain [0,L] in x


a = 0.001
x = (0:(M-1)) .* L / M # mesh in x
v = (1:N) * 2 * H / N .- H # mesh in v

E_exact = -a ./ k .* sin.(k * x)
E1 = fft(E_exact)
plot(x, E_exact)
# -

f0 = zeros(N, M)
for j = 1:M, i = 1:N
    f0[i, j] = (1.0 / sqrt(pi)) * exp(-v[i]^2) * (1.0 + a * cos(k * x[j]))
end
# +
f1 = zeros(N, M)
f2 = zeros(N, M)
f3 = zeros(N, M)

t = 0.0
E1 = Hv!(f0, f1, f2, f3, t, M, N, L, H)
plot(x, real(ifft(E1)) .- E_exact)
# -
