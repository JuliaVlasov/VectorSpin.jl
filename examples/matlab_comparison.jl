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
using MAT
using Plots
using Test
using VectorSpin

const M = 119 # mesh number in x direction
const N = 129 # mesh number in v direction
const H = 10.0 / 2 # computational domain [-H/2,H/2] in v
const k = 0.5 # wave number/frequency
const L = 2π / k # computational domain [0,L] in x
const tildeK = 0.1598 # normalized parameter tildeK
const h = 0.1 # time step size
const n_i = 1.0
const K_xc = tildeK

x = collect(0:(M-1)) .* L ./ M # mesh in x
v1 = collect(1:N) .* 2H ./ N .- H # mesh in v

const a = 0.001 # perturbation for f
E1 = fft(-1.0 * a / k * sin.(k * x)) # electric field
epsil = a # perturbation for S
# -

xue = matread(joinpath("..", "test", "onestep.mat"))

# +
# spin variables
S1 = zeros(M)
S2 = zeros(M)
S3 = zeros(M)
dS1 = zeros(M)
dS2 = zeros(M)
dS3 = zeros(M)

for j = 1:M
    normx = sqrt(1 + epsil^2) #    norm of S
    dS1[j] = epsil * cos(k * x[j]) / normx
    dS2[j] = epsil * sin(k * x[j]) / normx
    dS3[j] = 1.0 / normx - 1.0
    S1[j] = dS1[j]
    S2[j] = dS2[j]
    S3[j] = 1.0 / normx
end

# +
v1node = zeros(5N)
for i = 1:N
    v1node[5i-4] = v1[i] - 2H / N
    v1node[5i-3] = v1[i] - (2H / N) * 0.75
    v1node[5i-2] = v1[i] - (2H / N) * 0.50
    v1node[5i-1] = v1[i] - (2H / N) * 0.25
    v1node[5i] = v1[i]
end

f0node = zeros(5N, M)
f1node = zeros(5N, M)
f2node = zeros(5N, M)
f3node = zeros(5N, M)

df0node = zeros(5N, M)
df3node = zeros(5N, M)

femi1 = 1
femi2 = -1

for j = 1:M, i = 1:5N
    f0node[i, j], df0node[i, j] = init(j, x, i, v1node, k, a, femi1, tildeK)
    f3node[i, j], df3node[i, j] = init(j, x, i, v1node, k, a, femi2, tildeK)
end

f0 = zeros(N, M)
f1 = zeros(N, M)
f2 = zeros(N, M)
f3 = zeros(N, M)
for j = 1:M
    f0[:, j] .= integrate(f0node[:, j], N)
    f1[:, j] .= integrate(f1node[:, j], N)
    f2[:, j] .= integrate(f2node[:, j], N)
    f3[:, j] .= integrate(f3node[:, j], N)
end
# -

@testset "first" begin 
    @test ex_energy(E1, L, M) ≈ first(xue["Ex_energy"])
    @test kinetic_energy(f0, M, N, L, H) ≈ first(xue["energykinetic"])
    @test energy(f0, f1, f2, f3, S1, S2, S3, E1, M, N, L, H, tildeK, n_i) ≈ first(xue["energy"])
    bf1, bf2, bf3 = bf_energy(f1, f2, f3, S1, S2, S3, M, N, L, H, tildeK)
    @test bf1 ≈ first(xue["energyBf1"])
    @test bf2 ≈ first(xue["energyBf2"])
    @test bf3 ≈ first(xue["energyBf3"])
    S1_nrj, S2_nrj, S3_nrj = s_energy(S1, S2, S3, M, N, L, H)
    @test S1_nrj ≈ first(xue["S1energy"]) atol=1e-15
    @test S2_nrj ≈ first(xue["S2energy"]) atol=1e-15
    @test S3_nrj ≈ first(xue["S3energy"]) atol=1e-15
    @test snorm(S1, S2, S3) ≈ first(xue["Snorm"])
end

ex_energy(E1, L, M) 

Hv!(f0, f1, f2, f3, E1, h, M, N, L, H)
He!(f0, f1, f2, f3, E1, h, H)
H1fh!(f0, f1, f2, f3, S1, S2, S3, h, M, N, L, H, tildeK)
H2fh!(f0, f1, f2, f3, S1, S2, S3, h, M, N, L, H, tildeK)
H3fh!(f0, f1, f2, f3, S1, S2, S3, h, M, N, L, H, tildeK)

hvmat = matread(joinpath("..","test","hv.mat"))

ex_energy(hvmat["E1"], L, M) ≈ ex_energy(E1, L, M)

ex_energy(E1, L, M)#, last(xue["Ex_energy"])

@testset "last" begin 
    @test ex_energy(E1, L, M) ≈ last(xue["Ex_energy"])
    @test kinetic_energy(f0, M, N, L, H) ≈ last(xue["energykinetic"])
    @test energy(f0, f1, f2, f3, S1, S2, S3, E1, M, N, L, H, tildeK, n_i) ≈ last(xue["energy"])
    bf1, bf2, bf3 = bf_energy(f1, f2, f3, S1, S2, S3, M, N, L, H, tildeK)
    @test bf1 ≈ last(xue["energyBf1"])
    @test bf2 ≈ last(xue["energyBf2"])
    @test bf3 ≈ last(xue["energyBf3"])
    S1_nrj, S2_nrj, S3_nrj = s_energy(S1, S2, S3, M, N, L, H)
    @test S1_nrj ≈ last(xue["S1energy"]) atol=1e-15
    @test S2_nrj ≈ last(xue["S2energy"]) atol=1e-15
    @test S3_nrj ≈ last(xue["S3energy"]) atol=1e-15
    @test snorm(S1, S2, S3) ≈ last(xue["Snorm"])
end

@test snorm(S1, S2, S3) ≈ last(xue["Snorm"])

plot(x, S1, label = "S1")
plot!(x, S2, label = "S2")
plot!(x, S3, label = "S3")
