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
using Plots
using VectorSpin
using SparseArrays

"""
Given integral average in each cell;this function could compute
coefficients a;b;c of piecewise quadratic polynomial using PSM method
"""
function recover(f, N)

    # a;b;c are all row vectors
    f1 = zeros(N + 1)
    f1[2:N] .= f[1:N-1] .+ f[2:N]
    f1[1] = f[1]
    f1[N+1, 1] = f[N]
    downleft = ones(N) ./ 3
    diagonal = ones(N + 1)
    upright = ones(N) ./ 3
    diagonal[1] = 2 / 3
    diagonal[N+1] = 2 / 3
    diagonal[2:N] = 4 / 3 * ones(1, N - 1)
    A = spdiagm(-1 => downleft, 0 => diagonal, 1 => upright)
    # get result
    c1 = A \ f1
    c = c1[1:N]

    a = zeros(N)
    b = zeros(N)
    cc = zeros(N)
    for i = 2:N
        cc[i] = (-1)^i * (c1[i] - c1[i-1])
    end
    for i = 2:N
        b[i] = (-1)^i * 2 * sum(cc[2:i])
    end
    b[1] = 0
    a[1:N-1] = 1 / 2 * (b[2:N] - b[1:N-1])
    a[N] = -1 / 2 * b[N]
    return a, b, c
end

"""
- oldvector is the integral average value in each cell of function f(x)
- newvector is the integral average value in each cell of function f(x+delta)
"""
function translation(oldvector, N, delta, H)
    # first recover oldvector & get the coefficients of piecewise polynomials
    a, b, c = recover(oldvector, N)
    newvector = zeros(N)
    for i = 1:N
        beta = i + delta[i] / (H / N)
        loopnumber = floor(Int, beta / N)
        newbeta = beta - N * loopnumber

        if (abs(newbeta) < 1e-20) || (abs(newbeta - N) < 1e-20)
            newvector[i] = oldvector[N]
        elseif newbeta >= 1.0
            index = floor(Int, newbeta)
            k = 1 - (newbeta - index)
            valueI = a[index] / 3 + b[index] / 2 + c[index]
            valueI = valueI - a[index] / 3 * (1 - k)^3 - b[index] / 2 * (1 - k)^2
            valueI = valueI - c[index] * (1 - k)
            valueII = a[index+1] / 3 * (1 - k)^3 + b[index+1] / 2 * (1 - k)^2
            valueII = valueII + c[index+1] * (1 - k)
            newvector[i] = valueI + valueII

        else
            index = N
            k = 1 - newbeta
            valueI = a[index] / 3 + b[index] / 2 + c[index]
            valueI = valueI - a[index] / 3 * (1 - k)^3 - b[index] / 2 * (1 - k)^2
            valueI = valueI - c[index] * (1 - k)
            valueII = a[1] / 3 * (1 - k)^3 + b[1] / 2 * (1 - k)^2 + c[1] * (1 - k)
            newvector[i] = valueI + valueII
        end

    end
    return newvector
end

M = 119 # mesh number in x direction
N = 129 # mesh number in v direction
H = 10.0  # computational domain [-H/2,H/2] in v
L = 10.0  # computational domain [0,L] in x

xmin, xmax, nx = 0.0, L, M
vmin, vmax, nv = -H/2, H/2, N
mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)
adv = PSMAdvection(mesh)

dt = 0.1

f = zeros(N, M)

for i = eachindex(mesh.x), j = eachindex(mesh.v)
    f[j, i] = exp(-(mesh.x[i] - 2)^2) * exp(-mesh.v[j]^2)
end

f_old = copy(f)
f_new = copy(f)

v = ones(M)

@time advection!(f, adv, v, dt)


@time begin

    for i in eachindex(v)
        delta = - v[i] * dt * ones(N)
        f_new[:, i] .= translation(f_old[:, i], N, delta, H)
    end

end


@show maximum(abs.(f_new .- f))
