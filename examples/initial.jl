import Pkg
Pkg.add(["DispersionRelations", "ProgressMeter"])

using FFTW
using Plots
using ProgressMeter
using TimerOutputs
using VectorSpin

using DispersionRelations

const to = TimerOutput()

function maxwellian(k, x, i, v1int, frequency, a, femi, tiK)

    kk = 1.0 # vth

    if femi == 1
        femi = 1
    else
        femi = 0.5
    end

    f =
        (1 / sqrt(pi) / kk) *
        exp(-(v1int[i])^2 / kk / kk) *
        (1 + a * cos(frequency * x[k])) *
        femi
    df =
        (1 / sqrt(pi) / kk) *
        exp(-(v1int[i])^2 / kk / kk) *
        (a * cos(frequency * x[k])) *
        femi

    return f, df
end

function numeint(value, N)
    integral = 7 / 90 * value[1:5:5N-4] .+ 16 / 45 * value[2:5:5N-3]
    integral = integral .+ 2 / 15 * value[3:5:5N-2]
    integral = integral .+ 16 / 45 * value[4:5:5N-1]
    integral = integral .+ 7 / 90 * value[5:5:5N]
    integral
end

ex_energy(E1, mesh) = sum(abs2.(ifft(E1))) * mesh.dx

function main(T, M, N, H, kkk, L, h, a)

    NUM = floor(Int, T / h + 1.1) # time step number
    x = collect(0:(M-1)) .* L ./ M # mesh in x
    v1 = collect(1:N) .* 2H ./ N .- H # mesh in v
    E1 = fft(-1.0 * a / kkk * sin.(kkk * x)) # electric field
    epsil = a # perturbation for S

    # spin variables
    S1 = zeros(M)
    S2 = zeros(M)
    S3 = zeros(M)

    dS1 = zeros(M)
    dS2 = zeros(M)
    dS3 = zeros(M)

    # initial value of S
    for k = 1:M
        normx = sqrt(1.0 + epsil^2) #    norm of S
        dS1[k] = epsil * cos(kkk * x[k]) / normx
        dS2[k] = epsil * sin(kkk * x[k]) / normx
        dS3[k] = 1.0 / normx - 1.0
        S1[k] = dS1[k]
        S2[k] = dS2[k]
        S3[k] = 1.0 / normx
    end

    v1node = zeros(5N)
    for i = 1:N
        v1node[5i-4] = v1[i] - 2H / N
        v1node[5i-3] = v1[i] - (2H / N) * 3 / 4
        v1node[5i-2] = v1[i] - (2H / N) * 1 / 2
        v1node[5i-1] = v1[i] - (2H / N) * 1 / 4
        v1node[5i] = v1[i]
    end

    f0_node = zeros(5N, M)
    df0_node = zeros(5N, M)
    f1_node = zeros(5N, M)
    f2_node = zeros(5N, M)
    f3_node = zeros(5N, M)
    df3_node = zeros(5N, M)
    femi1 = 1
    femi2 = -1


    for k = 1:M, i = 1:5N
        f0_node[i, k], df0_node[i, k] = maxwellian(k, x, i, v1node, kkk, a, femi1, tildeK)
        f1_node[i, k] = 0.0
        f2_node[i, k] = 0.0
        f3_node[i, k], df3_node[i, k] = maxwellian(k, x, i, v1node, kkk, a, femi2, tildeK)
    end

    initialvalue_f0 = zeros(N, M)
    initialvalue_f1 = zeros(N, M)
    initialvalue_f2 = zeros(N, M)
    initialvalue_f3 = zeros(N, M)

    # initial value of f

    for k = 1:M
        initialvalue_f0[:, k] .= numeint(f0_node[:, k], N)
        initialvalue_f1[:, k] .= numeint(f1_node[:, k], N)
        initialvalue_f2[:, k] .= numeint(f2_node[:, k], N)
        initialvalue_f3[:, k] .= numeint(f3_node[:, k], N)
    end

    f0 = initialvalue_f0
    f1 = initialvalue_f1
    f2 = initialvalue_f2
    f3 = initialvalue_f3


    mesh = Mesh(0.0, L, M, -H, H, N)
    Hv = HvOperator(mesh)
    He = HeOperator(mesh)
    H1fh = H1fhOperator(mesh)
    H2fh = H2fhOperator(mesh)
    H3fh = H3fhOperator(mesh)

    t = Float64[]
    e = Float64[]
    push!(t, 0.0)
    push!(e, ex_energy(E1, mesh))

    @showprogress 1 for i = 2:NUM # run with time 
        # Lie splitting
        @timeit to "Hv" step!(Hv, f0, f1, f2, f3, E1, h)
        @timeit to "He" step!(He, f0, f1, f2, f3, E1, h)
        @timeit to "H1fh" step!(H1fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)
        @timeit to "H2fh" step!(H2fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)
        @timeit to "H3fh" step!(H3fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)
        push!(t, (i - 1) * h)
        push!(e, ex_energy(E1, mesh))
    end


    return t, e

end

# mesh and parameters 
T = 500 # final simulation time
M = 119 # mesh number in x direction
N = 129 # mesh number in v direction
H = 10.0 / 2 # computational domain [-H/2,H/2] in v
kx = 0.5 # wave number/frequency
L = 2π / kx # computational domain [0,L] in x
tildeK = 0.1598 # normalized parameter tildeK
h = 0.1 # time step size
a = 0.001 # perturbation for f

t, e = main(T, M, N, H, kx, L, h, a)

show(to)
plot(t, e, label = "ex energy", yscale = :log10)
line, γ = fit_complex_frequency(t, e)
plot!(t, line, label = "γ = $(imag(γ))", legend = :bottomleft)
