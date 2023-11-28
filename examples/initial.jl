using FFTW
using MAT
using Plots
using ProgressMeter
using TimerOutputs
using VectorSpin

const to = TimerOutput()

# mesh and parameters 
const T = 500 # final simulation time
const M = 119 # mesh number in x direction
const N = 129 # mesh number in v direction
const H = 10.0 / 2 # computational domain [-H/2,H/2] in v
const kkk = 0.5 # wave number/frequency
const L = 2pi / kkk # computational domain [0,L] in x
const tildeK = 0.1598 # normalized parameter tildeK
const h = 0.1 # time step size
const a = 0.001 # perturbation for f

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

function energy(f0, f1, f2, f3, E1, S1, S2, S3, mesh, tiK)

    M, N = mesh.nx, mesh.nv
    L = mesh.xmax - mesh.xmin
    H = 0.5 * (mesh.vmax - mesh.vmin)
    value1 = 1:(M-1)÷2+1
    value2 = (M-1)÷2+2:M
    mub = 0.3386
    h_int = 2.0 * mub
    aj0 = 0.01475 * h_int / 2.0
    K_xc = tiK
    n_i = 1.0

    S1value = fft(S1)
    S2value = fft(S2)
    S3value = fft(S3)
    S1value[value1] = ((2pi * 1im / L * (value1 .- 1))) .* S1value[value1]
    S1value[value2] = ((2pi * 1im / L * (value2 .- 1 .- M))) .* S1value[value2]
    SS1value = real(ifft(S1value))
    S2value[value1] = ((2pi * 1im / L * (value1 .- 1))) .* S2value[value1]
    S2value[value2] = ((2pi * 1im / L * (value2 .- 1 .- M))) .* S2value[value2]
    SS2value = real(ifft(S2value))
    S3value[value1] = ((2pi * 1im / L * (value1 .- 1))) .* S3value[value1]
    S3value[value2] = ((2pi * 1im / L * (value2 .- 1 .- M))) .* S3value[value2]
    SS3value = real(ifft(S3value))

    EE1value = zeros(M)
    EE1value = real(ifft(E1))
    BB1value = -K_xc * n_i * 0.5 * S1
    BB2value = -K_xc * n_i * 0.5 * S2
    BB3value = -K_xc * n_i * 0.5 * S3

    Ex_energy = (1 / 2) * sum(EE1value .^ 2) * L / M
    v1node = (collect(1:N) * 2 * H / N .- H .- H / N)
    ff0value = zeros(N, M)
    ff1value = zeros(N, M)
    ff2value = zeros(N, M)
    ff3value = zeros(N, M)
    DS1value = zeros(M)
    DS2value = zeros(M)
    DS3value = zeros(M)
    for j = 1:M
        for k = 1:N
            ff0value[k, j] = 1 / 2 * (f0[k, j] * ((v1node[k]^2))) * L / M * 2 * H / N
            ff1value[k, j] = mub * f1[k, j] * (BB1value[j]) * L / M * 2 * H / N
            ff2value[k, j] = mub * f2[k, j] * (BB2value[j]) * L / M * 2 * H / N
            ff3value[k, j] = mub * f3[k, j] * (BB3value[j]) * L / M * 2 * H / N
        end
        DS1value[j] = aj0 * n_i * (SS1value[j]^2) * L / M
        DS2value[j] = aj0 * n_i * (SS2value[j]^2) * L / M
        DS3value[j] = aj0 * n_i * (SS3value[j]^2) * L / M
    end

    energykinetic = sum(sum(ff0value))
    energyBf1 = sum(sum(ff1value))
    energyBf2 = sum(sum(ff2value))
    energyBf3 = sum(sum(ff3value))
    S1energy = sum(DS1value)
    S2energy = sum(DS2value)
    S3energy = sum(DS3value)
    energy =
        Ex_energy +
        energykinetic +
        energyBf1 +
        energyBf2 +
        energyBf3 +
        S1energy +
        S2energy +
        S3energy
    Snorm = maximum(abs.((S1 .^ 2 .+ S2 .^ 2 .+ S3 .^ 2) .- 1))

    return Ex_energy

end

function main()

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
    adv = PSMAdvection(mesh)
    He = HeOperator(adv)
    H1fh = H1fhOperator(adv)

    t = Float64[]
    e = Float64[]
    push!(t, 0.0)
    push!(e, energy(f0, f1, f2, f3, E1, S1, S2, S3, mesh, tildeK))

    @showprogress 1 for i = 2:NUM # run with time 
        # Lie splitting
        @timeit to "Hv" step!(Hv, f0, f1, f2, f3, E1, h)
        @timeit to "He" step!(He, f0, f1, f2, f3, E1, h)
        @timeit to "H1fh" step!(H1fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)
        @timeit to "H2fh" H2fh!(f0, f1, f2, f3, S1, S2, S3, h, mesh, tildeK)
        @timeit to "H3fh" H3fh!(f0, f1, f2, f3, S1, S2, S3, h, mesh, tildeK)
        push!(t, (i - 1) * h)
        push!(e, energy(f0, f1, f2, f3, E1, S1, S2, S3, mesh, tildeK))
    end


    return t, e

end

t, e = main()

show(to)
plot(t, log.(e), label = "new")
eref = matread("ref.mat")["eref"]
tref = collect(eachindex(eref)) .* h
plot!(tref, log.(eref), label = "ref")
