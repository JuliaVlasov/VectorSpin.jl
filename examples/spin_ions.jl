using FFTW
using MAT
using Plots
using VectorSpin
using ProgressMeter

function main(T)
    M = 119 # mesh number in x direction
    N = 149 # mesh number in v direction
    H = 6.0 # computational domain [-H/2,H/2] in v
    kx = 0.5 # wave number/frequency
    L = 2π / kx # computational domain [0,L] in x
    tildeK = 0.1598 # normalized parameter tildeK
    h = 0.1 # time step size()
    nsteps = floor(Int, T / h + 1.1) # time step number

    mesh = Mesh(0.0, L, M, -H, H, N)

    a = 0.001 # perturbation for f
    E1 = fft(-1.0 * a / kx * sin.(kx .* mesh.x)) # electric field
    epsil = a # perturbation for S

    # spin variables
    S1 = zeros(M)
    S2 = zeros(M)
    S3 = zeros(M)

    dS1 = zeros(M)
    dS2 = zeros(M)
    dS3 = zeros(M)

    for k = 1:M
        normx = sqrt(1 + epsil^2) #    norm of S
        dS1[k] = epsil * cos(kx * mesh.x[k]) / normx
        dS2[k] = epsil * sin(kx * mesh.x[k]) / normx
        dS3[k] = 1.0 / normx - 1.0
        S1[k] = dS1[k]
        S2[k] = dS2[k]
        S3[k] = 1.0 / normx
    end

    femi1 = 1
    femi2 = -1

    xmin, xmax = mesh.xmin, mesh.xmax
    vmin, vmax = mesh.vmin, mesh.vmax
    nx, nv = mesh.nx, mesh.nv
    dv = mesh.dv

    function maxwellian0(x, v)
        vth = 1.0
        femi = 1
        return (1 / sqrt(pi) / vth) * exp(-(v / vth)^2) * (1 + a * cos(kx * x)) * femi
    end

    function maxwellian3(x, v)
        vth = 1.0
        femi = 0.5
        return (1 / sqrt(pi) / vth) * exp(-(v / vth)^2) * (1 + a * cos(kx * x)) * femi
    end

    f0 = initialize_distribution(mesh, maxwellian0)
    f1 = zeros(nv, nx)
    f2 = zeros(nv, nx)
    f3 = initialize_distribution(mesh, maxwellian3)

    t = Float64[]
    push!(t, 0.0)
    e = Float64[]
    push!(e, ex_energy(E1, mesh))

    Hv = HvSubsystem(mesh)
    He = HeSubsystem(mesh)
    H1fh = H1fhSubsystem(mesh)
    H2fh = H2fhSubsystem(mesh)
    H3fh = H3fhSubsystem(mesh)

    @showprogress 1 for i = 1:nsteps
        step!(Hv, f0, f1, f2, f3, E1, h)
        step!(He, f0, f1, f2, f3, E1, h)
        step!(H1fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)
        step!(H2fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)
        step!(H3fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)
        push!(t, i * h)
        push!(e, ex_energy(E1, mesh))
    end

    t, e

end

T = 500 # final simulation time

t, e = main(T)

plot(t, e, yscale = :log10)
