using FFTW
using VectorSpin
using MAT
using ProgressMeter
using Test

function main(T::DataType, final_time, xmin, xmax, nx, vmin, vmax, nv, kx, dt, a, tiK)

    mesh = Mesh(T(xmin), T(xmax), nx, T(vmin), T(vmax), nv)
    @show nsteps = floor(Int, final_time / dt + 1.1) # time step number
    E1 = fft(-1.0 * a / kx * sin.(kx * mesh.x)) # electric field
    epsil = a # perturbation for S

    # spin variables
    S1 = zeros(T, nx)
    S2 = zeros(T, nx)
    S3 = zeros(T, nx)

    dS1 = zeros(T, nx)
    dS2 = zeros(T, nx)
    dS3 = zeros(T, nx)

    for k = 1:nx
        normx = sqrt(1.0 + epsil^2) #    norm of S
        dS1[k] = epsil * cos(kx * mesh.x[k]) / normx
        dS2[k] = epsil * sin(kx * mesh.x[k]) / normx
        dS3[k] = 1.0 / normx - 1.0
        S1[k] = dS1[k]
        S2[k] = dS2[k]
        S3[k] = 1.0 / normx
    end

    function maxwellian0(x, v)
        vth = 1.0
        femi = 1.0
        f = (1 / sqrt(pi) / vth) * exp(-(v / vth)^2) * (1 + a * cos(kx * x)) * femi
        return f
    end
    function maxwellian3(x, v)
        vth = 1.0
        femi = 0.5
        f = (1 / sqrt(pi) / vth) * exp(-(v / vth)^2) * (1 + a * cos(kx * x)) * femi
        return f
    end

    f0 = initialize_distribution(mesh, maxwellian0)
    f1 = zeros(T, mesh.nv, mesh.nx)
    f2 = zeros(T, mesh.nv, mesh.nx)
    f3 = initialize_distribution(mesh, maxwellian3)

    Hv = HvSubsystem(mesh)
    He = HeSubsystem(mesh)
    H1fh = H1fhSubsystem(mesh)
    H2fh = H2fhSubsystem(mesh)
    H3fh = H3fhSubsystem(mesh)

    t = T[]
    e = T[]
    push!(t, T(0.0))
    push!(e, ex_energy(E1, mesh))

    results = matread("results.mat")
    f0value = results["f0value"]
    f1value = results["f1value"]
    f2value = results["f2value"]
    f3value = results["f3value"]
    S1value = results["S1value"]
    S2value = results["S2value"]
    S3value = results["S3value"]

    @showprogress 1 for i = 1:nsteps # run with time 

        @test f0value[:, i] ≈ f0
        @test f1value[:, i] ≈ f1
        @test f2value[:, i] ≈ f2
        @test f3value[:, i] ≈ f3

        @test S1value[:, i] ≈ S1
        @test S2value[:, i] ≈ S2
        @test S3value[:, i] ≈ S3

        step!(Hv, f0, f1, f2, f3, E1, dt)
        step!(He, f0, f1, f2, f3, E1, dt)
        step!(H1fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)
        step!(H2fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)
        step!(H3fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)


        push!(t, (i - 1) * dt)
        push!(e, ex_energy(E1, mesh))

    end

    return t, e

end

T = Float64

# mesh and parameters 
final_time = 100 # final simulation time
nx = 119 # mesh number in x direction
nv = 129 # mesh number in v direction
vmin, vmax = -T(5.0), T(5.0) # computational domain [-H/2,H/2] in v
kx = T(0.5) # wave number/frequency
xmin, xmax = T(0.0), T(2π / kx) # computational domain [0,L] in x
tildeK = T(0.1598) # normalized parameter tildeK
dt = T(0.1) # time step size
a = T(0.001) # perturbation for f

t, e = main(T, final_time, xmin, xmax, nx, vmin, vmax, nv, kx, dt, a, tildeK)
