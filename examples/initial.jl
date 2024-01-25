import Pkg

using FFTW

try
    using Plots
catch
    Pkg.add("Plots")
    using Plots
end

try
    using ProgressMeter
catch
    Pkg.add("ProgressMeter")
    using ProgressMeter
end

try
    using TimerOutputs
catch
    Pkg.add("TimerOutputs")
    using TimerOutputs
end

using VectorSpin

const to = TimerOutput()

function main(T::DataType, final_time, xmin, xmax, nx, vmin, vmax, nv, kx, dt, a, tiK)

    mesh = Mesh(T(xmin), T(xmax), nx, T(vmin), T(vmax), nv)
    nsteps = floor(Int, final_time / dt + 1.1) # time step number
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

    @showprogress 1 for i = 2:nsteps # run with time 
        # Lie splitting
        @timeit to "Hv" step!(Hv, f0, f1, f2, f3, E1, dt)
        @timeit to "He" step!(He, f0, f1, f2, f3, E1, dt)
        @timeit to "H1fh" step!(H1fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)
        @timeit to "H2fh" step!(H2fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)
        @timeit to "H3fh" step!(H3fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)
        push!(t, (i - 1) * dt)
        push!(e, ex_energy(E1, mesh))
    end

    return t, e

end

T = Float64

# mesh and parameters 
final_time = 200 # final simulation time
nx = 129 # mesh number in x direction
nv = 159 # mesh number in v direction
vmin, vmax = -T(6.0), T(6.0) # computational domain [-H/2,H/2] in v
kx = T(0.5) # wave number/frequency
xmin, xmax = T(0.0), T(2π / kx) # computational domain [0,L] in x
tildeK = T(0.1598) # normalized parameter tildeK
dt = T(0.05) # time step size
a = T(0.001) # perturbation for f

t, e = main(T, final_time, xmin, xmax, nx, vmin, vmax, nv, kx, dt, a, tildeK)

show(to)
plot(t, e, label = "ex energy", yscale = :log10)
line, γ = fit_complex_frequency(t, e)
plot!(t, line, label = "γ = $(imag(γ))", legend = :bottomleft)
