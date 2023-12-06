import Pkg

using GenericFFT

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

try
    using DoubleFloats
catch
    Pkg.add("DoubleFloats")
    using DoubleFloats
end

using VectorSpin

const to = TimerOutput()

ex_energy(E1, mesh) = sum(abs2.(ifft(E1))) * mesh.dx

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

    function maxwellian(x, v, kx, a, femi, tiK)
    
        vth = 1.0 
    
        femi = femi == 1 ? 1 : 0.5
    
        f = (1 / sqrt(pi) / vth) * exp(-(v / vth)^2) * (1 + a * cos(kx * x)) * femi
    
        return f
    end

    function initialize(mesh::Mesh{T}, kx, a, femi1, femi2, tiK) where {T}
    
        xmin, xmax = mesh.xmin, mesh.xmax
        vmin, vmax = mesh.vmin, mesh.vmax
        nx, nv = mesh.nx, mesh.nv
        dv = mesh.dv
    
        f0 = zeros(T, nv, nx)
        f1 = zeros(T, nv, nx)
        f2 = zeros(T, nv, nx)
        f3 = zeros(T, nv, nx)
    
        for k = 1:nx, i = 1:nv
            v1 = mesh.v[i] - dv
            v2 = mesh.v[i] - dv * 0.75
            v3 = mesh.v[i] - dv * 0.50
            v4 = mesh.v[i] - dv * 0.25
            v5 = mesh.v[i]
            
            x = mesh.x[k]
    
            y1 = maxwellian(x, v1, kx, a, femi1, tiK)
            y2 = maxwellian(x, v2, kx, a, femi1, tiK)
            y3 = maxwellian(x, v3, kx, a, femi1, tiK)
            y4 = maxwellian(x, v4, kx, a, femi1, tiK)
            y5 = maxwellian(x, v5, kx, a, femi1, tiK)
    
            f0[i, k] = (7y1 + 32y2 + 12y3 + 32y4 + 7y5) / 90

            y1 = maxwellian(x, v1, kx, a, femi2, tiK)
            y2 = maxwellian(x, v2, kx, a, femi2, tiK)
            y3 = maxwellian(x, v3, kx, a, femi2, tiK)
            y4 = maxwellian(x, v4, kx, a, femi2, tiK)
            y5 = maxwellian(x, v5, kx, a, femi2, tiK)
    
            f3[i, k] = (7y1 + 32y2 + 12y3 + 32y4 + 7y5) / 90
        end
    
        return f0, f1, f2, f3
    
    end

    femi1 = 1
    femi2 = -1

    f0, f1, f2, f3 = initialize(mesh, kx, a, femi1, femi2, tiK)

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
nx = 119 # mesh number in x direction
nv = 129 # mesh number in v direction
vmin, vmax = -T(5.0), T(5.0) # computational domain [-H/2,H/2] in v
kx = T(0.5) # wave number/frequency
xmin, xmax = T(0.0), T(2π / kx) # computational domain [0,L] in x
tildeK = T(0.1598) # normalized parameter tildeK
dt = T(0.1) # time step size
a = T(0.001) # perturbation for f

t, e = main(T, final_time, xmin, xmax, nx, vmin, vmax, nv, kx, dt, a, tildeK)

show(to)
plot(t, e, label = "ex energy", yscale = :log10)
line, γ = fit_complex_frequency(t, e)
plot!(t, line, label = "γ = $(imag(γ))", legend = :bottomleft)
