# Model with ions and spin effects


```@example ions

using FFTW
using Plots
using VectorSpin

function main(T)

    nx = 119 # mesh number in x direction
    nv = 129 # mesh number in v direction
    vmin, vmax = -5.0, 5.0 # computational domain [-H/2,H/2] in v
    kx = 0.5 # wave number/frequency
    xmin, xmax = 0.0, 2π / kx # computational domain [0,L] in x
    tildeK = 0.1598 # normalized parameter tildeK
    dt = 0.1 # time step size()
    nsteps = floor(Int, T / dt + 1.1) # time step number

    mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)

    a = 0.001 # perturbation for f
    E1 = fft(-1.0 * a / kx * sin.(kx .* mesh.x)) # electric field
    epsil = a # perturbation for S

    femi1 = 1
    femi2 = -1

    # spin variables
    S1 = zeros(nx)
    S2 = zeros(nx)
    S3 = zeros(nx)

    dS1 = zeros(nx)
    dS2 = zeros(nx)
    dS3 = zeros(nx)

    for k = 1:nx
        normx = sqrt(1 + epsil^2) #    norm of S
        dS1[k] = epsil * cos(kx * mesh.x[k]) / normx
        dS2[k] = epsil * sin(kx * mesh.x[k]) / normx
        dS3[k] = 1.0 / normx - 1.0
        S1[k] = dS1[k]
        S2[k] = dS2[k]
        S3[k] = 1.0 / normx
    end

    function maxwellian(x, v, kx, a, femi)
    
        vth = 1.0 
        femi = femi == 1 ? 1 : 0.5
        return (1 / sqrt(pi) / vth) * exp(-(v / vth)^2) * (1 + a * cos(kx * x)) * femi
    
    end

    f0 = zeros(nv, nx)
    f1 = zeros(nv, nx)
    f2 = zeros(nv, nx)
    f3 = zeros(nv, nx)
    
    for k = 1:nx, i = 1:nv
        v1 = mesh.v[i] - mesh.dv
        v2 = mesh.v[i] - mesh.dv * 0.75
        v3 = mesh.v[i] - mesh.dv * 0.50
        v4 = mesh.v[i] - mesh.dv * 0.25
        v5 = mesh.v[i]
        
        x = mesh.x[k]
    
        y1 = maxwellian(x, v1, kx, a, femi1)
        y2 = maxwellian(x, v2, kx, a, femi1)
        y3 = maxwellian(x, v3, kx, a, femi1)
        y4 = maxwellian(x, v4, kx, a, femi1)
        y5 = maxwellian(x, v5, kx, a, femi1)
    
        f0[i, k] = (7y1 + 32y2 + 12y3 + 32y4 + 7y5) / 90

        y1 = maxwellian(x, v1, kx, a, femi2)
        y2 = maxwellian(x, v2, kx, a, femi2)
        y3 = maxwellian(x, v3, kx, a, femi2)
        y4 = maxwellian(x, v4, kx, a, femi2)
        y5 = maxwellian(x, v5, kx, a, femi2)
    
        f3[i, k] = (7y1 + 32y2 + 12y3 + 32y4 + 7y5) / 90
    end

    ex_energy(E1, mesh) = sum(abs2.(ifft(E1))) * mesh.dx

    t = Float64[]
    push!(t, 0.0)
    e = Float64[]
    push!(e, ex_energy(E1, mesh))

    Hv = HvSubsystem(mesh)
    He = HeSubsystem(mesh)
    H1fh = H1fhSubsystem(mesh)
    H2fh = H2fhSubsystem(mesh)
    H3fh = H3fhSubsystem(mesh)

    for i = 1:nsteps
        step!(Hv, f0, f1, f2, f3, E1, dt)
        step!(He, f0, f1, f2, f3, E1, dt)
        step!(H1fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)
        step!(H2fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)
        step!(H3fh, f0, f1, f2, f3, S1, S2, S3, dt, tildeK)
        push!(t, i * dt)
        push!(e, ex_energy(E1, mesh))
    end

    t, e

end

T = 150 # final simulation time

t, e = main(T)

plot(t, e, label = "Ex energy", yscale = :log10)
line, g = fit_complex_frequency(t, e)
plot!(t, line, label = "g = $(imag(g))")
```
