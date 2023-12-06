using GenericFFT
using MAT
using Plots
using VectorSpin
using ProgressMeter

function main(T)
    nx = 119 # mesh number in x direction
    nv = 129 # mesh number in v direction
    vmin, vmax = -5.0, 5.0 # computational domain [-H/2,H/2] in v
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
        v1 = mesh.v[i] - dv
        v2 = mesh.v[i] - dv * 0.75
        v3 = mesh.v[i] - dv * 0.50
        v4 = mesh.v[i] - dv * 0.25
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

    t = Float64[]
    push!(t, 0.0)
    e = Float64[]
    push!(e, ex_energy(E1, L, M))

    Hv = HvSubsystem(mesh)
    adv = PSMAdvection(mesh)
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
        push!(e, ex_energy(E1, L, M))
    end

    t, e

end

T = 500 # final simulation time

t, e = main(T)

plot(t, log.(e))
