using GenericFFT
using MAT
using VectorSpin

@testset "Spin Ions" begin

    sol = matread("onestep.mat")

    T = 5000 # final simulation time
    M = 119 # mesh number in x direction
    N = 129 # mesh number in v direction
    H = 10.0 / 2 # computational domain [-H/2,H/2] in v
    kx = 0.5 # wave number/frequency
    L = 2π / kx # computational domain [0,L] in x
    tildeK = 0.1598 # normalized parameter tildeK
    h = 0.1 # time step size()
    nsteps = 1 # floor(Int, T / h + 1.1) # time step number
    x = (0:(M-1)) .* L / M # mesh in x
    v1 = (1:N) * 2 * H / N .- H # mesh in v

    a = 0.001 # perturbation for f
    E1 = fft(-1.0 * a / kx * sin.(kx * x)) # electric field
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
        dS1[k] = epsil * cos(kx * x[k]) / normx
        dS2[k] = epsil * sin(kx * x[k]) / normx
        dS3[k] = 1.0 / normx - 1.0
        S1[k] = dS1[k]
        S2[k] = dS2[k]
        S3[k] = 1.0 / normx
    end

    @test sol["S1value"][:, 1] ≈ S1
    @test sol["S2value"][:, 1] ≈ S2
    @test sol["S3value"][:, 1] ≈ S3


    function maxwellian(x, v, kx, a, femi)
    
        vth = 1.0 
    
        femi = femi == 1 ? 1 : 0.5
    
        f = (1 / sqrt(pi) / vth) * exp(-(v / vth)^2) * (1 + a * cos(kx * x)) * femi
    
        return f
    end

    xmin, xmax = 0.0, L
    vmin, vmax = -H, H
    nx, nv = M, N
    mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)

    femi1 = 1
    femi2 = -1

    xmin, xmax = mesh.xmin, mesh.xmax
    vmin, vmax = mesh.vmin, mesh.vmax
    nx, nv = mesh.nx, mesh.nv
    dv = mesh.dv
    
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
    
    Hv = HvOperator(mesh)
    He = HeOperator(mesh)
    H1fh = H1fhOperator(mesh)
    H2fh = H2fhOperator(mesh)
    H3fh = H3fhOperator(mesh)

    # Lie splitting
    # Lie splitting
    step!(Hv, f0, f1, f2, f3, E1, h)

    hv = matread("hv.mat")

    @test E1 ≈ vec(hv["E1"])
    @test f0 ≈ hv["f0"]
    @test f1 ≈ hv["f1"]
    @test f2 ≈ hv["f2"]
    @test f3 ≈ hv["f3"]

    step!(He, f0, f1, f2, f3, E1, h)

    he = matread("he.mat")

    @test f0 ≈ he["f0"]
    @test f1 ≈ he["f1"]
    @test f2 ≈ he["f2"]
    @test f3 ≈ he["f3"]

    step!(H1fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)

    h1fh = matread("h1fh.mat")

    @test f0 ≈ h1fh["f0"]
    @test f1 ≈ h1fh["f1"]
    @test f2 ≈ h1fh["f2"]
    @test f3 ≈ h1fh["f3"]
    @test S2 ≈ vec(h1fh["S2"])
    @test S3 ≈ vec(h1fh["S3"])

    step!(H2fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)

    h2fh = matread("h2fh.mat")

    @test f0 ≈ h2fh["f0"]
    @test f1 ≈ h2fh["f1"]
    @test f2 ≈ h2fh["f2"]
    @test f3 ≈ h2fh["f3"]
    @test S1 ≈ vec(h2fh["S1"])
    @test S3 ≈ vec(h2fh["S3"])

    step!(H3fh, f0, f1, f2, f3, S1, S2, S3, h, tildeK)

    h3fh = matread("h3fh.mat")

    @test f0 ≈ h3fh["f0"]
    @test f1 ≈ h3fh["f1"]
    @test f2 ≈ h3fh["f2"]
    @test f3 ≈ h3fh["f3"]
    @test S1 ≈ vec(h3fh["S1"])
    @test S2 ≈ vec(h3fh["S2"])

    @test sol["S1value"][:, end] ≈ S1
    @test sol["S2value"][:, end] ≈ S2
    @test sol["S3value"][:, end] ≈ S3

end
