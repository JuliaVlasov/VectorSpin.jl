using FFTW
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


    xmin, xmax = 0.0, L
    vmin, vmax = -H, H
    nx, nv = M, N
    mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)

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
    f1 = zeros(mesh.nv, mesh.nx)
    f2 = zeros(mesh.nv, mesh.nx)
    f3 = initialize_distribution(mesh, maxwellian3)

    Hv = HvSubsystem(mesh)
    He = HeSubsystem(mesh)
    H1fh = H1fhSubsystem(mesh)
    H2fh = H2fhSubsystem(mesh)
    H3fh = H3fhSubsystem(mesh)

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
