using FFTW
using MAT

@testset "Spin Ions" begin

    sol = matread("onestep.mat")
    
    T = 5000 # final simulation time
    M = 119 # mesh number in x direction
    N = 129 # mesh number in v direction
    H = 10.0 / 2 # computational domain [-H/2,H/2] in v
    kkk = 0.5 # wave number/frequency
    L = 2π / kkk # computational domain [0,L] in x
    tildeK = 0.1598 # normalized parameter tildeK
    h = 0.1 # time step size()
    nsteps = 1 # floor(Int, T / h + 1.1) # time step number
    x = (0:(M-1)) .* L / M # mesh in x
    v1 = (1:N) * 2 * H / N .- H # mesh in v
    
    a = 0.001 # perturbation for f
    E1 = fft(-1.0 * a / kkk * sin.(kkk * x)) # electric field
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
        dS1[k] = epsil * cos(kkk * x[k]) / normx
        dS2[k] = epsil * sin(kkk * x[k]) / normx
        dS3[k] = 1.0 / normx - 1.0
        S1[k] = dS1[k]
        S2[k] = dS2[k]
        S3[k] = 1.0 / normx
    end
    
    @test sol["S1value"][:, 1] ≈ S1
    @test sol["S2value"][:, 1] ≈ S2
    @test sol["S3value"][:, 1] ≈ S3
    
    v1node = zeros(5N)
    for i = 1:N
        v1node[5i-4] = v1[i] - 2H / N
        v1node[5i-3] = v1[i] - (2H / N) * 3 / 4
        v1node[5i-2] = v1[i] - (2H / N) * 1 / 2
        v1node[5i-1] = v1[i] - (2H / N) * 1 / 4
        v1node[5i] = v1[i]
    end
    
    f0_value_at_node = zeros(5N, M)
    df0_value_at_node = zeros(5N, M)
    f1_value_at_node = zeros(5N, M)
    f2_value_at_node = zeros(5N, M)
    f3_value_at_node = zeros(5N, M)
    df3_value_at_node = zeros(5N, M)
    femi1 = 1
    femi2 = -1
    
    for k = 1:M, i = 1:5N
        f0_value_at_node[i, k], df0_value_at_node[i, k] =
            init(k, x, i, v1node, kkk, a, femi1, tildeK)
        f1_value_at_node[i, k] = 0.0
        f2_value_at_node[i, k] = 0.0
        f3_value_at_node[i, k], df3_value_at_node[i, k] =
            init(k, x, i, v1node, kkk, a, femi2, tildeK)
    end
    
    f0 = zeros(N, M)
    f1 = zeros(N, M)
    f2 = zeros(N, M)
    f3 = zeros(N, M)
    for k = 1:M
        f0[:, k] .= integrate(f0_value_at_node[:, k], N)
        f1[:, k] .= integrate(f1_value_at_node[:, k], N)
        f2[:, k] .= integrate(f2_value_at_node[:, k], N)
        f3[:, k] .= integrate(f3_value_at_node[:, k], N)
    end
    
    # Lie splitting
    Hv!(f0, f1, f2, f3, E1, h, M, N, L, H)
    
    hv = matread("hv.mat")
    
    @test E1 ≈ vec(hv["E1"])
    @test f0 ≈ hv["f0"]
    @test f1 ≈ hv["f1"]
    @test f2 ≈ hv["f2"]
    @test f3 ≈ hv["f3"]
    
    He!(f0, f1, f2, f3, E1, h, H)
    
    he = matread("he.mat")
    
    @test f0 ≈ he["f0"]
    @test f1 ≈ he["f1"]
    @test f2 ≈ he["f2"]
    @test f3 ≈ he["f3"]
    
    H1fh!(f0, f1, f2, f3, S1, S2, S3, h, M, N, L, H, tildeK)
    
    h1fh = matread("h1fh.mat")
    
    @test f0 ≈ h1fh["f0"]
    @test f1 ≈ h1fh["f1"]
    @test f2 ≈ h1fh["f2"]
    @test f3 ≈ h1fh["f3"]
    @test S2 ≈ vec(h1fh["S2"])
    @test S3 ≈ vec(h1fh["S3"])
    
    H2fh!(f0, f1, f2, f3, S1, S2, S3, h, M, N, L, H, tildeK)
    
    h2fh = matread("h2fh.mat")
    
    @test f0 ≈ h2fh["f0"]
    @test f1 ≈ h2fh["f1"]
    @test f2 ≈ h2fh["f2"]
    @test f3 ≈ h2fh["f3"]
    @test S1 ≈ vec(h2fh["S1"])
    @test S3 ≈ vec(h2fh["S3"])
    
    H3fh!(f0, f1, f2, f3, S1, S2, S3, h, M, N, L, H, tildeK)
   
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
