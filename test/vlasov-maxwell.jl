using FFTW
using MAT

import VectorSpin: initialfunction, initialfields

const M = 129   # partition of x
const N = 129   # partition of v
const H = 5.0 / 2   # v domain size()
const kkk = 1.2231333040331807  #ke
const L = 4pi / kkk  # x domain size()
const h = 0.04 #time step size()
const a = 0.02 # 0.001; perturbation coefficient
const h_int = 0.2 # hbar
const k0 = 2.0 * kkk
const ww = sqrt(1.0 + k0^2.0) # w0
const ata = 0.2
const kk = 0.17 # v_th

@testset "Vlasov-Maxwell" begin

    mesh = Mesh(0, L, M, -H, H, N)
    adv = PSMAdvection(mesh)

    E1, E2, E3, A2, A3 = initialfields( mesh, a, ww, kkk, k0)
    f0, f1, f2, f3 = initialfunction(mesh, a, kkk, kk, ata)

    fields = matread("fields0.mat")

    @test E1 ≈ fields["E1"]
    @test E2 ≈ fields["E2"]
    @test E3 ≈ fields["E3"]
    @test A2 ≈ fields["A2"]
    @test A3 ≈ fields["A3"]

    df = matread("df0.mat")

    @test f0 ≈ df["f0"]
    @test f1 ≈ df["f1"]
    @test f2 ≈ df["f2"]
    @test f3 ≈ df["f3"]

    H2fh = H2fhOperator(mesh)
    He = HeOperator(mesh)
    HAA = HAAOperator(mesh)
    H3fh = H3fhOperator(mesh)
    H1f = H1fOperator(mesh)

    step!(H2fh, f0, f1, f2, f3, E3, A3, h / 2, h_int)

    fields = matread("fields1.mat")
    df = matread("df1.mat")

    @test E1 ≈ fields["E1"]
    @test E2 ≈ fields["E2"]
    @test E3 ≈ fields["E3"]
    @test A2 ≈ fields["A2"]
    @test A3 ≈ fields["A3"]
    @test f0 ≈ df["f0"]
    @test f1 ≈ df["f1"]
    @test f2 ≈ df["f2"]
    @test f3 ≈ df["f3"]

    step!(He, f0, f1, f2, f3, E1, E2, E3, A2, A3, h / 2)

    fields = matread("fields2.mat")
    df = matread("df2.mat")

    @test E1 ≈ fields["E1"]
    @test E2 ≈ fields["E2"]
    @test E3 ≈ fields["E3"]
    @test A2 ≈ fields["A2"]
    @test A3 ≈ fields["A3"]
    @test f0 ≈ df["f0"]
    @test f1 ≈ df["f1"]
    @test f2 ≈ df["f2"]
    @test f3 ≈ df["f3"]

    step!(HAA, f0, f1, f2, f3, E2, E3, A2, A3, h / 2)

    fields = matread("fields3.mat")
    df = matread("df3.mat")

    @test E1 ≈ fields["E1"]
    @test E2 ≈ fields["E2"]
    @test E3 ≈ fields["E3"]
    @test A2 ≈ fields["A2"]
    @test A3 ≈ fields["A3"]
    @test f0 ≈ df["f0"]
    @test f1 ≈ df["f1"]
    @test f2 ≈ df["f2"]
    @test f3 ≈ df["f3"]

    step!(H3fh, f0, f1, f2, f3, E2, A2, h / 2, h_int)

    fields = matread("fields4.mat")
    df = matread("df4.mat")

    @test E1 ≈ fields["E1"]
    @test E2 ≈ fields["E2"]
    @test E3 ≈ fields["E3"]
    @test A2 ≈ fields["A2"]
    @test A3 ≈ fields["A3"]
    @test f0 ≈ df["f0"]
    @test f1 ≈ df["f1"]
    @test f2 ≈ df["f2"]
    @test f3 ≈ df["f3"]

    step!(H1f, f0, f1, f2, f3, E1, h)

    fields = matread("fields5.mat")
    df = matread("df5.mat")

    @test E1 ≈ fields["E1"]
    @test E2 ≈ fields["E2"]
    @test E3 ≈ fields["E3"]
    @test A2 ≈ fields["A2"]
    @test A3 ≈ fields["A3"]
    @test f0 ≈ df["f0"]
    @test f1 ≈ df["f1"]
    @test f2 ≈ df["f2"]
    @test f3 ≈ df["f3"]

end
