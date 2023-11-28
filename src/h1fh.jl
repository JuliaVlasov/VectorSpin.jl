export H1fhOperator

struct H1fhOperator

    adv::AbstractAdvection
    mesh::Mesh
    B10::Vector{Float64}
    fS1::Vector{ComplexF64}
    dS1::Vector{ComplexF64}
    d2S1::Vector{ComplexF64}
    S2t::Vector{Float64}
    S3t::Vector{Float64}
    v1::Vector{Float64}
    v2::Vector{Float64}
    u1::Matrix{Float64}
    u2::Matrix{Float64}
    n_i::Float64
    mub::Float64

    function H1fhOperator(adv; n_i = 1.0, mub = 0.3386)

        mesh = adv.mesh
        B10 = zeros(mesh.nx)
        fS1 = zeros(ComplexF64, mesh.nx)
        dS1 = zeros(ComplexF64, mesh.nx)
        d2S1 = zeros(ComplexF64, mesh.nx)

        S2t = zeros(mesh.nx)
        S3t = zeros(mesh.nx)

        v1 = similar(B10)
        v2 = similar(B10)

        u1 = zeros(mesh.nv, mesh.nx)
        u2 = zeros(mesh.nv, mesh.nx)

        new(adv, mesh, B10, fS1, dS1, d2S1, S2t, S3t, v1, v2, u1, u2, n_i, mub)

    end

end

export H1fh!

"""
$(SIGNATURES)

compute the subsystem Hs1
M is even number

"""
function step!(op::H1fhOperator, f0, f1, f2, f3, S1, S2, S3, dt, tiK)

    K_xc = tiK
    n_i = op.n_i
    mub = op.mub

    op.B10 .= -K_xc * n_i * 0.5 * S1
    op.fS1 .= fft(S1)

    op.dS1 .= -(K_xc * n_i * 0.5 * 1im .* op.mesh.kx) .* op.fS1
    ifft!(op.dS1)
    for i = 1:op.mesh.nx
        op.v1[i] = -(dt * real(op.dS1[i]) * mub)
        op.v2[i] = -op.v1[i]
    end

    op.d2S1 .= -op.mesh.kx .^ 2 .* op.fS1
    ifft!(op.d2S1)

    op.u1 .= 0.5 .* f0 .+ 0.5 .* f1
    op.u2 .= 0.5 .* f0 .- 0.5 .* f1

    advection!(op.u1, op.adv, op.v1, dt)
    advection!(op.u2, op.adv, op.v2, dt)

    f2 .= cos.(dt * op.B10') .* f2 .- sin.(dt * op.B10') .* f3
    f3 .= sin.(dt * op.B10') .* f2 .+ cos.(dt * op.B10') .* f3

    for i = 1:op.mesh.nx

        temi = K_xc / 4 * sum(view(f1, :, i)) * op.mesh.dv + 0.01475 * real(op.d2S1[i])

        op.S2t[i] = cos(dt * temi) * S2[i] + sin(dt * temi) * S3[i]
        op.S3t[i] = -sin(dt * temi) * S2[i] + cos(dt * temi) * S3[i]

    end

    f0 .= op.u1 .+ op.u2
    f1 .= op.u1 .- op.u2

    S2 .= op.S2t
    S3 .= op.S3t

end
