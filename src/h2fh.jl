export H2fhOperator

struct H2fhOperator

    adv1::AbstractAdvection
    adv2::AbstractAdvection
    mesh::Mesh
    partial::Vector{ComplexF64}
    fS2::Vector{ComplexF64}
    v1::Vector{Float64}
    v2::Vector{Float64}
    u1::Matrix{Float64}
    u2::Matrix{Float64}
    f2::Matrix{ComplexF64}
    n_i::Float64
    mub::Float64

    function H2fhOperator(mesh; n_i = 1.0, mub = 0.3386)

        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        partial = zeros(ComplexF64, mesh.nx)
        fS2 = zeros(ComplexF64, mesh.nx)
        v1 = zeros(mesh.nx)
        v2 = zeros(mesh.nx)
        u1 = zeros(mesh.nv, mesh.nx)
        u2 = zeros(mesh.nv, mesh.nx)
        f2 = zeros(ComplexF64, mesh.nx, mesh.nv)

        new(adv1, adv2, mesh, partial, fS2, v1, v2, u1, u2, f2, n_i, mub)

    end

end

export step!

"""
compute the subsystem H2

$(SIGNATURES)

"""
function step!(op::H2fhOperator, f0, f1, f2, f3, E3, A3, dt, h_int)

    kx::Vector{Float64} = op.mesh.kx
    dv::Float64 = op.mesh.dv

    op.partial .= -kx .^ 2 .* A3
    ifft!(op.partial)

    op.v1 .= -h_int .* real(op.partial) ./ sqrt(3)
    op.v2 .= -op.v1

    @sync begin

        @spawn begin
            op.u1 .= 0.5 * f0 .+ 0.5 * sqrt(3) .* f2
            advection!(op.u1, op.adv1, op.v1, dt)
        end

        @spawn begin
            op.u2 .= 0.5 * f0 .- 0.5 * sqrt(3) .* f2
            advection!(op.u2, op.adv2, op.v2, dt)
        end

    end

    transpose!(op.f2, f2)

    op.partial .= 1im .* kx .* A3
    ifft!(op.partial)

    f0 .= op.u1 .+ op.u2
    f2 .= op.u1 ./ sqrt(3) .- op.u2 ./ sqrt(3)
    op.u1 .= cos.(dt .* real(op.partial')) .* f1 .+ sin.(dt .* real(op.partial')) .* f3
    op.u2 .= -sin.(dt .* real(op.partial')) .* f1 .+ cos.(dt .* real(op.partial')) .* f3

    fft!(op.f2, 1)
    E3 .-= dt .* h_int .* (1im .* kx) .* vec(sum(op.f2, dims = 2)) .* dv

    copyto!(f1, op.u1)
    copyto!(f3, op.u2)

end

"""
$(SIGNATURES)

compute the subsystem Hs2

"""
function step!(op::H2fhOperator, f0, f1, f2, f3, S1, S2, S3, dt, tiK)

    K_xc = tiK

    op.u1 .= f1
    op.u2 .= f3

    for i in eachindex(S2)
        B20 = -K_xc * op.n_i * 0.5 * S2[i]
        for j = 1:op.mesh.nv
            f1[j, i] = cos(dt * B20) * op.u1[j, i] + sin(dt * B20) * op.u2[j, i]
            f3[j, i] = -sin(dt * B20) * op.u1[j, i] + cos(dt * B20) * op.u2[j, i]
        end
    end

    op.fS2 .= fft(S2)

    op.partial .= -K_xc * op.n_i * 0.5 * 1im .* op.mesh.kx .* op.fS2
    ifft!(op.partial)

    for i = 1:op.mesh.nx
        op.v1[i] = -real(op.partial[i]) * op.mub
        op.v2[i] = -op.v1[i]
    end

    op.partial .= -op.mesh.kx .^ 2 .* op.fS2
    ifft!(op.partial)

    @sync begin

        @spawn begin
            op.u1 .= 0.5 .* f0 .+ 0.5 .* f2
            advection!(op.u1, op.adv1, op.v1, dt)
        end

        @spawn begin
            op.u2 .= 0.5 .* f0 .- 0.5 .* f2
            advection!(op.u2, op.adv2, op.v2, dt)
        end

    end

    # v1 and v2 are used vor new values of S1 and S2 to reduce memeory print
    for i = 1:op.mesh.nx
        temi = K_xc / 4 * sum(view(f2, :, i)) * op.mesh.dv + 0.01475 * real(op.partial[i])
        op.v1[i] = cos(dt * temi) * S1[i] - sin(dt * temi) * S3[i]
        op.v2[i] = sin(dt * temi) * S1[i] + cos(dt * temi) * S3[i]
    end

    f0 .= op.u1 .+ op.u2
    f2 .= op.u1 .- op.u2

    copyto!(S1, op.v1)
    copyto!(S3, op.v2)

end
