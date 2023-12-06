export H2fhSubsystem

struct H2fhSubsystem{T}

    adv1::AbstractAdvection
    adv2::AbstractAdvection
    mesh::Mesh{T}
    partial::Vector{Complex{T}}
    fS2::Vector{Complex{T}}
    v1::Vector{T}
    v2::Vector{T}
    u1::Matrix{T}
    u2::Matrix{T}
    f2::Matrix{Complex{T}}
    n_i::T
    mub::T

    function H2fhSubsystem(mesh::Mesh{T}; n_i = 1.0, mub = 0.3386) where {T}

        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        partial = zeros(Complex{T}, mesh.nx)
        fS2 = zeros(Complex{T}, mesh.nx)
        v1 = zeros(T, mesh.nx)
        v2 = zeros(T, mesh.nx)
        u1 = zeros(T, mesh.nv, mesh.nx)
        u2 = zeros(T, mesh.nv, mesh.nx)
        f2 = zeros(Complex{T}, mesh.nx, mesh.nv)

        new{T}(adv1, adv2, mesh, partial, fS2, v1, v2, u1, u2, f2, n_i, mub)

    end

end

export step!

"""
$(SIGNATURES)

compute the subsystem Hs2

"""
function step!(
    op::H2fhSubsystem{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    S1::Vector{T},
    S2::Vector{T},
    S3::Vector{T},
    dt::T,
    tiK,
) where {T}

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
