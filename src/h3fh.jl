export H3fhSubsystem

struct H3fhSubsystem{T}

    adv1::AbstractAdvection
    adv2::AbstractAdvection
    mesh::Mesh{T}
    partial::Vector{Complex{T}}
    fS3::Vector{Complex{T}}
    v1::Vector{T}
    v2::Vector{T}
    u1::Matrix{T}
    u2::Matrix{T}
    n_i::T
    mub::T

    function H3fhSubsystem(mesh::Mesh{T}; n_i = 1.0, mub = 0.3386) where {T}


        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        nv, nx = mesh.nv, mesh.nx
        partial = zeros(Complex{T}, nx)
        fS3 = zeros(Complex{T}, nx)
        v1 = zeros(T, nx)
        v2 = zeros(T, nx)
        u1 = zeros(T, nv, nx)
        u2 = zeros(T, nv, nx)

        new{T}(adv1, adv2, mesh, partial, fS3, v1, v2, u1, u2, n_i, mub)

    end

end

"""
$(SIGNATURES)

compute the subsystem Hs3
"""
function step!(
    op::H3fhSubsystem{T},
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

    k = op.mesh.kx

    op.fS3 .= fft(S3)
    op.partial .= (-((K_xc * op.n_i * 0.5 * 1im * k)) .* op.fS3)
    ifft!(op.partial)

    @inbounds for i = eachindex(op.v1, op.v2)
        op.v1[i] = -dt * real(op.partial[i]) * op.mub
        op.v2[i] = -op.v1[i]
        op.partial[i] = -(k[i]^2) * op.fS3[i]
    end

    ifft!(op.partial)

    @sync begin
        @spawn begin
            @inbounds for i in eachindex(f0, f3)
                op.u1[i] = 0.5 * f0[i] + 0.5 * f3[i]
            end
            advection!(op.u1, op.adv1, op.v1, dt)
        end

        @spawn begin
            @inbounds for i in eachindex(f0, f3)
                op.u2[i] = 0.5 * f0[i] - 0.5 * f3[i]
            end
            advection!(op.u2, op.adv2, op.v2, dt)
        end
    end


    # We use v1 and v2 for new values of S1 and S2 to save memory print
    @inbounds for i = 1:op.mesh.nx
        temi = K_xc / 4 * sum(view(f3, :, i)) * op.mesh.dv + 0.01475 * real(op.partial[i])
        op.v1[i] = cos(dt * temi) * S1[i] + sin(dt * temi) * S2[i]
        op.v2[i] = -sin(dt * temi) * S1[i] + cos(dt * temi) * S2[i]
    end

    @inbounds for i in eachindex(f0, f3)
        f0[i] = op.u1[i] + op.u2[i]
        f3[i] = op.u1[i] - op.u2[i]
    end

    @inbounds for i in eachindex(S3)
        B30 = -K_xc * op.n_i * 0.5 * S3[i]
        for j = 1:op.mesh.nv
            op.u1[j, i] = cos(dt * B30) * f1[j, i] - sin(dt * B30) .* f2[j, i]
            op.u2[j, i] = sin(dt * B30) * f1[j, i] + cos(dt * B30) .* f2[j, i]
        end
    end

    copyto!(f1, op.u1)
    copyto!(f2, op.u2)

    copyto!(S1, op.v1)
    copyto!(S2, op.v2)

end
