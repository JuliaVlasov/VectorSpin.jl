export H1fhSubsystem

struct H1fhSubsystem{T}

    adv1::AbstractAdvection
    adv2::AbstractAdvection
    mesh::Mesh{T}
    fS1::Vector{Complex{T}}
    partial::Vector{Complex{T}}
    v1::Vector{T}
    v2::Vector{T}
    u1::Matrix{T}
    u2::Matrix{T}
    n_i::T
    mub::T

    function H1fhSubsystem(mesh::Mesh{T}; n_i = 1.0, mub = 0.3386) where {T}

        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        fS1 = zeros(Complex{T}, mesh.nx)
        partial = zeros(Complex{T}, mesh.nx)

        v1 = zeros(T, mesh.nx)
        v2 = zeros(T, mesh.nx)

        u1 = zeros(T, mesh.nv, mesh.nx)
        u2 = zeros(T, mesh.nv, mesh.nx)

        new{T}(adv1, adv2, mesh, fS1, partial, v1, v2, u1, u2, n_i, mub)

    end

end

"""
$(SIGNATURES)
"""
function step!(
    op::H1fhSubsystem{T},
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
    n_i = op.n_i
    mub = op.mub

    op.fS1 .= fft(S1)

    for i in eachindex(op.mesh.kx)
        op.partial[i] = -K_xc * n_i * 0.5 * 1im * op.mesh.kx[i] * op.fS1[i]
    end

    ifft!(op.partial)

    @inbounds for i in eachindex(op.partial)
        op.v1[i] = -real(op.partial[i]) * mub
        op.v2[i] = -op.v1[i]
    end

    @sync begin
        @spawn begin
            op.u1 .= 0.5 .* f0
            op.u1 .+= 0.5 .* f1
            advection!(op.u1, op.adv1, op.v1, dt)
        end
        @spawn begin
            op.u2 .= 0.5 .* f0
            op.u2 .-= 0.5 .* f1
            advection!(op.u2, op.adv2, op.v2, dt)
        end
    end

    @inbounds for i in eachindex(S1)
        B10 = -K_xc * n_i * 0.5 * S1[i]
        for j in axes(f2, 1)
            f2[j, i] = cos(dt * B10) * f2[j, i] - sin(dt * B10) * f3[j, i]
            f3[j, i] = sin(dt * B10) * f2[j, i] + cos(dt * B10) * f3[j, i]
        end
    end

    op.partial .= -op.mesh.kx .^ 2 .* op.fS1
    ifft!(op.partial)

    @inbounds for i in eachindex(S3)

        temi = K_xc / 4 * sum(view(f1, :, i)) * op.mesh.dv + 0.01475 * real(op.partial[i])

        op.v1[i] = cos(dt * temi) * S2[i] + sin(dt * temi) * S3[i]
        op.v2[i] = -sin(dt * temi) * S2[i] + cos(dt * temi) * S3[i]

    end

    @inbounds for i in eachindex(f0, f1)
        f0[i] = op.u1[i] + op.u2[i]
        f1[i] = op.u1[i] - op.u2[i]
    end

    copyto!(S2, op.v1)
    copyto!(S3, op.v2)

end
