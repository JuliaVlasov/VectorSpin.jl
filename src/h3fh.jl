export H3fhOperator

struct H3fhOperator

    adv::AbstractAdvection
    mesh::Mesh
    partial::Vector{ComplexF64}
    fS3::Vector{ComplexF64}
    v1::Vector{Float64}
    v2::Vector{Float64}
    u1::Matrix{Float64}
    u2::Matrix{Float64}
    f3::Matrix{ComplexF64}
    n_i::Float64
    mub::Float64

    function H3fhOperator(adv; n_i = 1.0, mub = 0.3386)


        mesh = adv.mesh
        nv, nx = adv.mesh.nv, adv.mesh.nx
        partial = zeros(ComplexF64, nx)
        fS3 = zeros(ComplexF64, nx)
        v1 = zeros(nx)
        v2 = zeros(nx)
        u1 = zeros(nv, nx)
        u2 = zeros(nv, nx)
        f3 = zeros(ComplexF64, nx, nv)

        new(adv, mesh, partial, fS3, v1, v2, u1, u2, f3, n_i, mub)

    end

end

"""
$(SIGNATURES)
"""
function step!(op::H3fhOperator, f0, f1, f2, f3, E2, A2, dt, h_int)

    nx::Int = op.adv.mesh.nx
    dv::Float64 = op.adv.mesh.dv
    k::Vector{Float64} = op.adv.mesh.kx

    op.partial .= -k .^ 2 .* A2
    ifft!(op.partial)
    transpose!(op.f3, f3)
    fft!(op.f3, 1)

    op.v1 .= h_int .* real(op.partial) ./ sqrt(3)
    op.v2 .= -op.v1

    op.u1 .= 0.5 .* f0 .+ 0.5 .* sqrt(3) .* f3
    op.u2 .= 0.5 .* f0 .- 0.5 .* sqrt(3) .* f3

    advection!(op.u1, op.adv, op.v1, dt)
    advection!(op.u2, op.adv, op.v2, dt)

    op.partial .= 1im .* k .* A2
    ifft!(op.partial)

    f0 .= op.u1 .+ op.u2
    f3 .= op.u1 ./ sqrt(3) .- op.u2 ./ sqrt(3)
    op.u1 .= cos.(dt .* real(op.partial')) .* f1 .+ sin.(dt .* real(op.partial')) .* f2
    op.u2 .= -sin.(dt .* real(op.partial')) .* f1 .+ cos.(dt .* real(op.partial')) .* f2

    f1 .= op.u1
    f2 .= op.u2

    @inbounds for i = 2:nx
        E2[i] += dt * h_int * 1im * k[i] * sum(view(op.f3, i, :)) * dv
    end
end

export H3fh!

"""
$(SIGNATURES)

compute the subsystem Hs3
"""
function step!(op::H3fhOperator, f0, f1, f2, f3, S1, S2, S3, dt, tiK)

    K_xc = tiK

    k = op.mesh.kx

    op.fS3 .= fft(S3)
    op.partial .= (-((K_xc * op.n_i * 0.5 * 1im * k)) .* op.fS3)
    ifft!(op.partial)

    for i = 1:op.mesh.nx
        op.v1[i] = - dt * real(op.partial[i]) * op.mub
        op.v2[i] = - op.v1[i]
    end

    op.u1 .= 0.5 * f0 .+ 0.5 * f3
    op.u2 .= 0.5 * f0 .- 0.5 * f3

    advection!(op.u1, op.adv, op.v1, dt)
    advection!(op.u2, op.adv, op.v2, dt)

    op.partial .= -(k .^ 2) .* op.fS3
    ifft!(op.partial)

    # We use v1 and v2 for new values of S1 and S2 to save memory print
    for i = 1:op.mesh.nx
        temi = K_xc / 4 * sum(view(f3, :, i)) * op.mesh.dv + 0.01475 * real(op.partial[i])
        op.v1[i] = cos(dt * temi) * S1[i] + sin(dt * temi) * S2[i]
        op.v2[i] = -sin(dt * temi) * S1[i] + cos(dt * temi) * S2[i]
    end

    f0 .= op.u1 .+ op.u2
    f3 .= op.u1 .- op.u2

    for i = eachindex(S3)
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
