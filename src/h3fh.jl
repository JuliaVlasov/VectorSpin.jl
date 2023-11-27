export H3fhOperator

struct H3fhOperator

    adv::AbstractAdvection
    partial::Vector{ComplexF64}
    v1::Vector{Float64}
    v2::Vector{Float64}
    u1::Matrix{Float64}
    u2::Matrix{Float64}
    f3::Matrix{ComplexF64}

    function H3fhOperator(adv)

        nv, nx = adv.mesh.nv, adv.mesh.nx
        partial = zeros(ComplexF64, nx)
        v1 = zeros(nx)
        v2 = zeros(nx)
        u1 = zeros(nv, nx)
        u2 = zeros(nv, nx)
        f3 = zeros(ComplexF64, nx, nv)

        new(adv, partial, v1, v2, u1, u2, f3)

    end

end

"""
$(SIGNATURES)
"""
function step!(op :: H3fhOperator, f0, f1, f2, f3, E2, A2, dt, h_int)

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
function H3fh!(f0, f1, f2, f3, S1, S2, S3, t, mesh, tiK)

    K_xc = tiK
    n_i = 1.0
    mub = 0.3386

    k = mesh.kx

    B30 = -K_xc * n_i * 0.5 * S3
    fS3 = fft(S3)
    partialB3 = (-((K_xc * n_i * 0.5 * 1im * k)) .* fS3)
    ifft!(partialB3)
    partial2S3 = -(k .^ 2) .* fS3
    ifft!(partial2S3)

    S1t = zeros(mesh.nx)
    S2t = zeros(mesh.nx)
    for i = 1:mesh.nx
        temi = K_xc / 4 * sum(view(f3, :, i)) * mesh.dv + 0.01475 * real(partial2S3[i])
        S1t[i] = cos(t * temi) * S1[i] + sin(t * temi) * S2[i]
        S2t[i] = -sin(t * temi) * S1[i] + cos(t * temi) * S2[i]
    end

    v1 = zeros(mesh.nx)
    v2 = zeros(mesh.nx)
    for i = 1:mesh.nx
        v1[i] = (t * real(partialB3[i]) * mub)
        v2[i] = -v1[i]
    end

    u1 = 0.5 * f0 .+ 0.5 * f3
    u2 = 0.5 * f0 .- 0.5 * f3

    H = 0.5 * (mesh.vmax - mesh.vmin)
    translation!(u1, v1, mesh)
    translation!(u2, v2, mesh)

    f0 .= u1 .+ u2
    f3 .= u1 .- u2

    for i = 1:mesh.nx, j = 1:mesh.nv
        u1[j, i] = cos(t * B30[i]) * f1[j, i] - sin(t * B30[i]) .* f2[j, i]
        u2[j, i] = sin(t * B30[i]) * f1[j, i] + cos(t * B30[i]) .* f2[j, i]
    end

    f1 .= u1
    f2 .= u2

    S1 .= S1t
    S2 .= S2t

end
