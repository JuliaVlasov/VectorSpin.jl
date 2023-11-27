export H2fhOperator

struct H2fhOperator

    adv::AbstractAdvection
    partial::Vector{ComplexF64}
    v1::Vector{Float64}
    v2::Vector{Float64}
    u1::Matrix{Float64}
    u2::Matrix{Float64}
    f2::Matrix{ComplexF64}

    function H2fhOperator(adv)

        partial = zeros(ComplexF64, adv.mesh.nx)
        v1 = zeros(adv.mesh.nx)
        v2 = zeros(adv.mesh.nx)
        u1 = zeros(adv.mesh.nv, adv.mesh.nx)
        u2 = zeros(adv.mesh.nv, adv.mesh.nx)
        f2 = zeros(ComplexF64, adv.mesh.nx, adv.mesh.nv)

        new(adv, partial, v1, v2, u1, u2, f2)

    end

end


export step!

"""
compute the subsystem H2

$(SIGNATURES)

"""
function step!(op::H2fhOperator, f0, f1, f2, f3, E3, A3, dt, h_int)

    kx::Vector{Float64} = op.adv.mesh.kx
    dv::Float64 = op.adv.mesh.dv

    op.partial .= -kx .^ 2 .* A3
    ifft!(op.partial)

    op.v1 .= -h_int .* real(op.partial) ./ sqrt(3)
    op.v2 .= -op.v1

    op.u1 .= 0.5 * f0 .+ 0.5 * sqrt(3) .* f2
    op.u2 .= 0.5 * f0 .- 0.5 * sqrt(3) .* f2

    advection!(op.u1, op.adv, op.v1, dt)
    advection!(op.u2, op.adv, op.v2, dt)

    transpose!(op.f2, f2)

    op.partial .= 1im .* kx .* A3
    ifft!(op.partial)

    f0 .= op.u1 .+ op.u2
    f2 .= op.u1 ./ sqrt(3) .- op.u2 ./ sqrt(3)
    op.u1 .= cos.(dt .* real(op.partial')) .* f1 .+ sin.(dt .* real(op.partial')) .* f3
    op.u2 .= -sin.(dt .* real(op.partial')) .* f1 .+ cos.(dt .* real(op.partial')) .* f3

    fft!(op.f2, 1)
    E3 .-= dt .* h_int .* (1im .* kx) .* vec(sum(op.f2, dims = 2)) .* dv

    f1 .= op.u1
    f3 .= op.u2

end


export H2fh!


"""
$(SIGNATURES)

compute the subsystem Hs2

"""
function H2fh!(f0, f1, f2, f3, S1, S2, S3, t, mesh, tiK)

    K_xc = tiK
    n_i = 1.0
    mub = 0.3386

    partialB2 = zeros(ComplexF64, mesh.nx)
    partial2S2 = zeros(ComplexF64, mesh.nx)
    B20 = -K_xc * n_i * 0.5 * S2

    f1 .= cos.(t * B20') .* f1 .+ sin.(t * B20') .* f3
    f3 .= -sin.(t * B20') .* f1 .+ cos.(t * B20') .* f3

    fS2 = fft(S2)

    partialB2 .= (-((K_xc * n_i * 0.5 * 1im .* mesh.kx)) .* fS2)
    ifft!(partialB2)
    partial2S2 = (-((mesh.kx) .^ 2) .* fS2)
    ifft!(partial2S2)

    v1 = zeros(mesh.nx)
    v2 = zeros(mesh.nx)
    for i = 1:mesh.nx
        v1[i] = (t * real(partialB2[i]) * mub)
        v2[i] = -v1[i]
    end

    S1t = zeros(mesh.nx)
    S3t = zeros(mesh.nx)

    u1 = 0.5 .* f0 .+ 0.5 .* f2
    u2 = 0.5 .* f0 .- 0.5 .* f2

    H = 0.5 * (mesh.vmax - mesh.vmin)
    translation!(u1, v1, mesh)
    translation!(u2, v2, mesh)

    for i = 1:mesh.nx
        temi = K_xc / 4 * sum(f2[:, i]) * mesh.dv + 0.01475 * real(partial2S2[i])
        S1t[i] = cos(t * temi) * S1[i] - sin(t * temi) * S3[i]
        S3t[i] = sin(t * temi) * S1[i] + cos(t * temi) * S3[i]
    end

    f0 .= u1 .+ u2
    f2 .= u1 .- u2

    S1 .= S1t
    S3 .= S3t

end
