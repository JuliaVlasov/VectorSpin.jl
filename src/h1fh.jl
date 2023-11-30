export H1fhOperator

struct H1fhOperator

    adv1::AbstractAdvection
    adv2::AbstractAdvection
    mesh::Mesh
    fS1::Vector{ComplexF64}
    partial::Vector{ComplexF64}
    v1::Vector{Float64}
    v2::Vector{Float64}
    u1::Matrix{Float64}
    u2::Matrix{Float64}
    n_i::Float64
    mub::Float64

    function H1fhOperator(mesh; n_i = 1.0, mub = 0.3386)

        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        fS1 = zeros(ComplexF64, mesh.nx)
        partial = zeros(ComplexF64, mesh.nx)

        v1 = zeros(mesh.nx)
        v2 = zeros(mesh.nx)

        u1 = zeros(mesh.nv, mesh.nx)
        u2 = zeros(mesh.nv, mesh.nx)

        new(adv1, adv2, mesh, fS1, partial, v1, v2, u1, u2, n_i, mub)

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

    op.fS1 .= fft(S1)

    op.partial .= -K_xc * n_i * 0.5 * 1im .* op.mesh.kx .* op.fS1

    ifft!(op.partial)

    for i in eachindex(op.partial)
        op.v1[i] = -real(op.partial[i]) * mub
        op.v2[i] = -op.v1[i]
    end

    @sync begin
        @spawn begin
            op.u1 .= 0.5 .* f0 .+ 0.5 .* f1
            advection!(op.u1, op.adv1, op.v1, dt)
        end
        @spawn begin
            op.u2 .= 0.5 .* f0 .- 0.5 .* f1
            advection!(op.u2, op.adv2, op.v2, dt)
        end
    end

    for i in eachindex(S1)
        B10 = -K_xc * n_i * 0.5 * S1[i]
        for j in axes(f2, 1)
            f2[j, i] = cos(dt * B10) * f2[j, i] - sin(dt * B10) * f3[j, i]
            f3[j, i] = sin(dt * B10) * f2[j, i] + cos(dt * B10) * f3[j, i]
        end
    end

    op.partial .= -op.mesh.kx .^ 2 .* op.fS1
    ifft!(op.partial)

    for i in eachindex(S3)

        temi = K_xc / 4 * sum(view(f1, :, i)) * op.mesh.dv + 0.01475 * real(op.partial[i])

        op.v1[i] = cos(dt * temi) * S2[i] + sin(dt * temi) * S3[i]
        op.v2[i] = -sin(dt * temi) * S2[i] + cos(dt * temi) * S3[i]

    end

    for i in eachindex(f0, f1)
        f0[i] = op.u1[i] + op.u2[i]
        f1[i] = op.u1[i] - op.u2[i]
    end

    copyto!(S2, op.v1)
    copyto!(S3, op.v2)

end
