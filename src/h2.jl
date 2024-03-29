export H2Subsystem

struct H2Subsystem{T}

    adv1::AbstractAdvection
    adv2::AbstractAdvection
    mesh::Mesh{T}
    partial::Vector{Complex{T}}
    v1::Vector{T}
    v2::Vector{T}
    u1::Matrix{T}
    u2::Matrix{T}
    f2::Matrix{Complex{T}}

    function H2Subsystem(mesh::Mesh{T}) where {T}

        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        partial = zeros(Complex{T}, mesh.nx)
        v1 = zeros(T, mesh.nx)
        v2 = zeros(T, mesh.nx)
        u1 = zeros(T, mesh.nv, mesh.nx)
        u2 = zeros(T, mesh.nv, mesh.nx)
        f2 = zeros(Complex{T}, mesh.nx, mesh.nv)

        new{T}(adv1, adv2, mesh, partial, v1, v2, u1, u2, f2)

    end

end

export step!

"""
compute the subsystem H2

$(SIGNATURES)

"""
function step!(
    op::H2Subsystem{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    E3::Vector{Complex{T}},
    A3::Vector{Complex{T}},
    dt::T,
    h_int,
) where {T}

    kx = op.mesh.kx
    dv = op.mesh.dv

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

    for i in eachindex(f0, f2)
        f0[i] = op.u1[i] + op.u2[i]
        f2[i] = op.u1[i] / sqrt(3) - op.u2[i] / sqrt(3)
    end

    @sync begin
        @spawn op.u1 .=
            cos.(dt .* real(op.partial')) .* f1 .+ sin.(dt .* real(op.partial')) .* f3
        @spawn op.u2 .=
            -sin.(dt .* real(op.partial')) .* f1 .+ cos.(dt .* real(op.partial')) .* f3
    end

    fft!(op.f2, 1)
    E3 .-= dt .* h_int .* (1im .* kx) .* vec(sum(op.f2, dims = 2)) .* dv

    copyto!(f1, op.u1)
    copyto!(f3, op.u2)

end
