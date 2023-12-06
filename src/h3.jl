export H3Subsystem

struct H3Subsystem{T}

    adv1::AbstractAdvection
    adv2::AbstractAdvection
    mesh::Mesh{T}
    partial::Vector{Complex{T}}
    v1::Vector{T}
    v2::Vector{T}
    u1::Matrix{T}
    u2::Matrix{T}
    f3::Matrix{Complex{T}}

    function H3Subsystem(mesh::Mesh{T}) where {T}


        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        nv, nx = mesh.nv, mesh.nx
        partial = zeros(Complex{T}, nx)
        v1 = zeros(T, nx)
        v2 = zeros(T, nx)
        u1 = zeros(T, nv, nx)
        u2 = zeros(T, nv, nx)
        f3 = zeros(Complex{T}, nx, nv)

        new{T}(adv1, adv2, mesh, partial, v1, v2, u1, u2, f3)

    end

end

"""
$(SIGNATURES)
"""
function step!(
    op::H3Subsystem{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    E2::Vector{Complex{T}},
    A2::Vector{Complex{T}},
    dt::T,
    h_int,
) where {T}

    nx = op.mesh.nx
    dv = op.mesh.dv
    k = op.mesh.kx

    op.partial .= -k .^ 2 .* A2
    ifft!(op.partial)
    transpose!(op.f3, f3)
    fft!(op.f3, 1)

    @sync begin
        @spawn begin
            op.v1 .= h_int .* real(op.partial) ./ sqrt(3)
            op.u1 .= 0.5 .* f0 .+ 0.5 .* sqrt(3) .* f3
            advection!(op.u1, op.adv1, op.v1, dt)
        end

        @spawn begin
            op.v2 .= -h_int .* real(op.partial) ./ sqrt(3)
            op.u2 .= 0.5 .* f0 .- 0.5 .* sqrt(3) .* f3
            advection!(op.u2, op.adv2, op.v2, dt)
        end
    end

    op.partial .= 1im .* k .* A2
    ifft!(op.partial)

    f0 .= op.u1 .+ op.u2
    f3 .= op.u1 ./ sqrt(3) .- op.u2 ./ sqrt(3)

    @sync begin
        @spawn op.u1 .= cos.(dt .* real(op.partial')) .* f1 .+ sin.(dt .* real(op.partial')) .* f2
        @spawn op.u2 .= -sin.(dt .* real(op.partial')) .* f1 .+ cos.(dt .* real(op.partial')) .* f2
    end

    copyto!(f1, op.u1)
    copyto!(f2, op.u2)

    @inbounds for i = 2:nx
        E2[i] += dt * h_int * 1im * k[i] * sum(view(op.f3, i, :)) * dv
    end
end

