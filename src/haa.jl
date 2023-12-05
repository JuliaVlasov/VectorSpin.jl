export HAOperator

struct HAOperator{T}

    mesh::Mesh{T}
    adv0::AbstractAdvection
    adv1::AbstractAdvection
    adv2::AbstractAdvection
    adv3::AbstractAdvection
    A2::Vector{T}
    A3::Vector{T}
    dA2::Vector{Complex{T}}
    dA3::Vector{Complex{T}}
    delta::Vector{T}

    function HAOperator(mesh::Mesh{T}) where {T}

        A2 = zeros(T, mesh.nx)
        A3 = zeros(T, mesh.nx)
        dA2 = zeros(Complex{T}, mesh.nx)
        dA3 = zeros(Complex{T}, mesh.nx)
        delta = zeros(T, mesh.nx)
        adv0 = PSMAdvection(mesh)
        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        adv3 = PSMAdvection(mesh)

        new{T}(mesh, adv0, adv1, adv2, adv3, A2, A3, dA2, dA3, delta)

    end

end

"""
$(SIGNATURES)

```math
\\begin{aligned}
\\dot{p} = (A_y, A_z) \\cdot \\partial_x (A_y, A_z)   \\\\
\\dot{Ey} = -\\partial_x^2 A_y + A_y \\rho \\\\
\\dot{Ez} = -\\partial_x^2 A_z + A_z \\rho \\\\
\\end{aligned}
```

[documentation](https://juliavlasov.github.io/VectorSpin.jl/hamiltonian_splitting.html#Subsystem-for-\\mathcal{H}_p)
"""
function step!(
    op::HAOperator{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    E2::Vector{Complex{T}},
    E3::Vector{Complex{T}},
    A2::Vector{Complex{T}},
    A3::Vector{Complex{T}},
    dt::T,
) where {T}

    nx = op.mesh.nx
    kx = op.mesh.kx
    dv = op.mesh.dv

    op.dA2 .= 1im .* kx .* A2
    ifft!(op.dA2)
    op.dA3 .= 1im .* kx .* A3
    ifft!(op.dA3)
    op.A2 .= real(ifft(A2))
    op.A3 .= real(ifft(A3))

    op.delta .= -real(op.dA2) .* op.A2 .- real(op.dA3) .* op.A3

    @inbounds for i = 2:nx
        E2[i] += dt * kx[i]^2 * A2[i]
        E3[i] += dt * kx[i]^2 * A3[i]
    end

    @inbounds for i = 1:nx
        s = sum(view(f0, :, i))
        op.A2[i] = dv * op.A2[i] * s
        op.A3[i] = dv * op.A3[i] * s
    end

    E2 .+= dt * fft(op.A2)
    E3 .+= dt * fft(op.A3)

    @sync begin
        @spawn advection!(f0, op.adv0, op.delta, dt)
        @spawn advection!(f1, op.adv1, op.delta, dt)
        @spawn advection!(f2, op.adv2, op.delta, dt)
        @spawn advection!(f3, op.adv3, op.delta, dt)
    end

end
