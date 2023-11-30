export HAAOperator

struct HAAOperator

    mesh::Mesh
    adv0::AbstractAdvection
    adv1::AbstractAdvection
    adv2::AbstractAdvection
    adv3::AbstractAdvection
    A2::Vector{Float64}
    A3::Vector{Float64}
    dA2::Vector{ComplexF64}
    dA3::Vector{ComplexF64}
    delta::Vector{Float64}

    function HAAOperator(mesh)

        A2 = zeros(mesh.nx)
        A3 = zeros(mesh.nx)
        dA2 = zeros(ComplexF64, mesh.nx)
        dA3 = zeros(ComplexF64, mesh.nx)
        delta = zeros(mesh.nx)
        adv0 = PSMAdvection(mesh)
        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        adv3 = PSMAdvection(mesh)

        new(mesh, adv0, adv1, adv2, adv3, A2, A3, dA2, dA3, delta)

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
"""
function step!(op::HAAOperator, f0, f1, f2, f3, E2, E3, A2, A3, dt)

    nx::Int = op.mesh.nx
    kx::Vector{Float64} = op.mesh.kx
    dv::Float64 = op.mesh.dv

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
        s::Float64 = sum(view(f0, :, i))
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
