export H1fOperator

struct H1fOperator

    adv::AbstractAdvection
    tmp::Matrix{ComplexF64}
    expv::Matrix{ComplexF64}

    H1fOperator(adv) = new(
        adv,
        zeros(ComplexF64, adv.mesh.nx, adv.mesh.nv),
        zeros(ComplexF64, adv.mesh.nx, adv.mesh.nv),
    )

end

"""
$(SIGNATURES)

```math
\\begin{aligned}
\\dot{x} & =p \\\\
\\dot{E}_x & = - \\int (p f ) dp ds
\\end{aligned}
```

``H_p`` operator
"""
function step!(op::H1fOperator, f0, f1, f2, f3, E1, dt)

    dv::Float64 = op.adv.mesh.dv
    nx::Int = op.adv.mesh.nx
    nv::Int = op.adv.mesh.nv
    kx::Vector{Float64} = op.adv.mesh.kx
    v::Vector{Float64} = op.adv.mesh.vnode
    op.expv .= exp.(-1im .* kx .* v' .* dt)

    transpose!(op.tmp, f0)
    fft!(op.tmp, 1)

    @inbounds for i = 2:nx
        E1[i] +=
            1 / (1im * kx[i]) * sum(view(op.tmp, i, :) .* (view(op.expv, i, :) .- 1.0)) * dv
    end

    op.tmp .*= op.expv
    ifft!(op.tmp, 1)
    transpose!(f0, real(op.tmp))

    transpose!(op.tmp, f1)
    fft!(op.tmp, 1)
    op.tmp .*= op.expv
    ifft!(op.tmp, 1)
    transpose!(f1, real(op.tmp))

    transpose!(op.tmp, f2)
    fft!(op.tmp, 1)
    op.tmp .*= op.expv
    ifft!(op.tmp, 1)
    transpose!(f2, real(op.tmp))

    transpose!(op.tmp, f3)
    fft!(op.tmp, 1)
    op.tmp .*= op.expv
    ifft!(op.tmp, 1)
    transpose!(f3, real(op.tmp))

end
