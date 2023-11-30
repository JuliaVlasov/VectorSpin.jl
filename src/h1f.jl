export H1fOperator

struct H1fOperator

    mesh::Mesh
    ff0::Matrix{ComplexF64}
    ff1::Matrix{ComplexF64}
    ff2::Matrix{ComplexF64}
    ff3::Matrix{ComplexF64}
    expv::Matrix{ComplexF64}

    function H1fOperator(mesh)

        ff0 = zeros(ComplexF64, mesh.nx, mesh.nv)
        ff1 = zeros(ComplexF64, mesh.nx, mesh.nv)
        ff2 = zeros(ComplexF64, mesh.nx, mesh.nv)
        ff3 = zeros(ComplexF64, mesh.nx, mesh.nv)
        expv = zeros(ComplexF64, mesh.nx, mesh.nv)

        new(mesh, ff0, ff1, ff2, ff3, expv)

    end

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

    dv::Float64 = op.mesh.dv
    nx::Int = op.mesh.nx
    nv::Int = op.mesh.nv
    kx::Vector{Float64} = op.mesh.kx
    v::Vector{Float64} = op.mesh.vnode
    op.expv .= exp.(-1im .* kx .* v' .* dt)


    @sync begin

        @spawn begin
            transpose!(op.ff0, f0)
            fft!(op.ff0, 1)

            @inbounds for i = 2:nx
                E1[i] +=
                    1 / (1im * kx[i]) *
                    sum(view(op.ff0, i, :) .* (view(op.expv, i, :) .- 1.0)) *
                    dv
            end

            op.ff0 .*= op.expv
            ifft!(op.ff0, 1)
            transpose!(f0, real(op.ff0))
        end

        @spawn begin
            transpose!(op.ff1, f1)
            fft!(op.ff1, 1)
            op.ff1 .*= op.expv
            ifft!(op.ff1, 1)
            transpose!(f1, real(op.ff1))
        end

        @spawn begin
            transpose!(op.ff2, f2)
            fft!(op.ff2, 1)
            op.ff2 .*= op.expv
            ifft!(op.ff2, 1)
            transpose!(f2, real(op.ff2))
        end

        @spawn begin
            transpose!(op.ff3, f3)
            fft!(op.ff3, 1)
            op.ff3 .*= op.expv
            ifft!(op.ff3, 1)
            transpose!(f3, real(op.ff3))
        end

    end

end
