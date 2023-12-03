export H1fOperator

struct H1fOperator{T}

    mesh::Mesh{T}
    ff0::Matrix{Complex{T}}
    ff1::Matrix{Complex{T}}
    ff2::Matrix{Complex{T}}
    ff3::Matrix{Complex{T}}
    expv::Matrix{Complex{T}}

    function H1fOperator(mesh::Mesh{T}) where {T}

        ff0 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff1 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff2 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff3 = zeros(Complex{T}, mesh.nx, mesh.nv)
        expv = zeros(Complex{T}, mesh.nx, mesh.nv)

        new{T}(mesh, ff0, ff1, ff2, ff3, expv)

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
function step!(
    op::H1fOperator{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    E1::Vector{Complex{T}},
    dt::T,
) where {T}

    dv = op.mesh.dv
    nx = op.mesh.nx
    nv = op.mesh.nv
    kx = op.mesh.kx
    v = op.mesh.vnode
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
