using .Threads
import LinearAlgebra: transpose!

export HvSubsystem

struct HvSubsystem{T}

    mesh::Mesh{T}
    ff0::Matrix{Complex{T}}
    ff1::Matrix{Complex{T}}
    ff2::Matrix{Complex{T}}
    ff3::Matrix{Complex{T}}
    expv::Matrix{Complex{T}}

    function HvSubsystem(mesh::Mesh{T}) where {T}

        ff0 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff1 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff2 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff3 = zeros(Complex{T}, mesh.nx, mesh.nv)
        expv = exp.(-1im .* mesh.kx .* mesh.vnode')

        new{T}(mesh, ff0, ff1, ff2, ff3, expv)

    end

end

"""
$(SIGNATURES)

subsystem for Hv:

```math
f_t + vf_x = 0
```

```math
-E_x = \\rho - 1
```

"""
function step!(
    op::HvSubsystem{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    E1::Vector{Complex{T}},
    dt::T,
) where {T}

    v = op.mesh.vnode
    k_fre = op.mesh.kx

    @sync begin

        @spawn begin
            transpose!(op.ff0, f0)
            fft!(op.ff0, 1)
            op.ff0 .*= op.expv .^ dt
            E1 .= vec(sum(op.ff0, dims = 2))
            E1[1] = 0.0

            for i = 2:op.mesh.nx
                E1[i] *= op.mesh.dv / (-1im * k_fre[i])
            end

            ifft!(op.ff0, 1)
            transpose!(f0, real(op.ff0))
        end

        @spawn begin
            transpose!(op.ff1, f1)
            fft!(op.ff1, 1)
            op.ff1 .*= op.expv .^ dt
            ifft!(op.ff1, 1)
            transpose!(f1, real(op.ff1))
        end

        @spawn begin
            transpose!(op.ff2, f2)
            fft!(op.ff2, 1)
            op.ff2 .*= op.expv .^ dt
            ifft!(op.ff2, 1)
            transpose!(f2, real(op.ff2))
        end

        @spawn begin
            transpose!(op.ff3, f3)
            fft!(op.ff3, 1)
            op.ff3 .*= op.expv .^ dt
            ifft!(op.ff3, 1)
            transpose!(f3, real(op.ff3))
        end
    end

end
