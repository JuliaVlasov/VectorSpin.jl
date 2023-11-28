export Hv!

export HvOperator

struct HvOperator

    mesh::Mesh
    ff0::Matrix{ComplexF64}
    ff1::Matrix{ComplexF64}
    ff2::Matrix{ComplexF64}
    ff3::Matrix{ComplexF64}

    function HvOperator(mesh)

        ff0 = zeros(ComplexF64, mesh.nx, mesh.nv)
        ff1 = zeros(ComplexF64, mesh.nx, mesh.nv)
        ff2 = zeros(ComplexF64, mesh.nx, mesh.nv)
        ff3 = zeros(ComplexF64, mesh.nx, mesh.nv)

        new(mesh, ff0, ff1, ff2, ff3)

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
function step!(op::HvOperator, f0, f1, f2, f3, E1, dt)

    v = op.mesh.vnode
    k_fre = op.mesh.kx

    transpose!(op.ff0, f0)
    transpose!(op.ff1, f1)
    transpose!(op.ff2, f2)
    transpose!(op.ff3, f3)

    fft!(op.ff0, 1)
    fft!(op.ff1, 1)
    fft!(op.ff2, 1)
    fft!(op.ff3, 1)

    for i in eachindex(v), j in eachindex(k_fre)

        expv = exp(-1im * k_fre[j] * v[i] * dt)

        op.ff0[j, i] *= expv
        op.ff1[j, i] *= expv
        op.ff2[j, i] *= expv
        op.ff3[j, i] *= expv

    end

    E1t = vec(sum(op.ff0, dims = 2))
    E1t[1] = 0.0

    for i = 2:op.mesh.nx
        E1t[i] *= op.mesh.dv / (-1im * k_fre[i])
    end

    ifft!(op.ff0, 1)
    ifft!(op.ff1, 1)
    ifft!(op.ff2, 1)
    ifft!(op.ff3, 1)

    transpose!(f0, real(op.ff0))
    transpose!(f1, real(op.ff1))
    transpose!(f2, real(op.ff2))
    transpose!(f3, real(op.ff3))

    E1 .= E1t

end
