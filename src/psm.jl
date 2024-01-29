import LinearAlgebra: SymTridiagonal

export PSMAdvection

struct PSMAdvection{T} <: AbstractAdvection

    mesh::Mesh{T}
    a::Vector{T}
    b::Vector{T}
    c::Vector{T}
    d::Vector{T}
    diag::Vector{T}
    dsup::Vector{T}
    A::SymTridiagonal{T,Vector{T}}

    function PSMAdvection(mesh::Mesh{T}) where {T}

        a = zeros(T, mesh.nv)
        b = zeros(T, mesh.nv)
        c = zeros(T, mesh.nv + 1)
        d = zeros(T, mesh.nv)

        diag = 4 / 3 .* ones(T, mesh.nv + 1)
        diag[begin] = 2 / 3
        diag[end] = 2 / 3
        dsup = 1 / 3 .* ones(T, mesh.nv)
        A = SymTridiagonal{T,Vector{T}}(diag, dsup)

        new{T}(mesh, a, b, c, d, diag, dsup, A)

    end


end

"""
$(SIGNATURES)

interpolate df(x + delta) with Parabolic Spline Method (PSM) 

We consider a
linear advection problem in ``p`` direction
```math
\\frac{\\partial f}{\\partial t} + a \\frac{\\partial f}{\\partial x} =0.
```
From the conservation of the volume, we have the following identity

```math
f_{j,\\ell}(t)=\\frac{1}{\\Delta p} \\int_{p_{\\ell-1/2}} ^{p_{\\ell+1/2}} f(x_j,p,t)\\mathrm{d}{p} =\\frac{1}{\\Delta p} \\int_{p_{\\ell-1/2}-at} ^{p_{\\ell+1/2}-at} f(x_j,p,0)\\mathrm{d}{p}.
```

For simplicity, denote by ``q\\in [1, M]`` the index such that
``p_{\\ell+1/2}-at \\in [p_{q-1/2},p_{q+1/2}]`` i.e.
``p_{\\ell+1/2}-at \\in C_q``, then we have

```math
f_{j,\\ell}(t) =\\frac{1}{\\Delta p} \\int_{p_{q-1/2}-at} ^{p_{q-1/2}} f(x_j,p,0)\\mathrm{d}{p}+f_{j,q}(0)-\\frac{1}{\\Delta p} \\int_{p_{q+1/2}} ^{p_{q+1/2}-at} f(x_j,p,0)\\mathrm{d}{p}.
```

Here we need to reconstruct a polynomial function ``f(x_j,p,0)`` using the
averages ``f_{j,l}(0)`` using the PSM approach. 

"""
function advection!(df, adv::PSMAdvection{T}, v, dt) where {T}

    nx = adv.mesh.nx
    nv = adv.mesh.nv
    dv = adv.mesh.dv

    @inbounds for j in eachindex(v)

        adv.c[begin] = df[1, j]
        for i = 2:nv
            adv.c[i] = df[i-1, j] + df[i, j]
        end
        adv.c[end] = df[nv, j]
        adv.c .= adv.A \ adv.c

        for i = 2:nv
            adv.d[i] = (-1)^i * (adv.c[i] - adv.c[i-1])
        end
        for i = 2:nv
            adv.b[i] = (-1)^i * 2 * sum(view(adv.d, 2:i))
        end
        adv.b[begin] = 0
        for i = 1:nv-1
            adv.a[i] = 0.5 * (adv.b[i+1] - adv.b[i])
        end
        adv.a[end] = -0.5 * adv.b[end]

        alpha = v[j] * dt / dv

        for i = 1:nv

            beta = i - alpha
            newbeta = beta - nv * floor(Int, beta / nv)

            if newbeta >= 1.0
                l = floor(Int, newbeta)
                k = T(1) - (newbeta - l)
            else
                l = nv
                k = T(1) - newbeta
            end

            l1 = mod1(l + 1, nv)
            k1 = T(1) - k
            k2 = k1 * k1
            k3 = k2 * k1

            val = adv.a[l] / 3 + 0.5 * adv.b[l] + adv.c[l]
            val += -adv.a[l] / 3 * k3 - 0.5 * adv.b[l] * k2
            val += -adv.c[l] * k1
            val += adv.a[l1] / 3 * k3 + 0.5 * adv.b[l1] * k2
            val += adv.c[l1] * k1

            df[i, j] = val

        end

    end

end
