
export BSplineAdvection

"""
$(SIGNATURES)

Return the value at x in [0,1] of the B-spline with integer nodes of degree
p with support starting at j.  Implemented recursively using the 
[De Boor's Algorithm](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm)
```math
B_{i,0}(x) := \\left\\{
\\begin{matrix}
1 & \\mathrm{if}  \\quad t_i ≤ x < t_{i+1} \\\\
0 & \\mathrm{otherwise} 
\\end{matrix}
\\right.
```
```math
B_{i,p}(x) := \\frac{x - t_i}{t_{i+p} - t_i} B_{i,p-1}(x) 
+ \\frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} B_{i+1,p-1}(x).
```
"""
function bspline(p, j, x)
    if p == 0
        j == 0 ? (return 1.0) : (return 0.0)
    else
        w = (x - j) / p
        w1 = (x - j - 1) / p
    end
    return w * bspline(p - 1, j, x) + (1 - w1) * bspline(p - 1, j + 1, x)
end

""" 
$(TYPEDEF)

Advection to be computed on each row

$(TYPEDFIELDS)

"""
struct BSplineAdvection{T} <: AbstractAdvection

    mesh::Mesh{T}
    dims::Symbol
    p::Int64
    step::T
    modes::Vector{T}
    eig_bspl::Vector{T}
    eigalpha::Vector{Complex{T}}
    fhat::Matrix{Complex{T}}

    function BSplineAdvection(mesh::Mesh{T}; p = 3, dims = :v) where {T}

        if dims == :v
            n = mesh.nv
            step = mesh.dv
            fhat = zeros(Complex{T}, mesh.nv, mesh.nx)
        else
            n = mesh.nx
            step = mesh.dx
            fhat = zeros(Complex{T}, mesh.nx, mesh.nv)
        end

        modes = zeros(T, n)
        modes .= [2π * i / n for i = 0:n-1]
        eig_bspl = zeros(T, n)
        eig_bspl .= bspline(p, -div(p + 1, 2), 0.0)
        for i = 1:div(p + 1, 2)-1
            eig_bspl .+= bspline(p, i - (p + 1) ÷ 2, 0.0) * 2 .* cos.(i * modes)
        end
        eigalpha = zeros(Complex{T}, n)
        new{T}(mesh, dims, p, step, modes, eig_bspl, eigalpha, fhat)
    end

end

export advection!

function advection!(f, adv::BSplineAdvection, v, dt)

    if adv.dims == :x
        transpose!(adv.fhat, f)
    else
        adv.fhat .= f
    end

    fft!(adv.fhat, 1)

    @inbounds for j in eachindex(v)
        alpha = dt * v[j] / adv.step
        # compute eigenvalues of cubic splines evaluated at displaced points
        ishift = floor(-alpha)
        beta = -ishift - alpha
        fill!(adv.eigalpha, 0.0im)
        for i = -div(adv.p - 1, 2):div(adv.p + 1, 2)
            adv.eigalpha .+= (
                bspline(adv.p, i - div(adv.p + 1, 2), beta) .*
                cis.((ishift + i) .* adv.modes)
            )
        end

        # compute interpolating spline using fft and properties of circulant matrices

        for i in axes(adv.fhat, 1)
            adv.fhat[i, j] *= adv.eigalpha[i] ./ adv.eig_bspl[i]
        end

    end

    ifft!(adv.fhat, 1)
    if adv.dims == :x
        transpose!(f, real(adv.fhat))
    else
        f .= real(adv.fhat)
    end

end
