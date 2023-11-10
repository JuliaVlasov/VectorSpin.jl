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
function bspline(p::Int, j::Int, x::Float64)
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
struct BSplineAdvection <: AbstractAdvection

    mesh::Mesh
    dims::Symbol
    p::Int64
    step::Float64
    modes::Vector{Float64}
    eig_bspl::Vector{Float64}
    eigalpha::Vector{Complex{Float64}}
    fhat::Matrix{ComplexF64}

    function BSplineAdvection(mesh::Mesh; p = 3, dims = :v)

        if dims == :v
            n = mesh.nv
            step = mesh.dv
            fhat = zeros(ComplexF64, mesh.nv, mesh.nx)
        else
            n = mesh.nx
            step = mesh.dx
            fhat = zeros(ComplexF64, mesh.nx, mesh.nv)
        end

        modes = zeros(Float64, n)
        modes .= [2π * i / n for i = 0:n-1]
        eig_bspl = zeros(Float64, n)
        eig_bspl .= bspline(p, -div(p + 1, 2), 0.0)
        for i = 1:div(p + 1, 2)-1
            eig_bspl .+= bspline(p, i - (p + 1) ÷ 2, 0.0) * 2 .* cos.(i * modes)
        end
        eigalpha = zeros(Complex{Float64}, n)
        new(mesh, dims, p, step, modes, eig_bspl, eigalpha, fhat)
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
                exp.((ishift + i) * 1im .* adv.modes)
            )
        end

        # compute interpolating spline using fft and properties of circulant matrices

        adv.fhat[:, j] .*= adv.eigalpha ./ adv.eig_bspl

    end

    ifft!(adv.fhat, 1)
    if adv.dims == :x
        transpose!(f, real(adv.fhat))
    else
        f .= real(adv.fhat)
    end

end
