export Mesh

""" 
$(TYPEDEF)
Mesh type to store domain parameters
$(TYPEDFIELDS)
"""
struct Mesh{T}

    "Number of points in v"
    nv::Int64
    "Number of points in x"
    nx::Int64
    "Domain size v ∈ ]vmin,vmax["
    vmin::T
    "Domain size v ∈ ]vmin,vmax["
    vmax::T
    "Domain size x ∈ [xmin,xmax]"
    xmin::T
    "Domain size x ∈ [xmin,xmax]"
    xmax::T
    "Wave number vector to compute derivative with FFTs"
    kx::Vector{T}
    "Size step along x"
    dx::T
    "Size step along v"
    dv::T
    "points along x direction"
    x::Vector{T}
    "points along v direction"
    v::Vector{T}
    "center points along v direction"
    vnode::Vector{T}

    function Mesh(xmin::T, xmax::T, nx, vmin::T, vmax::T, nv) where {T}

        dx = (xmax - xmin) / nx
        dv = (vmax - vmin) / nv
        kx = collect(2π ./ (xmax - xmin) .* fftfreq(nx, nx))
        x = collect(LinRange(xmin, xmax, nx + 1)[1:end-1]) # remove last point
        v = collect(LinRange(vmin, vmax, nv + 1)[2:end])   # remove first point
        vnode = v .- 0.5dv

        new{T}(nv, nx, vmin, vmax, xmin, xmax, kx, dx, dv, x, v, vnode)

    end
end
