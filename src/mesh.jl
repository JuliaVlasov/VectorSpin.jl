export Mesh

""" 
$(TYPEDEF)
Mesh type to store domain parameters
$(TYPEDFIELDS)
"""
struct Mesh

    "Number of points in v"
    nv::Int64
    "Number of points in x"
    nx::Int64
    "Domain size v ∈ ]vmin,vmax["
    vmin::Float64
    "Domain size v ∈ ]vmin,vmax["
    vmax::Float64
    "Domain size x ∈ [xmin,xmax]"
    xmin::Float64
    "Domain size x ∈ [xmin,xmax]"
    xmax::Float64
    "Wave number vector to compute derivative with FFTs"
    kx::Vector{Float64}
    "Size step along x"
    dx::Float64
    "Size step along v"
    dv::Float64
    "points along x direction"
    x::Vector{Float64}
    "points along v direction"
    v::Vector{Float64}
    "center points along v direction"
    vnode::Vector{Float64}

    function Mesh(xmin, xmax, nx, vmin, vmax, nv)

        dx = (xmax - xmin) / nx
        dv = (vmax - vmin) / nv
        kx = collect(2π ./ (xmax - xmin) .* fftfreq(nx, nx))
        x = LinRange(xmin, xmax, nx + 1)[1:end-1] # remove last point
        v = LinRange(vmin, vmax, nv + 1)[2:end]   # remove first point
        vnode = v .- 0.5dv

        new(nv, nx, vmin, vmax, xmin, xmax, kx, dx, dv, x, v, vnode)

    end
end
