import VectorSpin: bspline

using FFTW
using LinearAlgebra
using Plots
using ProgressMeter
using Statistics
using VectorSpin

try
    import SemiLagrangian
    using DoubleFloats
catch
    import Pkg
    Pkg.add("SemiLagrangian")
    import Pkg
    Pkg.add("DoubleFloats")
    using SemiLagrangian
    using DoubleFloats
end

"""
    compute_rho(meshv, f)

Compute charge density
ρ(x,t) = ∫ f(x,v,t) dv
"""
function compute_rho(mesh::Mesh, f::Array{Float64,2})

    dv = mesh.dv
    rho = dv * sum(real(f), dims = 1)
    vec(rho .- mean(rho)) # vec squeezes the 2d array returned by sum function

end

"""
    compute_e(meshx, rho)
compute Ex using that -ik*Ex = rho
"""
function compute_e(mesh::Mesh, rho::Vector{Float64})
    nx = mesh.nx
    modes = mesh.kx
    modes[1] = 1.0
    rhok = fft(rho) ./ modes
    rhok .*= -1im
    ifft!(rhok)
    real(rhok)
end

function landau(ϵ, kx, mesh)
    nx = mesh.nx
    nv = mesh.nv
    x = mesh.x
    v = mesh.v
    f = zeros(nv, nx)
    f .= transpose(1.0 .+ ϵ * cos.(kx * x)) / sqrt(2π) .* exp.(-0.5 * v .^ 2)
    f
end

function landau_damping(tf::Float64, nt::Int64)

    nx, nv = 127, 255
    xmin, xmax = 0.0, 4π
    vmin, vmax = -6.0, 6.0
    mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)
    x, v = mesh.x, mesh.v
    dx = mesh.dx

    # Set distribution function for Landau damping
    ϵ, kx = 0.001, 0.5
    f = landau(ϵ, kx, mesh)

    # Instantiate advection functions
    adv_x = BSplineAdvection(mesh, p = 5, dims = :x)
    adv_v = PSMAdvection(mesh)

    # Set time step
    dt = tf / nt

    # Run simulation
    ℰ = Float64[]

    @showprogress 1 for it = 1:nt

        VectorSpin.advection!(f, adv_x, v, 0.5dt)

        ρ = compute_rho(mesh, f)
        e = compute_e(mesh, ρ)

        push!(ℰ, sum(e .* e) * dx)

        VectorSpin.advection!(f, adv_v, e, dt)

        VectorSpin.advection!(f, adv_x, v, 0.5dt)

    end

    ℰ

end


function run_simulation(T::DataType, nbdt, sz, dt)

    epsilon = 0.001

    xmin, xmax, nx = T(0.0), T(4π), sz[1]
    vmin, vmax, nv = -T(6.0), T(6.0), sz[2]

    mesh_x = UniformMesh(xmin, xmax, nx)
    mesh_v = UniformMesh(vmin, vmax, nv)

    states = [([1, 2], 1, 1, true), ([2, 1], 1, 2, true)]

    interp = Lagrange(9, T)
    tab_coef = strangsplit(dt)

    adv = Advection(
        (mesh_x, mesh_v),
        [interp, interp],
        dt,
        states;
        tab_coef,
        timeopt = NoTimeOpt,
    )

    kx = T(0.5)
    x = mesh_x.points
    v = mesh_v.points
    fx = epsilon * cos.(kx * x) .+ 1
    fv = exp.(-v .^ 2 / 2) ./ sqrt(2π)
    f = T.(fx .* fv')

    pvar = getpoissonvar(adv)

    advd = AdvectionData(adv, f, pvar)

    t = T[]
    el = T[]
    @showprogress 1 for i = 1:nbdt
        while SemiLagrangian.advection!(advd)
        end
        push!(t, advd.time_cur)
        ee = compute_ee(advd)
        push!(el, ee)
    end
    return t, el
end

nt = 1000
tf = 200.0
t = LinRange(0.0, tf, nt)

sz = (128, 256)
dt = tf / nt
nrj = landau_damping(tf, nt);
plot(t, nrj; label = "PSM+BSpline")
line, g = fit_complex_frequency(t, nrj)
plot!(t, line, label = "growth = $(imag(g))")
t, el = run_simulation(Double64, nt, sz, Double64(dt))
plot!(t, el, label = "SemiLagrangian", yscale = :log10)
