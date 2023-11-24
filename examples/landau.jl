import VectorSpin: bspline

using FFTW
using LinearAlgebra
using Plots
using ProgressMeter
using Statistics
using VectorSpin
using SemiLagrangian

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

    # Set grid
    p = 3
    nx, nv = 128, 256
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

        push!(ℰ, 0.5 * log(sum(e .* e) * dx))

        VectorSpin.advection!(f, adv_v, e, dt)

        VectorSpin.advection!(f, adv_x, v, 0.5dt)

    end

    ℰ

end


function run_simulation(nbdt, sz, dt)

    epsilon = 0.001

    xmin, xmax, nx = 0.0, 4π, sz[1]
    vmin, vmax, nv = -6.0, 6.0, sz[2]

    mesh_x = UniformMesh(xmin, xmax, nx)
    mesh_v = UniformMesh(vmin, vmax, nv)

    states = [([1, 2], 1, 1, true), ([2, 1], 1, 2, true)]

    interp = Lagrange(9, Float64)
    tab_coef = strangsplit(dt)

    adv = Advection(
        (mesh_x, mesh_v),
        [interp, interp],
        dt,
        states;
        tab_coef,
        timeopt = NoTimeOpt,
    )

    kx = 0.5
    x = mesh_x.points
    v = mesh_v.points
    fx = epsilon * cos.(kx * x) .+ 1
    fv = exp.(-v .^ 2 / 2) ./ sqrt(2π)
    f = fx .* fv'

    pvar = getpoissonvar(adv)

    advd = AdvectionData(adv, f, pvar)

    time = Float64[]
    el = Float64[]
    @showprogress 1 for i = 1:nbdt
        while SemiLagrangian.advection!(advd)
        end
        push!(time, advd.time_cur)
        ee = compute_ee(advd)
        push!(el, 0.5 * log(ee))
    end
    return time, el
end

nt = 1000
tf = 100.0
t = range(0.0, stop = tf, length = nt)


sz = (128, 256)
dt = tf / nt
@time time, el = run_simulation(nt, sz, dt)
@time nrj = landau_damping(tf, nt);
plot(t, nrj; label = "LocalImplementation")
plot!(t, -0.1533 * t .- 5.50; label = "-0.1533t.-5.5")
plot!(time, el, label = "SemiLagrangian")
