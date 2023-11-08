# -*- coding: utf-8 -*-
using CSV
using DataFrames
using Plots
using FFTW
using MAT
using ProgressMeter
using Random
using SpinGEMPIC
using Statistics
import GEMPIC: OneDGrid, Maxwell1DFEM
import GEMPIC: eval_uniform_periodic_spline_curve
using VectorSpin
import GEMPIC: l2projection!

import VectorSpin: initialfunction
import VectorSpin: initialfields
import VectorSpin: diagnostics
import VectorSpin: H2fh!
import VectorSpin: He!
import VectorSpin: HAA!
import VectorSpin: H3fh!
import VectorSpin: H1f!

function compute_rho(mesh::Mesh, f::Array{Float64,2})

   dv = mesh.dv
   ρ = dv * sum(real(f), dims=1)
   vec(ρ .- mean(ρ)) # vec squeezes the 2d array returned by sum function
end

function compute_e(mesh::Mesh, ρ::Vector{Float64})
   nx = mesh.nx
   modes = mesh.kx
   modes[1] = 1.0
   ρ = fft(ρ)./modes
   ρk .*= -1im
   ifft!(ρk)
   real(ρk)
end


T = 1000 # 4000  # final time
nx = 128  # partition of x
nv = 128   # partition of v
vmin, vmax = -2.5,2.5   # v domain size()
ke = 1.2231333040331807
xmin, xmax = 0, 4pi / ke  # x domain size()
h = 0.05 #time step size()
nsteps = floor(Int, T / h + 1.1) # time step number
a = 0.02 # 0.001; perturbation coefficient
h_int = 0.2 # hbar
k0 = 2.0 * ke
ww = sqrt(1.0 + k0^2.0) # w0
ata = 0.2
σ = 0.17

mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)
#adv = BSplineAdvection(mesh)
adv = PSMAdvection(mesh);

E1, E2, E3, A2, A3 = initialfields( mesh, a, ww, ke, k0)
f0, f1, f2, f3 = initialfunction(mesh, a, ke, σ, ata);

x = LinRange(xmin ,xmax, nx+1)[1:end-1]
v = LinRange(vmin, vmax, nv+1)[1:end-1]
surface( x, v, f0)


domain = [xmin, xmax, xmax - xmin]
n_particles = 30000
mesh_1d = OneDGrid( xmin, xmax, nx)
spline_degree = 3
σ, μ = 0.17, 0.0
kx, α = ke, a

df = CosGaussian(kx, a, σ, μ)

rng = MersenneTwister(1234)
mass, charge = 1.0, 1.0

# +
n_particles = 20000
particle_group = ParticleGroup( n_particles, mass, charge, 1)   
sample!(rng, particle_group, df, mesh_1d, method=:weighted)

sum(particle_group.array[5,:] .* particle_group.array[6,:] .* particle_group.common_weight)

# +
n_particles = 100000
particle_group = ParticleGroup( n_particles, mass, charge, 1)   
sample!(rng, particle_group, df, mesh_1d, method=:weighted)

sum(particle_group.array[5,:] .* particle_group.array[6,:] .* particle_group.common_weight)
# -

kernel_smoother2 = ParticleMeshCoupling( mesh_1d, n_particles, spline_degree-2) 
kernel_smoother1 = ParticleMeshCoupling( mesh_1d, n_particles, spline_degree-1)    
kernel_smoother0 = ParticleMeshCoupling( mesh_1d, n_particles, spline_degree);

maxwell_solver = Maxwell1DFEM(mesh_1d, spline_degree);

ρ_scalar = zeros(nx)
efield_poisson = zeros(nx)

solve_poisson!( efield_poisson, particle_group, kernel_smoother0, maxwell_solver, ρ_scalar )

ρ_vector = compute_rho(mesh, f0)

p = plot(layout=(1,2))
xg = LinRange(xmin, xmax, nx)
sval = eval_uniform_periodic_spline_curve(spline_degree-1, ρ_scalar .- mean(ρ_scalar))
plot!(p[1,1], xg, sval, title=:ρ, label="scalar")
plot!(p[1,1], xg, ρ_vector .* ke ./ 4pi , title=:ρ, label="vector")
sval = eval_uniform_periodic_spline_curve(spline_degree-1, efield_poisson)
plot!(p[1,2], xg, sval, title=:ex, label="scalar" )
plot!(p[1,2], xg, real(ifft(E1)), title=:ex, label="vector" )
plot!(p[1,1], x -> a  * cos(ke *x) * ke / 4pi, xmin, xmax)
plot!(p[1,2], x -> a / ke * sin(ke *x), xmin, xmax)

xp = view(particle_group.array, 1, :)
vp = view(particle_group.array, 2, :)
s1 = view(particle_group.array, 3, :)
s2 = view(particle_group.array, 4, :)
s3 = view(particle_group.array, 5, :)
wp = view(particle_group.array, 6, :);

p = plot(layout=(2,1))
histogram!(p[1,1], xp, weights=wp, normalize=true, bins = 100, lab = "")
#plot!(p[1,1], x-> (1+α*cos(kx*x))/(sqrt(4π)/kx), 0., 4π/kx, lab="")
histogram!(p[2,1], vp, weights=wp, normalize=true, bins = 100, lab = "")
#plot!(p[2,1], v-> exp( - v^2 / 2) * 4 / π^2 , -6, 6, lab="")


E0 = 0.123 * ww
Ey(x) = E0*cos(k0*x)
Ez(x) = E0*sin(k0*x)
Ay(x) = -E0/ww*sin(k0*x)
Az(x) = E0/ww*cos(k0*x)

plot(mesh.x, real(ifft(A2)), markershape=:circle)
plot!(mesh.x, Ay.(mesh.x))

plot(mesh.x, real(ifft(A3)), markershape=:circle)
plot!(mesh.x, Az.(mesh.x))

plot(mesh.x, real(ifft(E2)), markershape=:circle)
plot!(mesh.x, Ey.(mesh.x))

plot(mesh.x, real(ifft(E3)), markershape=:circle)
plot!(mesh.x, Ez.(mesh.x))

# +
efield_dofs = [ zeros(nx), zeros(nx), zeros(nx)]
efield_dofs[1] .= efield_poisson
afield_dofs = [zeros(nx), zeros(nx)]

l2projection!( efield_dofs[2], maxwell_solver, Ey, spline_degree)
l2projection!( efield_dofs[3], maxwell_solver, Ez, spline_degree)
l2projection!( afield_dofs[1], maxwell_solver, Ay, spline_degree)
l2projection!( afield_dofs[2], maxwell_solver, Az, spline_degree)

propagator = HamiltonianSplitting( maxwell_solver,
                                   kernel_smoother0,
                                   kernel_smoother1,
                                   kernel_smoother2,
                                   efield_dofs,
                                   afield_dofs,
                                   domain);
thdiag = TimeHistoryDiagnostics( maxwell_solver,
                            kernel_smoother0, kernel_smoother1 )

write_step!(thdiag, 0.0, spline_degree,
                        efield_dofs,  afield_dofs,
                        efield_poisson,
                        propagator, particle_group)
# -

steps, Δt = 100, 0.04

# +
@showprogress 1 for j = 1:steps # loop over time

    strang_splitting!(propagator, particle_group, Δt, 1)

    write_step!(thdiag, j * Δt, spline_degree,
                        efield_dofs,  afield_dofs,
                        efield_poisson,
                        propagator, particle_group)

end

# -

pic = thdiag.data;

# +
h_int = 0.00022980575
results = Diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, mesh, h_int)

H2fh = H2fhOperator(adv)
He = HeOperator(adv)
HAA = HAAOperator(adv)
H3fh = H3fhOperator(adv)
H1f = H1fOperator(adv)

nsteps, dt = 100, 0.04

@showprogress 1 for i = 1:nsteps # Loop over time

    step!(f0, f1, f2, f3, E3, A3, H2fh, 0.5dt, h_int)
    step!(f0, f1, f2, f3, E1, E2, E3, A2, A3, He, 0.5dt)
    step!(f0, f1, f2, f3, E2, E3, A2, A3, HAA, 0.5dt)
    step!(f0, f1, f2, f3, E2, A2, H3fh, 0.5dt, h_int)
    step!(f0, f1, f2, f3, E1, H1f, dt)
    step!(f0, f1, f2, f3, E2, A2, H3fh, 0.5dt, h_int)
    step!(f0, f1, f2, f3, E2, E3, A2, A3, HAA, 0.5dt)
    step!(f0, f1, f2, f3, E1, E2, E3, A2, A3, He, 0.5dt)
    step!(f0, f1, f2, f3, E3, A3, H2fh, 0.5dt, h_int)
    
    save!(results, i*dt, f0, f2, f3, E1, E2, E3, A2, A3)

end

# -

plot(pic.Time, pic.PotentialEnergyE1)
plot!(results.time, results.Ex_energy)

e_energy = pic.PotentialEnergyE1 .+ pic.PotentialEnergyE2 .+ pic.PotentialEnergyE3
plot(pic.Time, e_energy .- mean(e_energy))
plot!(results.time, results.E_energy .- mean(results.E_energy))

b_energy = pic.PotentialEnergyB2 .+ pic.PotentialEnergyB3
plot(pic.Time, b_energy .- mean(b_energy))
plot!(results.time, results.B_energy .- mean(results.B_energy))

plot(pic.Time, pic.Momentum8)

energy = (pic.Momentum7 .- mean(pic.Momentum7)) ./ maximum(abs.(pic.Momentum7))
plot(pic.Time, energy, label="pic")
plot!(results.time, (results.Sz .- mean(results.Sz)) ./ maximum(abs.(results.Sz)), label="xue")

plot(pic.Time, pic.Momentum8)


