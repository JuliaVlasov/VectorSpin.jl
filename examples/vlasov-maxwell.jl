using Plots
using FFTW
using MAT
using ProgressMeter
using VectorSpin
using TimerOutputs

const to = TimerOutput()

function vlasov_maxwell()

    T = 50 # 4000  # final time
    nx = 129  # partition of x
    nv = 129   # partition of v
    vmin, vmax = -2.5, 2.5   # v domain size()
    ke = 1.2231333040331807  #ke
    xmin, xmax = 0.0, 4pi / ke  # x domain size()
    h = 0.04 #time step size()
    nsteps = floor(Int, T / h + 1.1) # time step number
    a = 0.02 # 0.001; perturbation coefficient
    h_int = 0.2 # hbar
    k0 = 2.0 * ke
    ww = sqrt(1.0 + k0^2.0) # w0
    ata = 0.2
    vth = 0.17

    mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)

    E1, E2, E3, A2, A3 = initialfields(mesh, a, ww, ke, k0)
    f0, f1, f2, f3 = initialfunction(mesh, a, ke, vth, ata)

    Ex_energy = Float64[]
    E_energy = Float64[]
    B_energy = Float64[]
    energy = Float64[]
    Sz = Float64[]
    Tvalue = Vector{Float64}[]
    time = Float64[]

    data = Diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, mesh, h_int)

    H2 = H2Subsystem(mesh)
    He = HeSubsystem(mesh)
    HA = HASubsystem(mesh)
    H3 = H3Subsystem(mesh)
    Hp = HpSubsystem(mesh)

    @showprogress 1 for i = 1:nsteps # Loop over time

        @timeit to "H2" step!(H2, f0, f1, f2, f3, E3, A3, h / 2, h_int)
        @timeit to "He" step!(He, f0, f1, f2, f3, E1, E2, E3, A2, A3, h / 2)
        @timeit to "HA" step!(HA, f0, f1, f2, f3, E2, E3, A2, A3, h / 2)
        @timeit to "H3" step!(H3, f0, f1, f2, f3, E2, A2, h / 2, h_int)
        @timeit to "Hp" step!(Hp, f0, f1, f2, f3, E1, h)
        @timeit to "H3" step!(H3, f0, f1, f2, f3, E2, A2, h / 2, h_int)
        @timeit to "HA" step!(HA, f0, f1, f2, f3, E2, E3, A2, A3, h / 2)
        @timeit to "He" step!(He, f0, f1, f2, f3, E1, E2, E3, A2, A3, h / 2)
        @timeit to "H2" step!(H2, f0, f1, f2, f3, E3, A3, h / 2, h_int)
        save!(data, i * h, f0, f2, f3, E1, E2, E3, A2, A3)

    end

    data

end

results = vlasov_maxwell()

vars = matread(joinpath(@__DIR__, "sVMEata0p2.mat"))

show(to)

p = plot(layout = (3, 2))
plot!(p[1, 1], results.time, log.(results.Ex_energy), label = "julia")
xlabel!(p[1, 1], "Ex energy - log")
plot!(p[2, 1], results.time, log.(results.E_energy), label = "julia")
xlabel!(p[2, 1], "E energy - log")
plot!(p[1, 2], results.time, log.(results.B_energy), label = "julia")
xlabel!(p[1, 2], "B energy - log")
plot!(p[2, 2], results.time, log.(results.energy), label = "julia")
xlabel!(p[2, 2], "energy - log")
plot!(p[3, 1], results.time, results.Sz, label = "julia")
xlabel!(p[3, 1], "Sz")

plot!(
    p[1, 1],
    vec(vars["time"]),
    log.(vec(vars["Ex_energy"])),
    label = "matlab",
    legend = :bottomright,
)
plot!(
    p[2, 1],
    vec(vars["time"]),
    log.(vec(vars["E_energy"])),
    label = "matlab",
    legend = :bottom,
)
plot!(p[1, 2], vec(vars["time"]), log.(vec(vars["B_energy"])), label = "matlab")
plot!(
    p[2, 2],
    vec(vars["time"]),
    log.(vec(vars["energy"])),
    label = "matlab",
    legend = :bottomleft,
)
plot!(p[3, 1], vec(vars["time"]), log.(vec(vars["Sz"])), label = "matlab")
