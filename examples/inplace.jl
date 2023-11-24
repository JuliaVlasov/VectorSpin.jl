using Plots
using FFTW
using ProgressMeter
using TimerOutputs
using MAT
using JLD2

import VectorSpin: initialfields
import VectorSpin: initialfunction
import VectorSpin: numeint
import VectorSpin: diagnostics
import VectorSpin: H2fh!
import VectorSpin: He!
import VectorSpin: HAA!
import VectorSpin: H3fh!
import VectorSpin: H1f!

const to = TimerOutput()

function inplace()

    T = 5000 # 4000  # final time
    #M = 129 
    M = 259   # partition of x
    N = 259   # partition of v
    #N = 256
    H = 5.0 / 2   # v domain size()
    #H = 8.0 / 2 
    ke = 1.2231333040331807
    L = 4pi / ke  # x domain size()
    h = 0.04 #time step size()
    NUM = floor(Int, T / h + 1.1) # time step number
    a = 0.02 # 0.001; perturbation coefficient
    h_int = 0.00022 # hbar
    #h_int = 0.2
    k0 = 2.0 * ke
    w0 = sqrt(1.0 + k0^2.0)
    ata = 0.5
    #ata = 0.2

    E1, E2, E3, A2, A3 = initialfields(H, L, N, M, a, w0, ke, k0)
    f0, f1, f2, f3 = initialfunction(H, L, N, M, a, ke, ata)

    # test several properties include electric energy; total energy; spectrum etc. save initial data
    Ex_energy = Float64[]
    E_energy = Float64[]
    B_energy = Float64[]
    energy = Float64[]
    Sz = Float64[]
    Tvalue = Vector{Float64}[]
    time = Float64[]

    results = diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, M, N, L, H, h_int)
    push!(Ex_energy, results[1])
    push!(E_energy, results[2])
    push!(B_energy, results[3])
    push!(energy, results[4])
    push!(Sz, results[5])
    push!(Tvalue, results[6])
    push!(time, 0.0)

    @showprogress 1 for i = 1:NUM-1 # Loop over time

        @timeit to "H2fh" H2fh!(f0, f1, f2, f3, E3, A3, h / 2, L, H, h_int)
        @timeit to "He" He!(f0, f1, f2, f3, E1, E2, E3, A2, A3, h / 2, H)
        @timeit to "HAA" HAA!(f0, f1, f2, f3, E2, E3, A2, A3, h / 2, L, H)
        @timeit to "H3fh" H3fh!(f0, f1, f2, f3, E2, A2, h / 2, L, H, h_int)
        @timeit to "H1f" H1f!(f0, f1, f2, f3, E1, h, L, H)
        @timeit to "H3fh" H3fh!(f0, f1, f2, f3, E2, A2, h / 2, L, H, h_int)
        @timeit to "HAA" HAA!(f0, f1, f2, f3, E2, E3, A2, A3, h / 2, L, H)
        @timeit to "He" He!(f0, f1, f2, f3, E1, E2, E3, A2, A3, h / 2, H)
        @timeit to "H2fh" H2fh!(f0, f1, f2, f3, E3, A3, h / 2, L, H, h_int)

        # save properties each time interation
        results = diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, M, N, L, H, h_int)
        push!(Ex_energy, results[1])
        push!(E_energy, results[2])
        push!(B_energy, results[3])
        push!(energy, results[4])
        push!(Sz, results[5])
        push!(Tvalue, results[6])
        push!(time, i * h)
    end

    time, Ex_energy, E_energy, B_energy, energy, Sz, Tvalue

end


time, Ex_energy, E_energy, B_energy, energy, Sz, Tvalue = inplace()
@save "example_M259_N259.jld2" time Ex_energy E_energy B_energy energy Sz Tvalue
#@save "example2594150p022E.jld2" time Ex_energy E_energy B_energy energy Sz Tvalue
#@load "example1.jld2" time Ex_energy E_energy B_energy energy Sz Tvalue
#load_object("example1.jld2") 
show(to)

#vars = matread(joinpath(@__DIR__,"sVMEata0p2.mat"))
#max.(abs.(Ex_energy-vec(vars["Ex_energy"])))
#plot(time[1:600],abs.(Ex_energy-vec(vars["Ex_energy"]))[1:600])
#max.(abs.(E_energy-vars["E_energy"]))
#max.(abs.(B_energy-vars["B_energy"]))
#max.(abs.(energyy-vars["energy"]))
p = plot(layout = (2, 2))
plot!(p[1, 1], time, log.(sqrt.(abs.(Ex_energy))), label = "example2590p032E.jld2")
xlabel!(p[1, 1], "Ex energy - log")
#plot!(p[2,1], time, log.(abs.(E_energy)), label="julia v1")
#xlabel!(p[2,1], "E energy - log")
plot!(p[2, 1], time, Sz, label = "julia v1")
xlabel!(p[2, 1], "Sz")
plot!(p[1, 2], time, B_energy, label = "julia v1")
xlabel!(p[1, 2], "B energy")
#plot!(p[2,2], time, log.(abs.((energy .- energy[begin])./energy[begin])), label="julia v1")
plot!(p[2, 2], time, (abs.((energy .- energy[begin]) ./ energy[begin])), label = "julia v1")
xlabel!(p[2, 2], "energy - log")

#plot!(p[1,1], vec(vars["time"]), log.(vec(vars["Ex_energy"])), label="matlab", legend = :bottomright)
#plot!(p[2,1], vec(vars["time"]), log.(vec(vars["E_energy"])), label="matlab", legend = :bottom)
#plot!(p[1,2], vec(vars["time"]), log.(vec(vars["B_energy"])), label="matlab")
#plot!(p[2,2], vec(vars["time"]), log.(abs.((vec(vars["energy"]).-vec(vars["energy"])[begin])./vec(vars["energy"])[begin])), label="matlab", legend = :bottomleft)
