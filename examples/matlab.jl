

using MAT
using Plots

vars = matread("sVMEata0p2.mat")

plot(vec(vars["time"]), vec(vars["Ex_energy"]))

plot(vec(vars["time"]), vec(vars["E_energy"]))

plot(vec(vars["time"]), vec(vars["B_energy"]))

plot(vec(vars["time"]), vec(vars["Sz"]))

plot(vec(vars["time"]), vec(vars["energy"]))

@gif for Tvalue in eachcol(vars["Tvalue"])
    plot(Tvalue)
    ylims!(0, 0.1)
end
