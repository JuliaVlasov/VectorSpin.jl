export diagnostics, Diagnostics, save!

struct Diagnostics

    mesh::Mesh
    h_int::Float64
    Ex_energy::Vector{Float64}
    E_energy::Vector{Float64}
    B_energy::Vector{Float64}
    energy::Vector{Float64}
    Sz::Vector{Float64}
    T::Vector{Vector{Float64}}
    time::Vector{Float64}

    function Diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, mesh::Mesh, h_int)

        results = diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, mesh, h_int)
        Ex_energy = [results[1]]
        E_energy = [results[2]]
        B_energy = [results[3]]
        energy = [results[4]]
        Sz = [results[5]]
        T = [results[6]]
        time = [0.0]

        new(mesh, h_int, Ex_energy, E_energy, B_energy, energy, Sz, T, time)

    end

end

function save!(results, time, f0, f2, f3, E1, E2, E3, A2, A3)

    diags = diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, results.mesh, results.h_int)

    push!(results.Ex_energy, diags[1])
    push!(results.E_energy, diags[2])
    push!(results.B_energy, diags[3])
    push!(results.energy, diags[4])
    push!(results.Sz, diags[5])
    push!(results.T, diags[6])
    push!(results.time, time)

end


function diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, mesh::Mesh, h_int)

    nv, nx = mesh.nv, mesh.nx
    kx = mesh.kx
    dx = mesh.dx
    dv = mesh.dv
    v = mesh.v

    B2 = -1im .* kx .* A3
    B3 = 1im .* kx .* A2

    EE1 = real(ifft(E1))
    EE2 = real(ifft(E2))
    EE3 = real(ifft(E3))
    BB2 = real(ifft(B2))
    BB3 = real(ifft(B3))
    AA2 = real(ifft(A2))
    AA3 = real(ifft(A3))

    # electric energy related to E1
    Ex_energy = 1 / 2 * sum(EE1 .^ 2) * dx
    # electric energy
    E_energy =
        1 / 2 * sum(EE1 .^ 2) * dx + 1 / 2 * sum(EE3 .^ 2) * dx + 1 / 2 * sum(EE2 .^ 2) * dx
    # magnetic energy
    B_energy = 1 / 2 * sum(BB2 .^ 2) * dx + 1 / 2 * sum(BB3 .^ 2) * dx
    energy2 = E_energy + B_energy

    ff0 = 1 / 2 * (f0 .* (mesh.vnode .^ 2 .+ AA2' .^ 2 .+ AA3' .^ 2)) * dx * dv
    ff2 = -h_int * f2 .* BB2' * dx * dv
    ff3 = -h_int * f3 .* BB3' * dx * dv

    Jbar = vec(sum(f0 .* mesh.vnode, dims = 1))
    ubar = Jbar ./ sum(f0)

    # temperature
    Tt = vec(sum(f0 .* (mesh.vnode .- ubar') .^ 2, dims = 1) .* dv)
    energy1 = sum(sum(ff0 .+ ff2 .+ ff3, dims = 1))
    # total energy
    energy = energy1 + energy2
    # spectrum
    Sz = sum(f3) * dx * dv
    return Ex_energy, E_energy, B_energy, energy, Sz, Tt
end

export kinetic_energy

function kinetic_energy(f0, M::Int, N::Int, L, H)
    v = collect(1:N) .* 2H ./ N .- H .- H ./ N
    e = 0.0
    for j = 1:M, i = 1:N
        e += 0.5 * f0[i, j] * v[i]^2 * L / M * 2H / N
    end
    e
end

export ex_energy

ex_energy(E1, mesh) = 0.5 * sum(real(ifft(E1)) .^ 2) * mesh.dx

export bf_energy

function bf_energy(f1, f2, f3, S1, S2, S3, M, N, L, H, tiK, n_i = 1.0, mub = 0.3386)

    K_xc = tiK
    bb1 = -K_xc * n_i * 0.5 * S1
    bb2 = -K_xc * n_i * 0.5 * S2
    bb3 = -K_xc * n_i * 0.5 * S3

    ebf1 = 0.0
    ebf2 = 0.0
    ebf3 = 0.0

    for j = 1:M, k = 1:N
        ebf1 += mub * f1[k, j] * bb1[j] * L / M * 2H / N
        ebf2 += mub * f2[k, j] * bb2[j] * L / M * 2H / N
        ebf3 += mub * f3[k, j] * bb3[j] * L / M * 2H / N
    end

    ebf1, ebf2, ebf3

end

export s_energy

function s_energy(S1, S2, S3, M, N, L, H, mub = 0.3386, n_i = 1.0)

    h_int = 2.0 * mub
    aj0 = 0.01475 * h_int / 2.0

    k = fftfreq(M, M) .* 2π ./ L

    S1t = fft(S1) .* 1im .* k
    S2t = fft(S2) .* 1im .* k
    S3t = fft(S3) .* 1im .* k

    ifft!(S1t)
    ifft!(S2t)
    ifft!(S3t)

    es1 = sum(aj0 .* n_i .* real(S1t) .^ 2 * L / M)
    es2 = sum(aj0 .* n_i .* real(S2t) .^ 2 * L / M)
    es3 = sum(aj0 .* n_i .* real(S3t) .^ 2 * L / M)

    return es1, es2, es3

end

export snorm

snorm(S1, S2, S3) = maximum(abs.((S1 .^ 2 .+ S2 .^ 2 .+ S3 .^ 2) .- 1))

export energy

function energy(f0, f1, f2, f3, S1, S2, S3, E1, M, N, L, H, tiK, n_i)

    res = sum(s_energy(S1, S2, S3, M, N, L, H))
    res += 0.5 * sum(real(ifft(E1)) .^ 2) * L / M
    res += sum(bf_energy(f1, f2, f3, S1, S2, S3, M, N, L, H, tiK, n_i))
    res += kinetic_energy(f0, M, N, L, H)

    return res

end
