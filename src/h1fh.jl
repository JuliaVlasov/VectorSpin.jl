export H1fhOperator

struct H1fhOperator

    

end

export H1fh!

"""
$(SIGNATURES)

compute the subsystem Hs1
M is even number

"""
function H1fh!(f0, f1, f2, f3, S1, S2, S3, dt, mesh, tiK)

    K_xc = tiK
    n_i = 1.0
    mub = 0.3386

    B10 = -K_xc * n_i * 0.5 * S1
    fS1 = fft(S1)

    tmp = -(K_xc * n_i * 0.5 * 1im .* mesh.kx) .* fS1
    ifft!(tmp)
    v1 = zeros(mesh.nx)
    v2 = zeros(mesh.nx)
    for i = 1:mesh.nx
        v1[i] = (dt * real(tmp[i]) * mub)
        v2[i] = -v1[i]
    end

    tmp .= -mesh.kx .^ 2 .* fS1
    ifft!(tmp)

    u1 = 0.5 .* f0 .+ 0.5 .* f1
    u2 = 0.5 .* f0 .- 0.5 .* f1

    H = 0.5 * ( mesh.vmax - mesh.vmin)
    translation!(u1, v1, mesh)
    translation!(u2, v2, mesh)

    f2 .= cos.(dt * B10') .* f2 .- sin.(dt * B10') .* f3
    f3 .= sin.(dt * B10') .* f2 .+ cos.(dt * B10') .* f3

    S2t = similar(S1)
    S3t = similar(S2)
    for i = 1:mesh.nx

        temi = K_xc / 4 * sum(view(f1, :, i)) * mesh.dv + 0.01475 * real(tmp[i])

        S2t[i] = cos(dt * temi) * S2[i] + sin(dt * temi) * S3[i]
        S3t[i] = -sin(dt * temi) * S2[i] + cos(dt * temi) * S3[i]

    end

    f0 .= u1 .+ u2
    f1 .= u1 .- u2

    S2 .= S2t
    S3 .= S3t

end

