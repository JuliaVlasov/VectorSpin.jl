export H1fh!

"""
$(SIGNATURES)

compute the subsystem Hs1
M is even number

"""
function H1fh!(f0, f1, f2, f3, S1, S2, S3, t, M, N, L, H, tiK)

    K_xc = tiK
    n_i = 1.0
    mub = 0.3386

    k = fftfreq(M, M)

    B10 = -K_xc * n_i * 0.5 * S1
    fS1 = fft(S1)

    partialB1 = -((K_xc * n_i * 0.5 * 2pi * 1im / L .* k)) .* fS1
    ifft!(partialB1)

    partial2S1 = (-((2pi / L .* k )) .^ 2) .* fS1
    ifft!(partial2S1)

    v1 = zeros(M)
    v2 = zeros(M)
    for i = 1:M
        v1[i] = (t * real(partialB1[i]) * mub)
        v2[i] = -v1[i]
    end


    u1 =  0.5 .* f0 .+ 0.5 .* f1
    u2 =  0.5 .* f0 .- 0.5 .* f1

    translation!(u1, v1, H)
    translation!(u2, v2, H)

    f2 .= cos.(t * B10') .* f2 .- sin.(t * B10') .* f3
    f3 .= sin.(t * B10') .* f2 .+ cos.(t * B10') .* f3


    S2t = similar(S1)
    S3t = similar(S2)
    for i = 1:M

        temi = K_xc / 4 * sum(view(f1,:, i)) * 2H / N + 0.01475 * real(partial2S1[i])

        S2t[i] = cos(t * temi) * S2[i] + sin(t * temi) * S3[i]
        S3t[i] = -sin(t * temi) * S2[i] + cos(t * temi) * S3[i]

    end

    f0 .= u1 .+ u2
    f1 .= u1 .- u2

    return S2t, S3t

end
