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
    savevaluef0 = f0
    savevaluef1 = f1

    partialB1 = zeros(ComplexF64, M)
    partial2S1 = zeros(ComplexF64, M)

    value1 = 1:((M-1)÷2+1)
    value2 = ((M-1)÷2+2):M

    B10 = -K_xc * n_i * 0.5 * S1
    fS1 = fft(S1)

    partialB1[value1] .=
        (-((K_xc * n_i * 0.5 * 2pi * 1im / L .* (value1 .- 1))) .* fS1[value1])

    partialB1[value2] .=
        (-((K_xc * n_i * 0.5 * 2pi * 1im / L .* (value2 .- M .- 1))) .* fS1[value2])

    partialB1 = real(ifft(partialB1))

    partial2S1[value1] .= (-((2pi / L .* (value1 .- 1)) .^ 2) .* fS1[value1])

    partial2S1[value2] .= (-((2pi / L .* (value2 .- M .- 1)) .^ 2) .* fS1[value2])

    partial2S1 .= real(ifft(partial2S1))

    translatorv1 = zeros(N, M)
    translatorv2 = zeros(N, M)
    for i = 1:M
        translatorv1[:, i] .= (t * partialB1[i] * mub)
        translatorv2[:, i] .= -translatorv1[:, i]
    end

    u1 = zeros(N, M)
    u2 = zeros(N, M)

    S2t = zeros(M)
    S3t = zeros(M)

    for i = 1:M

        u1[:, i] .= translation(
            0.5 * savevaluef0[:, i] .+ 0.5 * savevaluef1[:, i],
            N,
            translatorv1[:, i],
            H,
        )

        u2[:, i] .= translation(
            0.5 * savevaluef0[:, i] .- 0.5 * savevaluef1[:, i],
            N,
            translatorv2[:, i],
            H,
        )
    end

    f2 .= cos.(t * B10') .* f2 .- sin.(t * B10') .* f3
    f3 .= sin.(t * B10') .* f2 .+ cos.(t * B10') .* f3


    for i = 1:M

        temi = K_xc / 4 * sum(view(f1,:, i)) * 2H / N + 0.01475 * partial2S1[i]

        S2t[i] = cos(t * temi) * S2[i] + sin(t * temi) * S3[i]
        S3t[i] = -sin(t * temi) * S2[i] + cos(t * temi) * S3[i]

    end

    f0 .= u1 .+ u2
    f1 .= u1 .- u2

    return S2t, S3t

end
