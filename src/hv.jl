export Hv!

"""
$(SIGNATURES)

subsystem for Hv:

f_t+vf_x=0

-E_x=\rho-1

"""
function Hv!(f0, f1, f2, f3, t, M, N, L, H)

    ff0 = complex(f0)
    ff1 = complex(f1)
    ff2 = complex(f2)
    ff3 = complex(f3)

    v = collect(1:N) .* 2H ./ N .- H .- H ./ N

    # translate in the direction of x for f0, f1, f2 and f3
    #
    for i = 1:N
        ff0[i, :] .= fft(ff0[i, :])
        ff1[i, :] .= fft(ff1[i, :])
        ff2[i, :] .= fft(ff2[i, :])
        ff3[i, :] .= fft(ff3[i, :])
    end

    f0t = zeros(ComplexF64, N, M)
    fff0t = zeros(ComplexF64, N, M)
    f1t = zeros(ComplexF64, N, M)
    f2t = zeros(ComplexF64, N, M)
    f3t = zeros(ComplexF64, N, M)

    k_fre = fftfreq(M, M)

    for i = 1:N
        for j = 1:M
            f0t[i, j] = ff0[i, j] .* exp(-(2pi / L) * 1im * k_fre[j] * v[i] * t)
            f1t[i, j] = ff1[i, j] .* exp(-(2pi / L) * 1im * k_fre[j] * v[i] * t)
            f2t[i, j] = ff2[i, j] .* exp(-(2pi / L) * 1im * k_fre[j] * v[i] * t)
            f3t[i, j] = ff3[i, j] .* exp(-(2pi / L) * 1im * k_fre[j] * v[i] * t)
        end

        fff0t[i, :] .= f0t[i, :]

        f0t[i, :] .= real(ifft(f0t[i, :]))
        f1t[i, :] .= real(ifft(f1t[i, :]))
        f2t[i, :] .= real(ifft(f2t[i, :]))
        f3t[i, :] .= real(ifft(f3t[i, :]))
    end


    # below we compute E1t

    E1t = zeros(ComplexF64, M)

    for i = 2:((M-1)÷2+1)
        E1t[i] = sum(fff0t[:, i]) * (2H / N) * (-1.0) * L / (1im * 2pi * (i - 1))
    end

    for i = ((M-1)÷2+2):M
        k = i - M - 1
        E1t[i] = sum(fff0t[:, i]) * (2H / N) * (-1.0) * L / (1im * 2pi * k)
    end

    f0 .= real(f0t)
    f1 .= real(f1t)
    f2 .= real(f2t)
    f3 .= real(f3t)

    return E1t

end
