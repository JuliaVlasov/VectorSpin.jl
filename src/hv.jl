export Hv!

"""
$(SIGNATURES)

subsystem for Hv:

f_t+vf_x=0

-E_x=\rho-1

"""
function Hv!(f0, f1, f2, f3, t, M, N, L, H)

    ff0 = complex(f0')
    ff1 = complex(f1')
    ff2 = complex(f2')
    ff3 = complex(f3')

    v = collect(1:N) .* 2H ./ N .- H .- H ./ N

    # translate in the direction of x for f0, f1, f2 and f3

    fft!(ff0, 1)
    fft!(ff1, 1)
    fft!(ff2, 1)
    fft!(ff3, 1)

    k_fre = fftfreq(M, M) .* 2π ./ L

    for i in eachindex(v), j in eachindex(k_fre)

        expv = exp(-1im * k_fre[j] * v[i] * t)

        ff0[j, i] *= expv
        ff1[j, i] *= expv
        ff2[j, i] *= expv
        ff3[j, i] *= expv

    end

    E1t = vec(sum(ff0, dims = 2))
    E1t[1] = 0.0

    for i = 2:M
        E1t[i] *= (2H / N) / (-1im * k_fre[i])
    end

    ifft!(ff0, 1)
    ifft!(ff1, 1)
    ifft!(ff2, 1)
    ifft!(ff3, 1)

    f0 .= real(ff0')
    f1 .= real(ff1')
    f2 .= real(ff2')
    f3 .= real(ff3')

    return E1t

end
