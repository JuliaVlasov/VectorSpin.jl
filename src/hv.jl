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

    for j in eachindex(k_fre), i in eachindex(v)

        expv = exp(- 1im * k_fre[j] * v[i] * t)

        ff0[j, i] *= expv
        ff1[j, i] *= expv
        ff2[j, i] *= expv
        ff3[j, i] *= expv

    end

    E1t = zeros(ComplexF64, M)

    for i = 2:((M-1)÷2+1)
        E1t[i] = sum(view(ff0,i,:)) * (2H / N) * (-1.0) * L / (1im * 2pi * (i - 1))
    end

    for i = ((M-1)÷2+2):M
        k = i - M - 1
        E1t[i] = sum(view(ff0,i, :)) * (2H / N) * (-1.0) * L / (1im * 2pi * k)
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
