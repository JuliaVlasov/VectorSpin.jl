"""
$(SIGNATURES)
"""
function H1f!(f0, f1, f2, f3, E1, t, L, H)

    N, M = size(f0)

    ff0 = complex(f0')
    ff1 = complex(f1')
    ff2 = complex(f2')
    ff3 = complex(f3')
    
    fft!(ff0, 1)
    fft!(ff1, 1)
    fft!(ff2, 1)
    fft!(ff3, 1)

    k = [0:(M-1)÷2; -(M - 1)÷2:-1] .* 2π / L
    v = (1:N) .* 2 .* H ./ N .- H .- H ./ N

    expv = exp.(- 1im .* k .* v' .* t)

    for i = 2:M
        E1[i] += 1 / (1im * k[i]) * sum(ff0[i,:] .* (expv[i,:] .- 1.0)) * 2H / N
    end

    ff0 .*= expv
    ff1 .*= expv
    ff2 .*= expv
    ff3 .*= expv

    ifft!(ff0, 1)
    ifft!(ff1, 1)
    ifft!(ff2, 1)
    ifft!(ff3, 1)

    f0 .= real(ff0')
    f1 .= real(ff1')
    f2 .= real(ff2')
    f3 .= real(ff3')

end

export H1fOperator

struct H1fOperator

    adv::Translator
    tmp::Matrix{ComplexF64}
    expv::Matrix{ComplexF64}

    H1fOperator(adv) = new(adv, zeros(ComplexF64, adv.mesh.M, adv.mesh.N)
                              , zeros(ComplexF64, adv.mesh.M, adv.mesh.N))

end

"""
$(SIGNATURES)
"""
function step!(f0, f1, f2, f3, E1, op::H1fOperator, t)

    N, M = size(f0)
    L, H = op.adv.mesh.L, op.adv.mesh.H
    k = op.adv.mesh.k
    v = (1:N) .* 2H ./ N .- H .- H ./ N
    op.expv .= exp.(- 1im .* k .* v' .* t)

    transpose!(op.tmp, f0)
    fft!(op.tmp, 1)

    for i = 2:M
        E1[i] += 1 / (1im * k[i]) * sum(op.tmp[i,:] .* (op.expv[i,:] .- 1.0)) * 2H / N
    end

    op.tmp .*= op.expv
    ifft!(op.tmp, 1)
    transpose!( f0, real(op.tmp))

    transpose!(op.tmp, f1)
    fft!(op.tmp, 1)
    op.tmp .*= op.expv
    ifft!(op.tmp, 1)
    transpose!( f1, real(op.tmp))

    transpose!(op.tmp, f2)
    fft!(op.tmp, 1)
    op.tmp .*= op.expv
    ifft!(op.tmp, 1)
    transpose!( f2, real(op.tmp))

    transpose!(op.tmp, f3)
    fft!(op.tmp, 1)
    op.tmp .*= op.expv
    ifft!(op.tmp, 1)
    transpose!( f3, real(op.tmp))

end
