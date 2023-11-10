using LinearAlgebra

"""
$(SIGNATURES)

interpolate df(x - delta)
"""
function translation!(df, delta::Vector{Float64}, H)

    # first recover oldvector & get the coefficients of piecewise polynomials

    N, M = size(df)
    @assert length(delta) == M

    a = zeros(N)
    b = zeros(N)
    c = zeros(N + 1)
    cc = zeros(N)
    diagonal = ones(N + 1)
    upright = 1 / 3 .* ones(N)
    diagonal[1] = 2 / 3
    diagonal[N+1] = 2 / 3
    diagonal[2:N] .= 4 / 3
    A = SymTridiagonal(diagonal, upright)

    @inbounds for j = 1:M

        for i = 2:N
            c[i] = df[i-1, j] + df[i, j]
        end
        c[1] = df[1, j]
        c[N+1] = df[N, j]

        c .= A \ c

        for i = 2:N
            cc[i] = (-1)^i * (c[i] - c[i-1])
        end
        for i = 2:N
            b[i] = (-1)^i * 2 * sum(view(cc, 2:i))
        end
        b[1] = 0
        for i = 1:N-1
            a[i] = 0.5 * (b[i+1] - b[i])
        end
        a[N] = -0.5 * b[N]

        alpha = delta[j] / (2H / N)

        for i = 1:N

            beta::Float64 = i + alpha
            newbeta = beta - N * floor(Int, beta / N)

            if newbeta >= 1.0
                l = floor(Int, newbeta)
                k = 1 - (newbeta - l)
            else
                l = N
                k = 1 - newbeta
            end

            l1 = mod1(l + 1, N)
            k1 = 1 - k
            k2 = (1 - k)^2
            k3 = (1 - k)^3

            val = a[l] / 3 + 0.5 * b[l] + c[l]
            val += -a[l] / 3 * k3 - 0.5 * b[l] * k2
            val += -c[l] * k1
            val += a[l1] / 3 * k3 + 0.5 * b[l1] * k2
            val += c[l1] * k1

            df[i, j] = val

        end
    end

end
