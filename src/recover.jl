using LinearAlgebra

"""
$(SIGNATURES)

Given integral average in each cell,this function could compute
coefficients a,b,c of piecewise quadratic polynomial using PSM method
 
a,b,c are all row vectors
"""
function recover(f, N)


    f1 = zeros(N + 1)
    f1[2:N] .= f[1:(N-1)] .+ f[2:N]
    f1[1] = f[1]
    f1[N+1] = f[N]
    diagonal = ones(N + 1)
    upright = 1 / 3 .* ones(N)
    diagonal[1] = 2 / 3
    diagonal[N+1] = 2 / 3
    diagonal[2:N] .= 4 / 3 .* ones(N - 1)
    A = SymTridiagonal(diagonal, upright)
    # get result
    c1 = A \ f1
    c = c1[1:N]

    a = zeros(N)
    b = zeros(N)
    cc = zeros(N)
    for i = 2:N
        cc[i] = (-1)^i * (c1[i] - c1[i-1])
    end
    for i = 2:N
        b[i] = (-1)^i * 2 * sum(cc[2:i])
    end
    b[1] = 0
    a[1:(N-1)] .= 1 / 2 * (b[2:N] - b[1:(N-1)])
    a[N] = -1 / 2 * b[N]

    return a, b, c

end

"""
$(SIGNATURES)

oldvector is the integral average value in each cell of function f(x)
newvector is the integral average value in each cell of function f(x+delta)

"""
function translation(oldvector, N, delta, H)

    # first recover oldvector and get the coefficients of piecewise polynomials

    a, b, c = recover(oldvector, N)

    newvector = zeros(N)

    a, b, c = recover(oldvector, N)

    newvector = zeros(N)

    for i = 1:N

        beta = i + delta[i] / (2H / N)
        loopnumber = floor(Int, beta / N)
        newbeta = beta - N * loopnumber

        if (abs(newbeta) ≈ 0.0) || (abs(newbeta - N) ≈ 0.0)
            newvector[i] = oldvector[N]
        elseif newbeta >= 1.0
            index = floor(Int, newbeta)
            k = 1 - (newbeta - index)
            valueI = a[index] / 3 + b[index] / 2 + c[index]
            valueI = valueI - a[index] / 3 * (1 - k)^3 - b[index] / 2 * (1 - k)^2
            valueI = valueI - c[index] * (1 - k)
            valueII = a[index+1] / 3 * (1 - k)^3 + b[index+1] / 2 * (1 - k)^2
            valueII = valueII + c[index+1] * (1 - k)
            newvector[i] = valueI + valueII

        else
            index = N
            k = 1 - newbeta
            valueI = a[index] / 3 + b[index] / 2 + c[index]
            valueI = valueI - a[index] / 3 * (1 - k)^3 - b[index] / 2 * (1 - k)^2
            valueI = valueI - c[index] * (1 - k)
            valueII = a[1] / 3 * (1 - k)^3 + b[1] / 2 * (1 - k)^2 + c[1] * (1 - k)
            newvector[i] = valueI + valueII
        end

    end

    return newvector
end
