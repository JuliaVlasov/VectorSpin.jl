export init

"""
$(SIGNATURES)

Initialize the distribution function 
"""
function init(k, x, i, v1int, frequency, a, femi, tiK)

    # initial Maxwellian
    kk = 1.0
    if femi ≈ 1
        femi = 1
    else
        femi = 0.5
    end

    f =
        (1.0 / sqrt(pi) / kk) *
        exp(-(v1int[i])^2 / kk / kk) *
        (1.0 + a * cos(frequency * x[k])) *
        femi
    df =
        (1.0 / sqrt(pi) / kk) *
        exp(-(v1int[i])^2 / kk / kk) *
        (a * cos(frequency * x[k])) *
        femi

    return f, df
end
