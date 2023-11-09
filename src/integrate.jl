export integrate

"""
$(SIGNATURES)

computation of integral average in each cell using newton-cotes formula
"""
function integrate(value, n)

    integralvalue = 7 / 90 * value[1:5:5n-4] + 16 / 45 * value[2:5:5n-3]
    integralvalue = integralvalue + 2 / 15 * value[3:5:5n-2]
    integralvalue = integralvalue + 16 / 45 * value[4:5:5n-1]
    integralvalue = integralvalue + 7 / 90 * value[5:5:5n]

    return integralvalue

end
