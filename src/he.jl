"""
$(SIGNATURES)
 
compute the first subsystem He()
- M is odd number
"""
function He!(f0, f1, f2, f3, E1, E2, E3, A2, A3, t, H)

    A2 .= A2 .- t .* E2
    A3 .= A3 .- t .* E3

    e = -t .* real(ifft(E1))

    translation!(f0, e, H)
    translation!(f1, e, H)
    translation!(f2, e, H)
    translation!(f3, e, H)

end

export HeOperator

struct HeOperator

    adv::AbstractAdvection
    e::Vector{Float64}

    function HeOperator(adv)

        e = zeros(adv.mesh.nx)

        new(adv, e)

    end

end


"""
$(SIGNATURES)
 
compute the first subsystem He()
- M is odd number
"""
function step!(f0, f1, f2, f3, E1, E2, E3, A2, A3, op::HeOperator, dt)

    A2 .-= dt .* E2
    A3 .-= dt .* E3

    op.e .= - real(ifft(E1))

    advection!(f0, op.adv, op.e, dt)
    advection!(f1, op.adv, op.e, dt)
    advection!(f2, op.adv, op.e, dt)
    advection!(f3, op.adv, op.e, dt)

end
