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
"""
function step!( op::HeOperator, f0, f1, f2, f3, E1, E2, E3, A2, A3, dt)

    A2 .-= dt .* E2
    A3 .-= dt .* E3

    op.e .= real(ifft(E1))

    advection!(f0, op.adv, op.e, dt)
    advection!(f1, op.adv, op.e, dt)
    advection!(f2, op.adv, op.e, dt)
    advection!(f3, op.adv, op.e, dt)

end

export He!

"""
$(SIGNATURES)

subsystem for He:

f_t-Ef_v=0;

"""
function He!(f0, f1, f2, f3, E1, t, H)

    e = t .* real(ifft(E1))

    translation!(f0, e, H)
    translation!(f1, e, H)
    translation!(f2, e, H)
    translation!(f3, e, H)

end

function step!(op::HeOperator, f0, f1, f2, f3, E1, dt)

    op.e .= - real(ifft(E1))

    advection!(f0, op.adv, op.e, dt)
    advection!(f1, op.adv, op.e, dt)
    advection!(f2, op.adv, op.e, dt)
    advection!(f3, op.adv, op.e, dt)

end

