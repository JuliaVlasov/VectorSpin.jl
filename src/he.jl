using .Threads

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

```math
f_t - Ef_v = 0
```

"""
function step!(op::HeOperator, f0, f1, f2, f3, E1, E2, E3, A2, A3, dt)

    A2 .-= dt .* E2
    A3 .-= dt .* E3

    op.e .= real(ifft(E1))

    advection!(f0, op.adv, op.e, dt)
    advection!(f1, op.adv, op.e, dt)
    advection!(f2, op.adv, op.e, dt)
    advection!(f3, op.adv, op.e, dt)

end


function step!(op::HeOperator, f0, f1, f2, f3, E1, dt)

    op.e .= -real(ifft(E1))

    advection!(f0, op.adv, op.e, dt)
    advection!(f1, op.adv, op.e, dt)
    advection!(f2, op.adv, op.e, dt)
    advection!(f3, op.adv, op.e, dt)

end
