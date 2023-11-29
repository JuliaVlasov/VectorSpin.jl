using .Threads

export HeOperator

struct HeOperator

    adv0::AbstractAdvection
    adv1::AbstractAdvection
    adv2::AbstractAdvection
    adv3::AbstractAdvection
    e::Vector{Float64}

    function HeOperator(adv)

        e = zeros(adv.mesh.nx)
        adv0 = PSMAdvection(adv.mesh)
        adv1 = PSMAdvection(adv.mesh)
        adv2 = PSMAdvection(adv.mesh)
        adv3 = PSMAdvection(adv.mesh)

        new(adv0, adv1, adv2, adv3, e)

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

    @sync begin
       @spawn advection!(f0, op.adv0, op.e, dt)
       @spawn advection!(f1, op.adv1, op.e, dt)
       @spawn advection!(f2, op.adv2, op.e, dt)
       @spawn advection!(f3, op.adv3, op.e, dt)
    end

end


function step!(op::HeOperator, f0, f1, f2, f3, E1, dt)

    op.e .= -real(ifft(E1))

    @sync begin
        @spawn advection!(f0, op.adv0, op.e, dt)
        @spawn advection!(f1, op.adv1, op.e, dt)
        @spawn advection!(f2, op.adv2, op.e, dt)
        @spawn advection!(f3, op.adv3, op.e, dt)
    end

end
