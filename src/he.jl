using .Threads

export HeOperator

struct HeOperator{T}

    adv0::AbstractAdvection
    adv1::AbstractAdvection
    adv2::AbstractAdvection
    adv3::AbstractAdvection
    e::Vector{T}

    function HeOperator(mesh::Mesh{T}) where {T}

        e = zeros(T, mesh.nx)
        adv0 = PSMAdvection(mesh)
        adv1 = PSMAdvection(mesh)
        adv2 = PSMAdvection(mesh)
        adv3 = PSMAdvection(mesh)

        new{T}(adv0, adv1, adv2, adv3, e)

    end

end


"""
$(SIGNATURES)
 
compute the first subsystem He()

```math
f_t - Ef_v = 0
```

"""
function step!(
    op::HeOperator{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    E1::Vector{Complex{T}},
    E2::Vector{Complex{T}},
    E3::Vector{Complex{T}},
    A2::Vector{Complex{T}},
    A3::Vector{Complex{T}},
    dt::T,
) where {T}

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


function step!(
    op::HeOperator{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    E1::Vector{Complex{T}},
    dt::T,
) where {T}

    op.e .= -real(ifft(E1))

    @sync begin
        @spawn advection!(f0, op.adv0, op.e, dt)
        @spawn advection!(f1, op.adv1, op.e, dt)
        @spawn advection!(f2, op.adv2, op.e, dt)
        @spawn advection!(f3, op.adv3, op.e, dt)
    end

end
