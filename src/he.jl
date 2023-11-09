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

    op.e .= real(ifft(E1))

    advection!(f0, op.adv, op.e, dt)
    advection!(f1, op.adv, op.e, dt)
    advection!(f2, op.adv, op.e, dt)
    advection!(f3, op.adv, op.e, dt)

end


"""
$(SIGNATURES)

subsystem for He:

f_t-Ef_v=0;

"""
function He!(f0, f1, f2, f3, E1, t, M :: Int, N :: Int, H)

    e = t .* real(ifft(E1))

    # translate in the direction of v1

    for j = 1:M
        f0[:, j] .= translation(f0[:, j], N, e[j] .* ones(N), H)
        f1[:, j] .= translation(f1[:, j], N, e[j] .* ones(N), H)
        f2[:, j] .= translation(f2[:, j], N, e[j] .* ones(N), H)
        f3[:, j] .= translation(f3[:, j], N, e[j] .* ones(N), H)
    end

end
