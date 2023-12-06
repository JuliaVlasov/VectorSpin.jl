using .Threads
import LinearAlgebra: transpose!, mul!, ldiv!

export HvSubsystem

struct HvSubsystem{T}

    mesh::Mesh{T}
    f0::Matrix{Complex{T}}
    f1::Matrix{Complex{T}}
    f2::Matrix{Complex{T}}
    f3::Matrix{Complex{T}}
    ff0::Matrix{Complex{T}}
    ff1::Matrix{Complex{T}}
    ff2::Matrix{Complex{T}}
    ff3::Matrix{Complex{T}}
    ev::Matrix{Complex{T}}
    p0::GenericFFT.FFTW.cFFTWPlan{Complex{T}, -1, false, 2, Int64}
    p1::GenericFFT.FFTW.cFFTWPlan{Complex{T}, -1, false, 2, Int64}
    p2::GenericFFT.FFTW.cFFTWPlan{Complex{T}, -1, false, 2, Int64}
    p3::GenericFFT.FFTW.cFFTWPlan{Complex{T}, -1, false, 2, Int64}

    function HvSubsystem(mesh::Mesh{T}) where {T}

        f0 = zeros(Complex{T}, mesh.nx, mesh.nv)
        f1 = zeros(Complex{T}, mesh.nx, mesh.nv)
        f2 = zeros(Complex{T}, mesh.nx, mesh.nv)
        f3 = zeros(Complex{T}, mesh.nx, mesh.nv)

        ff0 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff1 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff2 = zeros(Complex{T}, mesh.nx, mesh.nv)
        ff3 = zeros(Complex{T}, mesh.nx, mesh.nv)

        ev = exp.(-1im .* mesh.kx .* mesh.vnode')

        p0 = plan_fft(f0,  1, flags=GenericFFT.FFTW.PATIENT)    
        p1 = plan_fft(f1,  1, flags=GenericFFT.FFTW.PATIENT)    
        p2 = plan_fft(f2,  1, flags=GenericFFT.FFTW.PATIENT)    
        p3 = plan_fft(f3,  1, flags=GenericFFT.FFTW.PATIENT)    

        new{T}(mesh, f0, f1, f2, f3, ff0, ff1, ff2, ff3, ev, p0, p1, p2, p3)

    end

end

"""
$(SIGNATURES)

subsystem for Hv:

```math
f_t + vf_x = 0
```

```math
-E_x = \\rho - 1
```

"""
function step!(
    op::HvSubsystem{T},
    f0::Matrix{T},
    f1::Matrix{T},
    f2::Matrix{T},
    f3::Matrix{T},
    E1::Vector{Complex{T}},
    dt::T,
) where {T}

    v = op.mesh.vnode
    kx = op.mesh.kx

    @sync begin

        @spawn begin
            transpose!(op.f0, f0)
            mul!(op.ff0, op.p0, op.f0)
            op.ff0 .*= op.ev .^ dt

            E1 .= vec(sum(op.ff0, dims = 2))
            E1[1] = 0.0
            @inbounds for i = 2:op.mesh.nx
                E1[i] *= op.mesh.dv / (-1im * kx[i])
            end

            ldiv!(op.f0, op.p0, op.ff0)
            transpose!(f0, real(op.f0))
        end

        @spawn begin
            transpose!(op.f1, f1)
            mul!(op.ff1, op.p1, op.f1)
            op.ff1 .*= op.ev .^ dt
            ldiv!(op.f1, op.p1, op.ff1)
            transpose!(f1, real(op.f1))
        end

        @spawn begin
            transpose!(op.f2, f2)
            mul!(op.ff2, op.p2, op.f2)
            op.ff2 .*= op.ev .^ dt
            ldiv!(op.f2, op.p2, op.ff2)
            transpose!(f2, real(op.f2))
        end

        @spawn begin
            transpose!(op.f3, f3)
            mul!(op.ff3, op.p3, op.f3)
            op.ff3 .*= op.ev .^ dt
            ldiv!(op.f3, op.p3, op.ff3)
            transpose!(f3, real(op.f3))
        end
    end

end
