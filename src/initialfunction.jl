function f(x, v, kx, α; σ = 0.17)

    return exp(-0.5 * v^2 / σ^2) * (1 + α * cos(kx * x)) / sqrt(2π) / σ

end

export initialfunction

"""
$(SIGNATURES)

Gaussian function with a perturbation in x direction
"""
function initialfunction(mesh::Mesh{T}, α, kx, σ, ata) where {T}

    xmin, xmax = mesh.xmin, mesh.xmax
    vmin, vmax = mesh.vmin, mesh.vmax
    nx, nv = mesh.nx, mesh.nv
    dv = mesh.dv

    f0 = zeros(T, nv, nx)
    f1 = zeros(T, nv, nx)
    f2 = zeros(T, nv, nx)
    f3 = zeros(T, nv, nx)

    for k = 1:nx, i = 1:nv
        v1 = mesh.v[i] - dv
        v2 = mesh.v[i] - dv * 0.75
        v3 = mesh.v[i] - dv * 0.50
        v4 = mesh.v[i] - dv * 0.25
        v5 = mesh.v[i]

        y1 = f(mesh.x[k], v1, kx, α, σ = σ)
        y2 = f(mesh.x[k], v2, kx, α, σ = σ)
        y3 = f(mesh.x[k], v3, kx, α, σ = σ)
        y4 = f(mesh.x[k], v4, kx, α, σ = σ)
        y5 = f(mesh.x[k], v5, kx, α, σ = σ)

        f0[i, k] = (7y1 + 32y2 + 12y3 + 32y4 + 7y5) / 90
    end

    f3 .= (ata / 3.0) .* f0

    return f0, f1, f2, f3

end
