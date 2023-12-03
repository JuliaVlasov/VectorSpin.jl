export initialfields

function initialfields(mesh, a, ww, frequency, k0)

    E0 = 0.123 * ww # Eref
    E1 = fft(a ./ frequency .* sin.(frequency .* mesh.x))
    E2 = fft(E0 .* cos.(k0 .* mesh.x))
    E3 = fft(E0 .* sin.(k0 .* mesh.x))
    A2 = -fft(E0 ./ ww .* sin.(k0 .* mesh.x))
    A3 = fft(E0 ./ ww .* cos.(k0 .* mesh.x))

    return E1, E2, E3, A2, A3

end
