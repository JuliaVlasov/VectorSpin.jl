# Example


```@example test
using Plots
using FFTW
using VectorSpin

function run()

    T = 50 # 4000  # final time
    nx = 65   # partition of x
    nv = 129   # partition of v
    vmin, vmax = -2.5, 2.5  
    ke = 1.2231333040331807  
    xmin, xmax = 0.0, 4pi / ke  
    dt = 0.04
    nsteps = floor(Int, T / dt + 1.1) 
    a = 0.02 # 0.001; perturbation coefficient
    h_int = 0.2 # hbar
    k0 = 2.0 * ke
    ww = sqrt(1.0 + k0^2.0) # w0
    ata = 0.2
    vth = 0.17

    mesh = Mesh(xmin, xmax, nx, vmin, vmax, nv)
    adv  = PSMAdvection(mesh)
    
    E1, E2, E3, A2, A3 = initialfields( mesh, a, ww, ke, k0)
    f0, f1, f2, f3 = initialfunction(mesh, a, ke, vth,  ata)

    results = Diagnostics(f0, f2, f3, E1, E2, E3, A2, A3, mesh, h_int)

    H2fh = H2fhOperator(adv)
    He = HeOperator(adv)
    HAA = HAAOperator(adv)
    H3fh = H3fhOperator(adv)
    H1f = H1fOperator(adv)

    for i = 1:nsteps # Loop over time

        step!(f0, f1, f2, f3, E3, A3, H2fh, 0.5dt, h_int)
        step!(f0, f1, f2, f3, E1, E2, E3, A2, A3, He, 0.5dt)
        step!(f0, f1, f2, f3, E2, E3, A2, A3, HAA, 0.5dt)
        step!(f0, f1, f2, f3, E2, A2, H3fh, 0.5dt, h_int)
        step!(f0, f1, f2, f3, E1, H1f, dt)
        step!(f0, f1, f2, f3, E2, A2, H3fh, 0.5dt, h_int)
        step!(f0, f1, f2, f3, E2, E3, A2, A3, HAA, 0.5dt)
        step!(f0, f1, f2, f3, E1, E2, E3, A2, A3, He, 0.5dt)
        step!(f0, f1, f2, f3, E3, A3, H2fh, 0.5dt, h_int)
        
        save!(results, i*dt, f0, f2, f3, E1, E2, E3, A2, A3)

    end

    results

end
```

```@example test
data = run()
```

```@example test
plot(data.time, log.(data.Ex_energy))
```

```@example test
plot(data.time, data.E_energy)
```

```@example test
plot(data.time, data.B_energy)
```

```@example test
plot(data.time, data.energy)
```

```@example test
plot(data.time, data.Sz)
```
