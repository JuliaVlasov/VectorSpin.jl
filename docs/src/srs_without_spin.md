# SRS without spin 

```julia
using Plots
using GenericFFT
using VectorSpin
```

```math
\left\{
\begin{aligned}
&\frac{\partial f}{\partial t} + p \frac{\partial f}{\partial x} + [ E_x  - {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x} ]\frac{\partial f}{\partial p} = 0,\\
&\frac{\partial E_x}{\partial t} = -\int_{\mathbb{R}} p f  \mathrm{d}{p},\\
&\frac{\partial E_y}{\partial t} = - \frac{\partial^2 A_y}{\partial x^2} + A_y \int_{\mathbb{R}}  f  \mathrm{d}{p},\\
&\frac{\partial E_z}{\partial t} = - \frac{\partial^2 A_z}{\partial x^2} + A_z \int_{\mathbb{R}}  f  \mathrm{d}{p},\\
&\frac{\partial {\mathbf A}_\perp}{\partial t} = - {\mathbf E}_\perp,\\
&\frac{\partial E_x}{\partial x} = \int_{\mathbb{R}} f \mathrm{d}{p} - 1.
\end{aligned}
\right.
```

We consider the periodic condition with spatial period ``L=4\pi/k_e``, also take
``H=5`` for the computational domain in ``v``-direction.

Mathematical domain parameters are taken as ``N_x=129, N_v=129, \Delta t =0.05``

We take the following values for physical parameters:

```math
\alpha=0.02, k_e=1.2231, k_0=2k_e, v_{th}=0.17,
```

```math
w_0=2.6428, k_s=k_e, w_s=1.5799, w_e=1.0629. 
```

We use a perturbed Maxwellian as an initial condition for ``f``

```math
f(t=0,x,p)=(1+\alpha \cos(k_e x))\frac{1}{\sqrt{2\pi}v_{th}}e^{-\frac{p^2}{2v_{th}^2}},
```

and the initial longitudinal electric field
```math
E_x(t=0,x)=(\alpha /k_e)\sin(k_e x). 
```

Here ``\alpha`` and ``k_e`` are the amplitude and the wave number
of the perturbation respectively, and the ``v_{th}`` is the electron
thermal speed. For the transverse fields, we consider an incident
electromagnetic wave with circular polarization:

```math
\begin{aligned}
& E_y(t=0,x)=E_0 \cos(k_0 x), \\
& E_z(t=0,x)=E_0 \sin(k_0 x),\\
& A_y(t=0,x)=-\frac{E_0}{w_0} \sin(k_0 x), \\ 
& A_z(t=0,x)=\frac{E_0}{w_0} \cos(k_0 x),
\end{aligned} 
```

where the ``k_0`` and ``w_0`` are the wave number and the amplitude
of the transverse electric field respectively. 

We also take the amplitude of the incident wave ``E_{ref}=0.325``
as a reference value. 
be in the range ``0.25E_{ref} \leq E_0  \leq 2E_{ref}.``


time evolution of the longitudinal electric field norm
```math
|| E_x (t)|| =\left(\frac{1}{2}\int_0^L E_x^2(t,x) \mathrm{d}\mathrm{x}\right )^{\frac{1}{2}}
```

