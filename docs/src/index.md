# VectorSpin

This package is designed to solve Vlasov system with spin effects. 

This code is used in the paper [crouseilles_hervieux_hong_manfredi_2023](@cite). 

This is the vectorial version
of the [SpinGEMPIC.jl](https://github.com/juliavlasov/SpinGEMPIC.jl) code that has been used 
for the paper [crouseilles_hervieux_li_manfredi_sun_2021](@cite)

```@bibliography
```

## Scalar spin laser plasma model

Particle distribution function ``f(x, p, {\mathbf s}, t)``, 
- ``x\in [0,L]``, 
- ``p \in \mathbb{R}`` are scalars, 
- ``{\mathbf s}=(s_1,s_2,s_3) \in \mathbb{R}^3``, 
- ``{\mathbf E} = (E_x, {\mathbf E}_\perp) = (E_x, E_y, E_z)``, 
- ``{\mathbf A} = (A_x, {\mathbf A}_\perp) = (0, A_y, A_z)``, 
- ``{\mathbf B} =\nabla\times{\mathbf  A} = (0,- \partial_xA_z,  \partial_xA_y)``.

The scalar spin Vlasov-Maxwell system is:

```math
\left\{
\begin{aligned}
&\frac{\partial f}{\partial t} + p \frac{\partial f}{\partial x} + [ E_x - \mathfrak{h} s_2 \frac{\partial^2 A_z}{\partial x^2} + \mathfrak{h} s_3 \frac{\partial^2 A_y}{\partial x^2}  - {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x} ]\frac{\partial f}{\partial p}  \\ 
& \hspace{3cm}+ [s_3 \frac{\partial A_z}{\partial x} + s_2 \frac{\partial A_y}{\partial x}, -s_1 \frac{\partial A_y}{\partial x}, -s_1 \frac{\partial A_z}{\partial x} ] \cdot \frac{\partial f}{\partial {\mathbf s}} = 0,\\
&\frac{\partial E_x}{\partial t} = -\int_{\mathbb{R}^4} p f  \mathrm{d}{p}\mathrm{d}\mathrm{\mathbf s},\\
&\frac{\partial E_y}{\partial t} = - \frac{\partial^2 A_y}{\partial x^2} + A_y \int_{\mathbb{R}^4}  f  \mathrm{d}{p}\mathrm{d}\mathrm{\mathbf s} + \mathfrak{h}\int_{\mathbb{R}^4} s_3 \frac{\partial f}{\partial x}\mathrm{d}{p}\mathrm{d}\mathrm{\mathbf s},\\
&\frac{\partial E_z}{\partial t} = - \frac{\partial^2 A_z}{\partial x^2} + A_z \int_{\mathbb{R}^4}  f  \mathrm{d}{p}\mathrm{d}\mathrm{\mathbf s} - \mathfrak{h}\int_{\mathbb{R}^4} s_2 \frac{\partial f}{\partial x}\mathrm{d}{p}\mathrm{d}\mathrm{\mathbf s},\\
& \frac{\partial {\mathbf A}_\perp}{\partial t} = - {\mathbf E}_\perp,\\
&\frac{\partial E_x}{\partial x} = \int_{\mathbb{R}^4} f \mathrm{d}{p}\mathrm{d}\mathrm{\mathbf s} - 1. \ \text{(Poisson equation)}
\end{aligned}
\right.
```

The system numerically solve is the vector model:

```math
f(t, x,p,{\mathbf{s}})=\frac{1}{4\pi}(f_0(t, x,p)+3s_1f_1(t, x,p)+3s_2f_2(t, x,p)+3s_3f_3(t, x,p)).
```

```math
\left\{
\begin{aligned}
&\frac{\partial f_0}{\partial t} + p \frac{\partial f_0}{\partial x} + \left(E_x - {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x}  \right) \frac{\partial f_0}{\partial p} - \mathfrak{h}\frac{\partial^2 A_z}{\partial x^2}\frac{\partial f_2}{\partial p} +  \mathfrak{h}\frac{\partial^2 A_y}{\partial x^2} \frac{\partial f_3}{\partial p}  = 0,\\
&\frac{\partial f_1}{\partial t} + p \frac{\partial f_1}{\partial x} + \left(E_x - {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x}  \right) \frac{\partial f_1}{\partial p}
 - \frac{\partial A_z }{\partial x}  f_3  -  \frac{\partial A_y }{\partial x} f_2 = 0,\\
& \frac{\partial f_2}{\partial t} + p \frac{\partial f_2}{\partial x} + \left(E_x - {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x}  \right) \frac{\partial f_2}{\partial p} - {\frac{\mathfrak{h}}{3}} \frac{\partial^2 A_z}{\partial x^2}\frac{\partial f_0}{\partial p}
  +  \frac{\partial A_y }{\partial x} f_1 = 0,\\
 & \frac{\partial f_3}{\partial t} + p \frac{\partial f_3}{\partial x} + \left(E_x - {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x}  \right) \frac{\partial f_3}{\partial p} + {\frac{\mathfrak{h}}{3}}  \frac{\partial^2 A_y}{\partial x^2}\frac{\partial f_0}{\partial p}
  +  \frac{\partial A_z }{\partial x} f_1 = 0,\\
&\frac{\partial E_x}{\partial t} = -\int_{\mathbb{R}} p f_0  \mathrm{d}\mathrm{p},\\
&\frac{\partial E_y}{\partial t} = - \frac{\partial^2 A_y}{\partial x^2} + A_y \int_{\mathbb{R}}  f_0  \mathrm{d}\mathrm{p} + \mathfrak{h}\int_{\mathbb{R}}  \frac{\partial f_3}{\partial x}\mathrm{d}\mathrm{p},\\
&\frac{\partial E_z}{\partial t} = - \frac{\partial^2 A_z}{\partial x^2} + A_z \int_{\mathbb{R}}  f_0  \mathrm{d}\mathrm{p} -\mathfrak{h} \int_{\mathbb{R}}  \frac{\partial f_2}{\partial x}\mathrm{d}\mathrm{p},\\
& \frac{\partial {\mathbf A}_\perp}{\partial t} = - {\mathbf E}_\perp,\\
&\frac{\partial E_x}{\partial x} = \int_{\mathbb{R}} f_0 \mathrm{d}\mathrm{p} - 1.\ \text{(Poisson equation)}
\end{aligned}
\right.
```
