# Hamiltonian splitting

```math
\mathcal{H}=\mathcal{H}_{p}+\mathcal{H}_{A}+\mathcal{H}_{E}+\mathcal{H}_{2}+\mathcal{H}_{3},
```

where 

```math
\begin{aligned}
\mathcal{H}_{p} & = \frac{1}{2}\int p^2 f_0 \mathrm{d}{ x}\mathrm{d}{p},\\
\mathcal{H}_{A} &= \frac{1}{2}\int |{\mathbf A}_\perp|^2 f_0 \mathrm{d}{ x}\mathrm{d}{ p}+\frac{1}{2}\int \left|\frac{\partial {\mathbf A}_\perp}{\partial x}\right|^2 \mathrm{d}{x},\\
\mathcal{H}_{E} &= \frac{1}{2}\int |{\mathbf E}|^2  \mathrm{d}{ x}=\frac{1}{2}\int (E_x^2+|{\mathbf E_\perp}|^2 ) \mathrm{d}{ x}, \\
\mathcal{H}_{2} &= \int_{\Omega}  \mathfrak{h} f_2 \frac{\partial A_z}{\partial x} \mathrm{d}x\mathrm{d}p,\\
\mathcal{H}_{3} &= \int_{\Omega}  -\mathfrak{h} f_3 \frac{\partial A_y}{\partial x} \mathrm{d}x\mathrm{d}p.
\end{aligned} 
```

## Subsystem for ``\mathcal{H}_p``

The subsystem
``\frac{\partial \mathcal{Z}}{\partial t} = \{ \mathcal{Z}, \mathcal{H}_p \}``
associated to
``\mathcal{H}_{p} = \frac{1}{2}\int p^2 f_0 \mathrm{d}{ x}\mathrm{d}{p}``
is 

```math
\left\{
\begin{aligned}
&\frac{\partial f_0}{\partial t} = \{f_0, \mathcal{H}_{p} \} = -p\frac{\partial f_0}{\partial x}, \\
&\frac{\partial \mathbf{f}}{\partial t} = \{\mathbf{f}, \mathcal{H}_{p} \}= -p\frac{\partial \mathbf{f}}{\partial x}, \\
%&\frac{\partial f_1}{\partial t} = \{f_1, \mathcal{H}_{p} \}= -p\frac{\partial f_1}{\partial x}, \\
%&\frac{\partial f_2}{\partial t} = \{f_2, \mathcal{H}_{p} \}= -p\frac{\partial f_2}{\partial x}, \\
%&\frac{\partial f_3}{\partial t} = \{f_3, \mathcal{H}_{p} \}= -p\frac{\partial f_3}{\partial x},\\
& \frac{\partial E_x}{\partial t} = \{ E_x, \mathcal{H}_{p} \} =- \int_{\mathbb{R}} p f_0\mathrm{d}{p},\\
& \frac{\partial {\mathbf E}_\perp}{\partial t} =\frac{\partial {\mathbf A}_\perp}{\partial t} =0. 
%& \frac{\partial {\mathbf E}_\perp}{\partial t} =  \{ {\mathbf E}_\perp, \mathcal{H}_{p} \}  = {\mathbf 0},\;\; \frac{\partial {\mathbf A}_\perp}{\partial t} =  \{ {\mathbf A}_\perp, \mathcal{H}_{p} \}  = {\mathbf 0}.
\end{aligned}
\right.
```


We denote the initial value as
``(f_0^0(x,p),\mathbf{f}^0(x,p), E_x^0(x),{\mathbf E}_\perp^0(x),{\mathbf A}_\perp^0(x))``
at time ``t=0``. The solution at time ``t`` of this subsystem can be written
explicitly, 

```math
\begin{aligned}
        &f_0(x,p,t)=f_0^0(x-pt,p), \;\;  \mathbf{f}(x,p,t)=\mathbf{f}^0(x-pt,p),\\ %f_2(x,p,t)=f_2^0(x-pt,p), f_3(x,p,t)=f_3^0(x-pt,p), \\
        &E_x(x,t)=E_x^0(x)-\int_0^t\int_{\mathbb{R}} pf_0(x,p,\tau) \mathrm{d}p\mathrm{d}\tau=E_x^0(x)-\int_0^t\int_{\mathbb{R}} pf_0^0(x-p\tau,p) \mathrm{d}p\mathrm{d}\tau, \\
        &{\mathbf E}_\perp(x,t)={\mathbf E}_\perp^0(x), \;\; {\mathbf A}_\perp(x,t)={\mathbf A}_\perp^0(x). 
    \end{aligned}
```

Next, we check that the solution propagates the
Poisson equation. To do so, we assume that the Poisson equation holds
initially, i.e.

```math
\frac{\partial E_x^0}{\partial x}=\int_{\mathbb{R}} f_0^0\mathrm{d}{p}-1.
```

Then we have, by differentiating the expression of ``E_x(t, x)`` with
respect to ``x`` 

```math
\begin{aligned}
\frac{\partial E_x(x,t)}{\partial x}&=\frac{\partial E_x^0}{\partial x}-\int_0^t\int_{\mathbb{R}} p\frac{\partial f_0^0(x-p\tau,p)}{\partial x} \mathrm{d}p\mathrm{d}\tau=\frac{\partial E_x^0}{\partial x}+\int_0^t\int_{\mathbb{R}} \frac{\partial f_0^0(x-p\tau,p)}{\partial \tau} \mathrm{d}p\mathrm{d}\tau\\
&=\frac{\partial E_x^0}{\partial x}+\int_{\mathbb{R}}  f_0^0(x-pt,p)\mathrm{d}p-\int_{\mathbb{R}}  f_0^0(x,p)\mathrm{d}p=\int_{\mathbb{R}}  f_0(x,p,t)\mathrm{d}p-1, 
\end{aligned}
```

which proves that the Poisson equation is satisfied at
time ``t``.

## Subsystem for ``\mathcal{H}_A``

The subsystem
``\frac{\partial \mathcal{Z}}{\partial t} = \{ \mathcal{Z}, \mathcal{H}_A \}``
associated to the sub-Hamiltonian
``\mathcal{H}_{A} = \frac{1}{2}\int |{\mathbf A}_\perp|^2 f_0 \mathrm{d}{\mathbf x}\mathrm{d}{\mathbf p}+\frac{1}{2}\int |\frac{\partial {\mathbf A}_\perp}{\partial x}|^2 \mathrm{d}{\mathbf x}``
is 

```math
\left\{
\begin{aligned}
&\frac{\partial f_0}{\partial t} = \{f_0, \mathcal{H}_{A} \} = {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x}\frac{\partial f_0}{\partial p}, \\
&\frac{\partial {\mathbf f}}{\partial t} = \{{\mathbf f}, \mathcal{H}_{A} \}= {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x} \frac{\partial {\mathbf f}}{\partial p}, \\
%&\frac{\partial f_2}{\partial t} = \{f_2, \mathcal{H}_{A} \}= {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x} \frac{\partial f_2}{\partial p}, \\
%&\frac{\partial f_3}{\partial t} = \{f_3, \mathcal{H}_{A} \}= {\mathbf A}_\perp \cdot \frac{\partial {\mathbf A}_\perp}{\partial x} \frac{\partial f_3}{\partial p},\\
%& \frac{\partial E_x}{\partial t} = \{ E_x, \mathcal{H}_{A} \} = 0,\\
& \frac{\partial {\mathbf E}_\perp}{\partial t} =  \{ {\mathbf E}_\perp, \mathcal{H}_{A} \}  = - \frac{\partial^2 {\mathbf A}_\perp}{\partial x^2} + {\mathbf A}_\perp \int_{\mathbb{R}}  f_0  \mathrm{d}\mathrm{p},\\
& \frac{\partial E_x}{\partial t} =\frac{\partial {\mathbf A}_\perp}{\partial t} = 0. % \{ {\mathbf A}_\perp, \mathcal{H}_{A} \}  = {\mathbf 0}.
\end{aligned}
\right.
``` 


We denote by
``(f_0^0(x,p),{\mathbf f}^0(x,p), E_x^0(x),{\mathbf E}_\perp^0(x),{\mathbf A}_\perp^0(x))``
the initial value at time ``t=0``. The exact solution at time t is,

```math
\begin{aligned}
        &f_0(x,p,t)=f_0^0 \left( x,p+t{\mathbf A}_\perp^0(x) \cdot \frac{\partial {\mathbf A}_\perp^0(x)}{\partial x}  \right) ,\\
&   {\mathbf f}(x,p,t)={\mathbf f}^0 \left( x,p+t{\mathbf A}_\perp^0(x) \cdot \frac{\partial {\mathbf A}_\perp^0(x)}{\partial x}  \right), \\
        &{\mathbf E}_\perp(x,t)={\mathbf E}_\perp^0(x)-t\frac{\partial^2 {\mathbf A}_\perp^0(x)}{\partial x^2} +t{\mathbf A}_\perp^0(x)\int_{\mathbb{R}} f_0^0(x,p) \mathrm{d}p, \\
        &E_x(x,t)=E_x^0(x), \;\; {\mathbf A}_\perp(x,t)={\mathbf A}_\perp^0(x), \\
\end{aligned}
```

## Subsystem for ``\mathcal{H}_E``

The subsystem
``\frac{\partial \mathcal{Z}}{\partial t} = \{ \mathcal{Z}, \mathcal{H}_E \}``
associated to the sub-Hamiltonian
``\mathcal{H}_{E} = \frac{1}{2}\int |{\mathbf E}|^2  \mathrm{d}{\mathbf x}``
is 

```math
\left\{
\begin{aligned}
&\frac{\partial f_0}{\partial t} = \{f_0, \mathcal{H}_{E} \} = -E_x \frac{\partial f_0}{\partial p}, \\
&\frac{\partial {\mathbf f}}{\partial t} = \{{\mathbf f}, \mathcal{H}_{E} \}= -E_x \frac{\partial {\mathbf f}}{\partial p}, \\
%&\frac{\partial f_2}{\partial t} = \{f_2, \mathcal{H}_{E} \}= -E_x \frac{\partial f_2}{\partial p}, \\
%&\frac{\partial f_3}{\partial t} = \{f_3, \mathcal{H}_{E} \}=-E_x \frac{\partial f_3}{\partial p},\\
%& \frac{\partial E_x}{\partial t} = \{ E_x, \mathcal{H}_{E} \} = 0,\\
& \frac{\partial {\mathbf E}_\perp}{\partial t} =  \{ {\mathbf E}_\perp, \mathcal{H}_{E} \}  = {\mathbf 0},\\
& \frac{\partial E_x}{\partial t} = \frac{\partial {\mathbf A}_\perp}{\partial t} = 0. % \{ {\mathbf A}_\perp, \mathcal{H}_{E} \}  =  -{\mathbf E}_\perp.
\end{aligned}
\right.
```

With the initial
value``(f_0^0(x,p),{\mathbf f}^0(x,p),E_x^0(x),{\mathbf E}_\perp^0(x),{\mathbf A}_\perp^0(x))``
at time ``t=0``, the solution at time t is as follows, 


```math
\begin{aligned}
& f_0(x,p,t)=f_0^0 ( x,p-tE_x^0(x) ), \\
& {\mathbf f}(x,p,t)={\mathbf f}^0 ( x,p-tE_x^0(x) ),  \\
& E_x(x,t)=E_x^0(x), \\
& {\mathbf E}_\perp(x,t)={\mathbf E}_\perp^0(x), \\
& {\mathbf A}_\perp(x,t)={\mathbf A}_\perp^0(x) -t{\mathbf E}_\perp^0(x).
\end{aligned}
```

## Subsystem for ``\mathcal{H}_2``

The subsystem
``\frac{\partial \mathcal{Z}}{\partial t} = \{ \mathcal{Z}, \mathcal{H}_2 \}``
associated to the sub-Hamiltonian
``\mathcal{H}_{2} = \int_{\Omega}  \mathfrak{h} f_2 \frac{\partial A_z}{\partial x} \mathrm{d}x\mathrm{d}p``
is 

```math
\left\{
\begin{aligned}
&\frac{\partial f_0}{\partial t} = \{f_0, \mathcal{H}_{2} \} = \mathfrak{h}\frac{\partial^2 A_z}{\partial x^2}\frac{\partial f_2}{\partial p},  \\
&\frac{\partial f_1}{\partial t} = \{f_1, \mathcal{H}_{2} \}= \frac{\partial A_z }{\partial x}  f_3, \\
&\frac{\partial f_2}{\partial t} = \{f_2, \mathcal{H}_{2} \}= {{\frac{\mathfrak{h}}{3}}} \frac{\partial^2 A_z}{\partial x^2}\frac{\partial f_0}{\partial p}, \\
&\frac{\partial f_3}{\partial t} = \{f_3, \mathcal{H}_{2} \}= -\frac{\partial A_z }{\partial x} f_1,\\
& \frac{\partial { E}_z}{\partial t} =  \{ {E}_z, \mathcal{H}_{2} \}  =-\mathfrak{h}{ \int_{\mathbb{R}}  \frac{\partial f_2}{\partial x}\mathrm{d}{p}},\\
& \frac{\partial E_x}{\partial t} =\frac{\partial { E}_y}{\partial t} =\frac{\partial {\mathbf A}_\perp}{\partial t} =0. 
\end{aligned}
\right.
```

In this subsystem, we observe some coupling between the
distribution functions. To write down the exact solution, we reformulate
the equations on ``(f_0, {\mathbf f})`` as, using ``A_z(x, t)=A_z^0(x)``

```math
    \begin{aligned}
    & \partial_t    \begin{pmatrix}
            f_1  \\
            f_3
        \end{pmatrix}-\frac{\partial A_z^0 }{\partial x} J \begin{pmatrix}
f_1  \\
f_3
\end{pmatrix} =0,  \\
& \partial_t    \begin{pmatrix}
    f_0 \\
    f_2
\end{pmatrix}-\mathfrak{h}\frac{\partial^2 A_z^0}{\partial x^2} \begin{pmatrix}
    0 & 1 \\
    \frac{1}{3}& 0
\end{pmatrix} \partial_p    \begin{pmatrix}
f_0 \\
f_2
\end{pmatrix} =0, \\
    \end{aligned}
```

where ``J`` denotes the symplectic matrix

```math
J=\begin{pmatrix}
    0 & 1 \\
-1 & 0
\end{pmatrix}.
```

With the initial value
``(f_0^0(x,p),{\mathbf f}^0(x,p),E_x^0(x),{\mathbf E}_\perp^0(x),{\mathbf A}_\perp^0(x))``
at time ``t=0``, the exact solution for the first system is

```math
\begin{pmatrix}
            f_1  \\
            f_3
        \end{pmatrix}(x,p,t)=\exp{\left(\frac{\partial A_z^0(x) }{\partial x} J t\right)}\begin{pmatrix}
f_1^0(x,p)  \\
f_3^0(x,p)
\end{pmatrix},\  \text{with}\ \exp{(Js)}=\begin{pmatrix}
    \cos(s) & \sin(s) \\
-\sin(s) & \cos(s)
\end{pmatrix}.
```

Let us now focus on the second system

```math
\partial_t  \begin{pmatrix}
    f_0 \\
    f_2
\end{pmatrix}-\mathfrak{h}\frac{\partial^2 A_z^0}{\partial x^2} \begin{pmatrix}
    0 & 1 \\
    \frac{1}{3}& 0
\end{pmatrix} \partial_p    \begin{pmatrix}
f_0 \\
f_2
\end{pmatrix} =0. 
```

By the eigen-decomposition 


```math
\begin{pmatrix}
    \frac{1}{2} & \frac{\sqrt{3}}{2} \\
    \frac{1}{2}& -\frac{\sqrt{3}}{2}
\end{pmatrix}
\begin{pmatrix}
    0 & 1 \\
    \frac{1}{3}& 0
\end{pmatrix}
\begin{pmatrix}
    1 & 1 \\
    \frac{1}{\sqrt{3}}& \frac{-1}{\sqrt{3}}
\end{pmatrix}
=\begin{pmatrix}
\frac{1}{\sqrt{3}}& 0 \\
    0& -\frac{1}{\sqrt{3}}
\end{pmatrix},
```

then, one can diagonalize the transport equation to get

```math
\partial_t  \begin{pmatrix}
    \frac{1}{2}f_0+\frac{\sqrt{3}}{2}f_2 \\
    \frac{1}{2}f_0-\frac{\sqrt{3}}{2}f_2 
\end{pmatrix}- 
\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_z^0}{\partial x^2}
\begin{pmatrix}
    1 & 0 \\
    0& -1
\end{pmatrix} \partial_p    \begin{pmatrix}
    \frac{1}{2}f_0+\frac{\sqrt{3}}{2}f_2 \\
    \frac{1}{2}f_0-\frac{\sqrt{3}}{2}f_2 
\end{pmatrix} =0.

```

Thus, we can solve the transport equation

```math
\Big(\frac{1}{2}f_0\pm\frac{\sqrt{3}}{2}f_2\Big)(x,p,t)=\Big(\frac{1}{2}f_0^0 \pm\frac{\sqrt{3}}{2}f_2^0\Big)( x,p\pm t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_z^0}{\partial x^2}(x)),
```

and compute the exact solution at time ``t`` as follows, 

```math
\begin{aligned}
        &f_1(x,p,t)=\cos(t \frac{\partial A_z^0(x) }{\partial x} )f_1^0 ( x,p)+\sin(t \frac{\partial A_z^0(x) }{\partial x} )f_3^0 ( x,p), \\
        &f_3(x,p,t)=-\sin(t \frac{\partial A_z^0(x) }{\partial x} )f_1^0 ( x,p)+\cos(t \frac{\partial A_z^0(x) }{\partial x} )f_3^0 ( x,p) \\
        &f_0(x,p,t)=\Big(\frac{1}{2}f_0^0 +\frac{\sqrt{3}}{2}f_2^0\Big)( x,p+t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_z^0}{\partial x^2}(x) )+\Big(\frac{1}{2}f_0^0 -\frac{\sqrt{3}}{2}f_2^0\Big)( x,p-t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_z^0}{\partial x^2}(x) ),\\
        &f_2(x,p,t)=\frac{1}{\sqrt{3}}\Big(\frac{1}{2}f_0^0 +\frac{\sqrt{3}}{2}f_2^0\Big)( x,p+t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_z^0}{\partial x^2}(x) )-\frac{1}{\sqrt{3}}\Big(\frac{1}{2}f_0^0 -\frac{\sqrt{3}}{2}f_2^0\Big)( x,p-t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_z^0}{\partial x^2}(x) ),\\
        &{\mathbf A}_\perp(x,t)={\mathbf A}_\perp^0(x), E_x(x,t)=E_x^0(x), E_y(x,t)=E_y^0(x),\\
        &E_z(x,t)=E_z^0(x)-t\mathfrak{h}\int_{\mathbb{R}} \frac{\partial f_2^0}{\partial x}\mathrm{d}{p}.
    \end{aligned}
```

## Subsystem for ``\mathcal{H}_3``

The subsystem
``\frac{\partial \mathcal{Z}}{\partial t} = \{ \mathcal{Z}, \mathcal{H}_3 \}``
associated to the sub-Hamiltonian
``\mathcal{H}_{3} = -\int_{\Omega} \mathfrak{h} f_3 \frac{\partial A_y}{\partial x} \mathrm{d}x\mathrm{d}p``
is 

```math
\left\{
\begin{aligned}
&\frac{\partial f_0}{\partial t} = \{f_0, \mathcal{H}_{3} \} = -\mathfrak{h}\frac{\partial^2 A_y}{\partial x^2} \frac{\partial f_3}{\partial p},  \\
&\frac{\partial f_1}{\partial t} = \{f_1, \mathcal{H}_{3} \}=  \frac{\partial A_y }{\partial x} f_2, \\
&\frac{\partial f_2}{\partial t} = \{f_2, \mathcal{H}_{3} \}= - \frac{\partial A_y }{\partial x} f_1,\\
&\frac{\partial f_3}{\partial t} =- \{f_3, \mathcal{H}_{3} \}= -{{\frac{\mathfrak{h}}{3}}} \frac{\partial^2 A_y}{\partial x^2} \frac{\partial f_0}{\partial p}, \\
& \frac{\partial { E}_y}{\partial t} =  \{ {E}_y, \mathcal{H}_{3} \}  = {\mathfrak{h} \int_{\mathbb{R}}  \frac{\partial f_3}{\partial x}\mathrm{d}{p}},\\
& \frac{\partial E_x}{\partial t} = \frac{\partial { E}_z}{\partial t} = \frac{\partial {\mathbf A}_\perp}{\partial t} =0. 
\end{aligned}
\right.
```

This subsystem is very similar to the ``\mathcal{H}_2`` one,
hence, as previously, we reformulate the equations on the distribution
functions as 

```math
\begin{aligned}
        & \partial_t    \begin{pmatrix}
            f_1  \\
            f_2
        \end{pmatrix}
        -\frac{\partial A_y^0 }{\partial x} J 
        %\begin{pmatrix}
    %       0 & -1 \\
%           1 & 0
        %\end{pmatrix} 
        \begin{pmatrix}
        f_1  \\
        f_2
    \end{pmatrix} =0, \\
        & \partial_t    \begin{pmatrix}
            f_0 \\
            f_3
        \end{pmatrix}+\mathfrak{h}\frac{\partial^2 A_y^0}{\partial x^2} \begin{pmatrix}
            0 & 1 \\
            \frac{1}{3}& 0
        \end{pmatrix} \partial_p    \begin{pmatrix}
            f_0 \\
            f_3
        \end{pmatrix} =0, \\
    %& \frac{\partial E_x}{\partial t}= 0,\ 
 %\frac{\partial { E}_y}{\partial t} =   {\mathfrak{h} \int_{\mathbb{R}}  \frac{\partial f_3}{\partial x}\mathrm{d}{p}},\\
    %& \frac{\partial { E}_z}{\partial t} =  \{ {E}_z, H_{3} \}  = { 0},\ 
     %\frac{\partial {\mathbf A}_\perp}{\partial t} =  \{ {\mathbf A}_\perp, H_{3} \}  =  0,
    \end{aligned}
```

with initial value

``(f_0^0(x,p),{\mathbf f}^0(x,p),E_x^0(x),{\mathbf E}_\perp^0(x),{\mathbf A}_\perp^0(x))``
at time ``t=0``. We derive similar formula with ``\mathcal{H}_2`` for the
exact solution at time ``t`` 

```math
\begin{aligned}
        &f_1(x,p,t)=\cos(t \frac{\partial A_y^0(x) }{\partial x} )f_1^0 ( x,p)+\sin(t \frac{\partial A_y^0(x) }{\partial x} )f_2^0 ( x,p), \\
        &f_2(x,p,t)=-\sin(t \frac{\partial A_y^0(x) }{\partial x} )f_1^0 ( x,p)+\cos(t \frac{\partial A_y^0(x) }{\partial x} )f_2^0 ( x,p), \\
        &f_0(x,p,t)=\Big(\frac{1}{2}f_0^0 +\frac{\sqrt{3}}{2}f_3^0\Big)( x,p-t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_y^0}{\partial x^2}(x) )+\Big(\frac{1}{2}f_0^0 -\frac{\sqrt{3}}{2}f_3^0\Big)( x,p+t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_y^0}{\partial x^2}(x) ),\\
        &f_3(x,p,t)=\frac{1}{\sqrt{3}}\Big(\frac{1}{2}f_0^0 +\frac{\sqrt{3}}{2}f_3^0\Big)( x,p-t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_y^0}{\partial x^2}(x) )-\frac{1}{\sqrt{3}}\Big(\frac{1}{2}f_0^0 -\frac{\sqrt{3}}{2}f_3^0\Big)( x,p+t\frac{\mathfrak{h}}{\sqrt{3}}\frac{\partial^2 A_y^0}{\partial x^2}(x) ),
    \end{aligned} 
```

```math
\begin{aligned}
& E_y(x,t)=E_y^0(x)+t\mathfrak{h}\int_{\mathbb{R}} \frac{\partial f_3^0}{\partial x}\mathrm{d}{p}, \\
& {\mathbf A}_\perp(x,t)={\mathbf A}_\perp^0(x), \\
& E_x(x,t)=E_x^0(x), \\
& E_z(x,t)=E_z^0(x). 
\end{aligned} 
```


To compute the solution ``E_y(x,t)``, we use the fact
that
```math
\int_{\mathbb{R}} f_3(x,p,t) \mathrm{d}p =\int_{\mathbb{R}} f_3^0(x,p)\mathrm{d} p.
```

