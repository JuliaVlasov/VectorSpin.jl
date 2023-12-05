# Numerical scheme

We assume that the computational domain is $[0,L]\times[-P_L,P_R]$ for
$x$ and $v$. The system is periodic in $x$-direction with period $L$ and
has compact support on $[-P_L,P_R]$ in $v$-direction. The mesh is as
follows:
```math
x_j=(j-1)\Delta x,\ j=1,...,M,\ \Delta x=L/M, \ (M \text{ is odd)}
```

```math
p_{\ell-1} = (\ell-1)\Delta p-P_L,\  \ell=1,...,N, \ \Delta p=(P_L+P_L)/N.
```

We use the spectral Fourier expansion to
approximate $E_x$ as it is periodic in the $x$-direction,

```math
E_{x,j}=\sum_{k=-(M-1)/2}^{(M-1)/2} \hat{E}_{x,k}(t)e^{2\pi ijk/M}, \;\; \text{ for } j=1,...,M.
```

For the distribution functions $(f_0, {\mathbf f})$, we use a spectral
Fourier expansion for the $x$-direction and a finite-volume method for
the $p$-direction. For simplicity, we only present the representation
for $f_0$, the notations for ${\mathbf f}$ are the same. Here
$f_{0,j,\ell}(t)$ denotes the average of $f_0(x_j,p,t)$ over a cell
$C_\ell=[p_{\ell-1/2}, p_{l+1/2}]$ with the midpoint
$p_{\ell-1/2}=(\ell-1/2)\Delta p-P_L$, that is,

```math
f_{0,j,\ell}(t)=\frac{1}{\Delta p} \int_{C_\ell} f_0(x_j,p,t)\mathrm{d}{p},
```

and also by Fourier expansion in $x$-direction, then

```math
f_{0,j,\ell}(t)=\sum_{k=-(M-1)/2}^{(M-1)/2} \hat{f}_{0,k,\ell}(t)e^{2\pi ijk/M}, \;\; j=1,...,M.
```

To evaluate the value of $f_0$ off-grid in $p$-direction, we need to
reconstruct a continuous function by using the cell average quantity
$f_{0,j,\ell}$. 

