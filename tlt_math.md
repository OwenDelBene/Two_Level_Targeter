### Two-Level Targeter (TLT)

By dividing the corrections process into different parts, the size of the Jacobian and, therefore, the computational load in calculating its matrix inverse is decreased. Additionally, monotonic convergence behavior is more frequently observed in multi-stage corrections processes than in standard multiple shooting algorithms. In on-board applications where communication with ground stations may be limited or lost, the predictability in convergence gained from a staged targeting algorithm is valuable. One such corrections strategy with two levels is called a *Two-Level Targeter (TLT)*. As originally formulated, the TLT is decomposed into Level-I which corrects for position continuity and Level-II which targets velocity continuity. To model maneuvers at patch points by chemical propulsion systems that occur over time intervals significantly smaller than the arc propagation times, the thrusting acceleration over a small duration may be represented as an instantaneous velocity change, labelled a Delta-V or impulsive maneuver. A two-level corrections algorithm that uses this assumption is named an impulsive TLT.

#### 3.4.1 Impulsive Level-I

In Level-I of an impulsive TLT algorithm, position continuity within the convergence tolerance is achieved for each arc in series. An initially discontinuous set of arcs is propagated starting with the first patch point. The spacecraft trajectory is propagated from the initial patch point, $P_o$, at time $t_o$, to the next patch point, $P_p$, at time $t_p$. The design vector,

$$
X = v^+_o
$$

> (3.53)

consists of the outgoing velocity vector at $p_o$. This quantity is varied within a single shooting algorithm to achieve position continuity at $p_p$, mathematically described by the constraint vector,

$$
F = ρ^−_p - ρ^+_p = 0
$$

> (3.54)

<p align="center">
<img src="https://i.imgur.com/KL1Xp48.png">
</p>

In practical terms, Level-I represents an impulsive maneuver to target a new upstream position. The remaining component necessary to formulate an iterative Newton algorithm is the Jacobian,

$$
DF = \frac{∂ρ^-_p}{∂v^+_o}
$$

> (3.55)

that models the outgoing position at $p_p$ as a constant. The STM is used to solve an iterative single shooting algorithm. The initial position, $p^+_o$, and patch point times are constant, resulting in $\delta p^+_o = 0$ and $\delta t_o = dt_p = 0$. By employing the STM and simplifying, the variations at $p_o$ and $p_p$ are related by

$$
\begin{align}
\begin{bmatrix} 
d ρ^-_p \\ 
d v^-_p 
\end{bmatrix} = Φ(t_p, t_o)
\begin{bmatrix} 
d ρ^+_o \\ 
d v^+_o 
\end{bmatrix}
= \begin{bmatrix} 
A_op & B_op \\ 
C_op & D_op 
\end{bmatrix}
\begin{bmatrix} 
d ρ^+_o \\ 
d v^+_o 
\end{bmatrix}
\end{align}
$$

> (3.56); double-subscripts are currently broken and the workaround seems to fall short here: https://github.com/github/markup/issues/1575; assume "op" is the subscript here, as in 3.56 in York's .pdf

This relationship leads to determination of the Jacobian,

$$
\frac{∂p^+_p}{∂v^+_o} = B_op
$$

> > (3.57)

which is substituted into the previous equation to yield

$$
dv^+_o = −B^{−1}_op dρ^+_p
$$

> (3.58)

Each iteration in the single shooting algorithm is denoted a *local iteration*. Once the $F$ constraint is within the specified convergence tolerance, the algorithm is repeated for the next patch point in the series. The previous $p_p$ becomes the new $p_o$, and the next patch point is renamed $p_p$. Once all the patch points have reached convergence through these sequential single shooting processes, Level-I is finished for a single *global iteration*.

#### 3.4.2 Impulsive Level-II

Following completion of Level-I for a global iteration, the algorithm executes one multiple shooting correction in Level-II. The goal of Level-II is to reduce velocity discontinuities by shifting patch point positions and times. For a trajectory comprised of $k$ patch points, velocity continuity at the interior patch points is enforced by creating a constraint vector, i.e.,

$$
F = \begin{bmatrix} 
v^2 − v^+_2 \\ 
\vdots \\ 
v^−_k−1 − v^+_k−1 
\end{bmatrix}
$$

> (3.59); multi-subscript issue strikes again here, I'm afraid

As apparent in the provided content, all patch points may be considered continuous in position, expressed as

$$
ρ_i = ρ^+_i = ρ^−_i
$$

> (3.60)

and, therefore, no superscript is necessary for the position vectors. Thus, the design variable vector is formulated as

$$
X = 
\begin{bmatrix} 
p_1 \\ 
t_1 \\ 
\vdots \\ 
p_k \\ 
t_k 
\end{bmatrix}
$$

> (3.61)

<p align="center">
<img src="https://i.imgur.com/KL1Xp48.png">
</p>

and consists of the positions and times for all k patch points.
To construct the Level-II corrections algorithm, the Jacobian, $DF$, is constructed
in pieces. An interior patch point from $F$ is deemed the present patch point, $P_p$,
with corresponding preceeding and subsequent patch points, denoted Po and $P_f$,
respectively. States at Pp are assumed to depend only on states and times at $P_o$, $P_p$,
and $P_f$. The constraint and design vectors are rewritten as

$$
F =
\begin{bmatrix}
\vdots \\
v^-_p - v^+_p \\
\vdots
\end{bmatrix}
$$

> (3.62)

$$
X = 
\begin{bmatrix}
\vdots \\
P_o \\
t_o \\
P_p \\
t_p \\
P_f \\
t_f \\
\vdots
\end{bmatrix}
$$

> (3.63)

respectively, to reflect this notation. The Jacobian,

$$
DF = \begin{bmatrix}
\vdots \\
\frac{∂(v_p^-v^+_p)}{∂X} \\
\vdots
\end{bmatrix}
$$

> (3.64)

is built row by row. Further simplifying assumptions lead to derivation of the partial derivatives. The incoming velocity at $P_p, v^-_p$, is only a function of $P_o$ and $P_p$, and the outgoing velocity at $P_p, v^+_p$, is only a function of $P_p$ and $P_f$. From these assumptions, the Jacobian row of interest is reformulated as

$$
DF = 
\begin{bmatrix}
\vdots & & & & & & & \vdots \\
0 & \frac{∂v^-_p}{∂ρ_o} & \frac{∂v^-_p}{∂t_o} & \frac{∂v^-_p}{∂ρ_p} - \frac{∂v^+_p}{∂ρ_p} & \frac{∂v^-_p}{∂t_p} - \frac{∂v^+_p}{∂t_p} & \frac{∂v^+_p}{∂ρ_f} & \frac{∂v^+_p}{∂t_f} & 0 \\
\vdots & & & & & & & \vdots
\end{bmatrix}
$$

> (3.65)

in terms of the individual vector derivatives.

First, the partial derivatives of $v^p$ with respect to the design variables are determined. The STM maps contemporaneous variations at $P^-_p$ backwards in time to $P^+_o$. Since patch point times are variable, substitution for total variations based on Equation (3.40) yields

$$
\begin{bmatrix}
dP^+_o - v^+_o dt_o \\
dv^+_o - a^+_o dt_o \\
\end{bmatrix}
$$

$$
\begin{bmatrix}
A_{po} & B_{po} \\
C_{po} & D_{po}
\end{bmatrix}
\begin{bmatrix}
dP^-_p - v^-_p dt_p \\
dv^-_p - a^-_p dt_p
\end{bmatrix}
$$

> Oh, equation (3.66), why are you the way that you are? The top matrix is equal to the bottom matrix. This is the ONLY way I could get it to render.

where the backward-time STM is generated through inversion, i.e.,

$$
\begin{bmatrix}
A_{po} & B_{po} \\
C_{po} & D_{po}
\end{bmatrix}
= \begin{bmatrix}
A_{op} & B_{op} \\
C_{op} & D_{op}
\end{bmatrix}^{-1}
$$

> (3.67); wh-- NOW the double subscripts are able to work? They break literally everything else; I don't even know why I bothered trying on this one, and yet -- agh.

based on the property of STMs proven in Equation (3.29). From the position continuity assumption in Equation (3.60), the superscripts on the position vectors may be disregarded $(dρ^+_i = dρ^-_i)$. Selecting the components of Equation (3.66) relating the first row on the left side, a relationship is determined between variations in only the constraint and design variables,

$$
dP_o - v^-_o dt_o = A_po(dP_p - v^-_p dt_p) + B_po(dv^-_p - a^-_p dt_p)
$$

> (3.68); yet another victim of the double-subscript issue. I don't even know anymore.

Rearranging this equation yields partial derivatives,

$$
\frac{∂v^-_p}{∂P_o} = B^{-1}_po
$$
$$
\frac{∂v^-_p}{∂t_o} = - B^{-1}_po v^+_o
$$
$$
\frac{∂v^-_p}{∂P_p} = - B^{-1}_po A_po
$$
$$
\frac{∂v^-_p}{∂t_p} = -B^{-1}_po A_po v^-_p + a^-_p
$$

> (3.69)

that aid in construction of the Jacobian in Equation (3.65).

Since the constraint vector is formulated in terms of both incoming and outgoing velocity at the patch point, the mapping of changes in the design vector to changes in \( v^+_p \) is required. For the partial derivatives of \( v^+_p \) with respect to the design variables, another variational relationship,

$$
\begin{bmatrix}
dP^- - v^-f dt_f \\
dv^- - a^-f dt_f
\end{bmatrix}
= \begin{bmatrix}
A_{pf} & B_{pf} \\
C_{pf} & D_{pf}
\end{bmatrix}
\begin{bmatrix}
dP^+_p - v^+_p dt_p \\
dv^+_p - a^+_p dt_p
\end{bmatrix}
$$

> (3.70)

employs the STM to map variations at \( P^+_p \) forward in time to \( P^-_f \). Selecting the portion related to the top row of the left side yields

$$
dρ_f - v^f dt_f = A_pf(dρ^+_p - v^+_p dt_p) + B_pf(dv^+_p - a^+_p dt_p)
$$

> (3.71)

where position is again assumed continuous at the patch points. From this equation, expressions of the partial derivatives of $v^+_p$ with respect to the design variables is determined, i.e.,

$$
\frac{∂v^+_p}{∂P_f} = B^-1_pf
$$

$$
\frac{∂v^+_p}{∂t_f} = - B^-1_pf v^-_f
$$

$$
\frac{∂v^+_p}{∂P_p} = - B^-1_pf A_pf
$$

$$
\frac{∂v^+_p}{∂t_f} = -B^{1_pf A_pf v^+_p + a^+_p
$$

> (3.72)

These partial derivatives allow construction of the DF matrix row for the designated patch point $P_p$. To complete the DF matrix, the process is repeated with $P_f$ becoming the new $P_p$ and shifting all other subscripts accordingly. Once the Level-II Jacobian is fully allocated, the minimum-norm solution is computed to update the design vector. The updates to position and time variables lead to discontinuities entering the next global iteration of Level-I, illustrated in Figure 3.13. The entire TLT algorithm is

<p align="center">
<img src="https://i.imgur.com/NZFgrKC.png">
</p>

complete when constraints fall within the convergence criteria for both Level-I and Level-II. Fundamental differences exist between the implementation and goals of Level-I and Level-II. In contrast to Level-I, which may consist of multiple local iterations for each global iteration, this Level-II step executes only once for each global iteration. Additionally, while Level-I is designed to reduce the norm of the constraint vector after each local iteration, the same may not be true of Level-II. Since position continuity is an assumption of Level-II, the function of Level-II is to reposition the patch points such that Level-II constraints are achieved when the position continuity is restored. The more visibly continuous slopes at the patch points are obvious in Figure 3.14, following the second global Level-I iteration. Thus, the velocity discontinuities

<p align="center">
<img src="https://i.imgur.com/O1RMboZ.png">
</p>

may not reduce substantially immediately following a Level-II iteration; however, the benefit of the Level-II correction is expected to occur after reconvergence of Level-I.
