# E-flite UMX Eratix 3D FF 450 mm: first flight-dynamics model

A model of the lab's airplane (E-flite part EFLU02050) built from its published specs, its product photos and textbook
aerodynamics, before any flight data. It is good enough to start training policies (`FW_AIRPLANE=2`, target
`train_sac_c128_b256_tracking_g995_eratix_i10a20`) and to plan the identification flights of section 9. Code:
`cpp/include/fixed_wing/dynamics.h` (`eratix_coefficients`, `eratix()`, `eratix_propeller()`, `eratix_aerodynamics()`),
task settings in `cpp/include/fixed_wing/fixed_wing.h`, checks in `cpp/src/eratix_trim.cpp`.

Labels: **[source]** a number stated by the cited document; **[photo]** measured on a product photo (section 1.2);
**[estimate]** our own calculation, inputs given; **[assumption]** a value chosen because the real one is unknown.
Every ± is the range we would randomize over, not a statistical error.

## 1. Sources and method

### 1.1 Documents

| Ref | Document |
|---|---|
| M | E-flite UMX Eratix 3D FF 450mm manual, EN, 817803.1 (updated 12/25): https://assets.horizonhobby.com/on/demandware.static/-/Sites-horizon-master/default/dw271f7a6d/Manuals/EFLU02050_MANUAL_EN.pdf |
| H | Horizon Hobby product page, Specs tab: https://www.horizonhobby.com/product/umx-eratix-3d-ff-flat-foamy-450mm-bnf-basic-with-as3x-and-safe-select/EFLU02050.html (same specs on Tower Hobbies) |
| P | Horizon product photos of EFLU02050, 2000 x 2000 px: `EFLU02050_A2` (top view), `A1` (bottom view), `A5` (side view), `A47` (in hand), under https://www.horizonhobby.com/on/demandware.static/-/Sites-horizon-master/default/dw618c8fab/Images/EFL/ |
| M860 | Manual of the parent design, Eratix 3D FF 860 mm (EFL01950), DE: https://www.astramodel.cz/manualy/eflite/EFL01950_MANUAL_DE.pdf |
| SC | Spektrum servo chart (Horizon, 2025) |
| FP | Flat-plate aerodynamics: Helmbold, Torres & Mueller 2004, Ortiz et al. 2015, ESDU 70015, Su et al. 2023 |
| D | USAF DATCOM methods as given in Nelson, *Flight Stability and Automatic Control*, and Roskam, *Airplane Design* VI (downwash, tail volume, strip theory) [textbook] |

Not published anywhere we could find: wing area, chords, foam thickness, tail and fuselage dimensions, throws in mm or
degrees for the 450 mm model, thrust data, flight speeds, motor and receiver weights, the A202 servo speed.

### 1.2 Photo scaling

The silhouettes of photos A2, A1 and A5 were measured pixel by pixel against their white background, in the browser
(columns and rows of non-background pixels every 30 to 40 px). Scales: top view 1531 px for the 450 mm span, 0.294 mm/px;
side view 1610 px for the 487 mm length, 0.3025 mm/px. Cross-check: the top-view length comes out at 483 mm against the
manual's 487 mm (-0.8 %), the bottom view (A1) gives 477 mm. We take ±3 % on lengths and ±6 % on areas for scale and
perspective, more where an edge was hard to see (noted below).

## 2. Geometry

| Item | Value | ± | Origin |
|---|---|---|---|
| Span b | 450 mm | — | [source] M |
| Length | 487 mm | — | [source] M |
| Mass with 2S 300 mAh battery | 83.5 g | ±1 g | [source] M (H: 65.2 g without battery, M: 66.5 g) |
| Airframe | laser-cut flat EPS foam, composite reinforcement | — | [source] H |
| Airfoil | flat plate, symmetric, no camber, no incidence, no dihedral | — | [source] H ("flat foam"), [photo] |
| Wing planform | straight unswept leading edge; trailing edge swept forward; root chord 151 mm, tip chord about 92 mm with clipped tip corners; taper 0.6 | ±3 % | [photo] A2 |
| Wing area S | 0.0531 m² (5.31 dm²) | ±6 % | [photo] A2, chord integrated over the span (A1 gives 5.5 dm²) |
| Aspect ratio | 3.81 | ±6 % | b²/S |
| Mean aerodynamic chord c̄ | 121 mm, 95 mm from the centre line, its leading edge on the wing's | ±5 % | [estimate] ∫c²dy / ∫c dy |
| Wing loading | 15.7 g/dm² (5.1 oz/ft²) | ±6 % | UMX Sbach 342 for comparison: 5.9 oz/ft², 4.2 dm², 75 g (Amain listing) |
| Ailerons | hinge line straight, about 75 mm behind the leading edge; chord 71 mm at the root (49 %), 34 mm at the tip (34 %); from 10 % of the half span to the tip | ±15 % | [photo] A1 (the hinge seam is faint) |
| Horizontal tail | span 221 mm, area 0.0155 m² (S_h/S = 0.29), mean chord 76 mm, AR 3.2; elevator about 60 % of the chord, full span with a central cut-out | area ±10 %, elevator chord ±10 points | [photo] A2, A1 |
| Tail arm | CG to tail quarter chord l_t = 216 mm; wing to tail quarter chords 262 mm | ±5 % | [photo] A2 |
| Horizontal tail volume V_H | 0.52 | ±15 % | S_h l_t / (S c̄) |
| Vertical tail | height 148 mm (above and below the fuselage), area 0.0085 m², rudder about 80 % of it; effective AR 2.6 | area ±15 % | [photo] A5, A2 |
| Fin arm and height | CG to fin quarter chord l_v = 277 mm; centroid 25 mm above the CG | l_v ±5 %, height ±20 mm | [photo] A5 |
| Vertical tail volume | 0.099 | ±20 % | S_v l_v / (S b) |
| Fuselage | profile (vertical sheet), 60-75 mm deep, side area about 0.027 m², centre of side force about 60 mm ahead of the CG | ±30 % | [photo] A5 |
| Side force generators, wing braces | small vertical plates near the wing roots, about 35 cm² together | ±50 % | [photo] A1, A47; parts EFL-3139, EFL-3142 (M) |
| CG | 76 ± 4 mm behind the wing leading edge = 63 % of c̄ | ±4 mm | [source] M ("The CG location is 76mm ± 4mm behind the leading edge of the wing") |
| Propeller | 148 mm x 35 mm (5.8 x 1.4 in), part EFL-3144 | — | [source] H, M |
| Motor | 1404-3200 Kv brushless outrunner, 12 poles (SPMXAM4600) | — | [source] M |
| ESC | 15 A, on the receiver board (SPM-1090) | — | [source] H |
| Battery | 2S 300 mAh 30C LiPo (SPMX3002S30): 9 A continuous | — | [source] M |
| Surface throws | the 450 mm manual gives transmitter percentages only (aileron and elevator 100 %/70 %, rudder 100 %/70 %). The 860 mm manual gives high rates of 88 mm (aileron, at the root), 58 mm (elevator, widest point), 85 mm (rudder, at the bottom). Over the surface chords of the 450 mm airplane scaled by 863.5/450: about 40°, 39°, 54°. Model: ±40°, ±40°, ±50° | ±10° | [estimate] from M860 and the photos |
| Servos | aileron: 2 x Spektrum A202 2.9 g long-throw linear (speed not published); elevator and rudder: linear servos on the SPM-1090 board | — | [source] M, H |

The 76 mm CG is far aft for a 121 mm mean chord, but it sits on the aileron hinge line at the root (75 mm), a usual
balance point for 3D foam airplanes, and the mass breakdown of section 3 balances there with the battery near the CG.
The 860 mm parent puts it at 79-84 mm on a chord about twice as long, so the 450 mm airplane is not a geometric scale of
it. This CG drives the most uncertain result of the model, the negative static margin (section 4.3).

## 3. Mass and inertia [estimate]

Component breakdown (x forward of the CG, from the photos; masses estimated where not published):

| Component | Mass [g] | x [mm] | Note |
|---|---|---|---|
| Battery | 17.0 | -13 | 83.5 - 66.5 g [source]; position from the balance |
| Motor | 9.0 | +146 | typical 1404 motor |
| Propeller and saver | 2.0 | +162 | |
| Receiver/ESC board SPM-1090 | 6.0 | +40 | weight not published |
| Aileron servos | 5.8 | +27 | 2 x 2.9 g [source], ±75 mm spanwise |
| Landing gear, pushrods, wires | 5.0 | +20 | 40 mm below |
| Wing (foam, braces) | 17.0 | +15 | centroid at c̄/2 |
| Horizontal tail | 4.5 | -235 | |
| Vertical tail | 2.5 | -288 | 20 mm above |
| Fuselage sheet | 9.7 | -40 | spread over 420 mm |
| Carbon, glue, decals | 5.0 | 0 | |

With the battery 13 mm behind the CG the moments balance at the manual's CG. Inertia about the CG (point masses plus
plates and rods for the wing, tails and fuselage, taper of the wing included):

| | Value [kg m²] | ± |
|---|---|---|
| Ixx | 3.9e-4 | 30 % |
| Iyy | 9.3e-4 | 30 % |
| Izz | 1.27e-3 | 30 % |
| Ixz | 0 (neglected, about 1e-5) | 3e-5 |

The wing makes 65 % of Ixx; the motor, tails and fuselage make most of Iyy and Izz. A pendulum test (section 9) would
cut these ranges to about 5 %.

## 4. Aerodynamic coefficients

Reference: S = 0.0531 m², b = 0.45 m, c̄ = 0.121 m, about the manual's CG, rates non-dimensional with b/2V or c̄/2V,
angles in rad. Reynolds number 4e4 to 8e4 (5-10 m/s on c̄). Signs as Beard & McLain (the Aerosonde of this simulator):
aileron > 0 rolls right, elevator > 0 (trailing edge down) pitches down, rudder > 0 (trailing edge left) yaws left.
In `dynamics.h` the 28 linear coefficients are `eratix_coefficients::linear`, in the order of the `Coefficient` enum.

Shared inputs [estimate]:
- Wing lift slope, Helmbold with a thin-plate 2π: a_w = 2π AR / (2 + sqrt(AR² + 4)) = 3.80 /rad. Torres & Mueller's
  low-AR flat plates at Re 7e4-2e5 agree with Helmbold within about 8 % (FP).
- Tail slopes: horizontal a_t = 3.45 /rad (AR 3.15), vertical a_v = 3.09 /rad (effective AR 2.6).
- Downwash at the tail (DATCOM, D): dε/dα = 4.44 (K_A K_λ K_H)^1.19 with K_A = 1/AR - 1/(1 + AR^1.7) = 0.169,
  K_λ = (10 - 3λ)/7 = 1.18, K_H = 0.95 (tail in the wing plane, 2 l/b = 1.17): dε/dα = 0.61 ± 0.15.
- Tail dynamic-pressure ratio η = 1.0 ± 0.15: 0.9 gliding, about 1.15 at cruise power, when roughly half of the tail
  span sits in the slipstream of the 148 mm propeller (T/(q A_disk) = 0.37 at 8 m/s).

### 4.1 Lift and drag

| Coefficient | Value | ± | Derivation |
|---|---|---|---|
| C_L0 | 0 | ±0.05 | symmetric flat plate, no incidence |
| C_Lα | 4.2 /rad | ±0.6 | a_w + η a_t (S_h/S)(1 - dε/dα) = 3.80 + 0.40 |
| C_Lq | 2.9 | ±2.5 | tail 2 η a_t V_H = +3.6, wing ahead of the CG -2 a_w (h - 0.25) = -2.9, downwash lag 2 η a_t V_H dε/dα = +2.2 lumped in (no α̇ input) |
| C_Lδe | 0.63 /rad | ±0.2 | η a_t (S_h/S) τ_e 0.95 (span); τ_e = 0.88 from thin-airfoil theory for a 60 % flap, times 0.75 for the large deflections and the hinge gap: 0.66 |
| C_D0 | 0.065 | ±0.02 | sum of C_D S: wing 0.025 x S (laminar friction 0.010 both sides at Re 6.5e4 plus bluntness of 4 % thick foam; Pelletier & Mueller flat plates about 0.02), tails 0.025, fuselage sheet friction and edge 7e-4 m², landing gear and wheels 5e-4 m², motor 2e-4 m², servos, horns, braces 3e-4 m²: 3.6e-3 m² |
| Oswald e | 0.8 | ±0.1 | low-AR flat plate with a profile fuselage |
| Deflection drag | 0.4 δa², 0.2 δe², 0.15 δr² | ±50 % | each surface as a flat flap of its area normal force C_N90 sin² δ; not in the AdvancedLiftDrag structure, added because 3D throws of 40° cost a lot of drag |

### 4.2 Stall and post-stall

The `AdvancedLiftDrag` blend σ(α) of the advanced_plane, with:
- α_stall = 12° ± 3°: attached C_L = 0.88 there; flat plates of AR 2 leave the attached-flow correlation at 13-15°
  (Torres & Mueller, via FP), higher AR earlier.
- M = 25 instead of the default 15, so that σ(0) = 1 % (with 15 the post-stall model would weigh 8 % at α = 0 for this
  stall angle). The blend spans about ±2.5°.
- Past the stall the force is normal to the plate (C_D/C_L = tan α, Ortiz et al. 2015):
  C_N = C_N90 sqrt(|sin α|) sign(α), C_L = C_N cos α, C_D = C_D0 + C_N sin α. C_N90 = 1.19 ± 0.1 (ESDU 70015:
  1.11 + 0.02 (AR + 1/AR)). The square root is our fit to the plateau that low-AR plates show between the stall and 45°
  (C_N 0.7 at 20°, 0.84 at 30°, 1.0 at 45°; Holmes et al. for AR 1, Ortiz et al.), smoothed below |sin α| = 0.05.
- Resulting lift: 0.70 at 12° (blend midpoint), about 0.6 just past the stall, 0.73 at 30°, 0.71 at 45°.
- Pitching moment past the stall: C_m = -0.42 ± 0.25 x the same shape. At 45° the wing's normal force (1.0, centre of
  pressure at 41 % of the chord after Su et al. 2023) acts 0.22 c̄ ahead of the CG, +0.22; the tail (local angle 38°,
  C_N 0.9, 228 mm behind) gives -0.50: -0.28 in all. The tail keeps the nose down after the stall.
- Control derivatives are kept linear outside the blend, as AdvancedLiftDrag does; they overpredict the surfaces at
  full throw (the elevator at 40° would ask the tail for 30° of angle of attack, past its own stall).

### 4.3 Pitching moment

| Coefficient | Value | ± | Derivation |
|---|---|---|---|
| C_m0 | 0 | ±0.03 | symmetric airframe; thrust line assumed through the CG |
| C_mα | +0.65 /rad | ±0.6 | neutral point h_n = 0.25 + η (a_t/a_w)(S_h/S)(l_t'/c̄)(1 - dε/dα) = 0.25 + 1.0 x 0.908 x 0.29 x 2.17 x 0.39 = 0.47 (±0.08 from dε/dα and η), l_t' from the wing's quarter chord; static margin h_n - h_cg = -0.16 ± 0.14 with h_cg = 0.63; C_mα = -C_Lα (h_n - h_cg) = 0.67, or 0.72 summing the wing and tail moments about the CG directly: 0.65 kept |
| C_mq | -11 | ±3.5 | tail -2 η a_t V_H (l_t/c̄) = -6.4, wing about its offset ahead of the CG -2 a_w (0.38)² = -1.1, wing itself -0.3, downwash lag C_mα̇ = -3.9 lumped in |
| C_mδe | -1.13 /rad | ±0.35 | -C_Lδe l_t/c̄ |

The airplane is statically unstable at the manual CG with these estimates. It still flies: the short-period roots are
real and well damped (-28 and -6 /s at 8 m/s, the large C_mq and C_Lα/V), and the instability shows as a slow
speed-pitch divergence of +0.6 /s (doubling in 1.1 s). If the true neutral point is at the aft end of its range
(power-on, less downwash) the margin is near 0. This is the first number to identify (section 9).

### 4.4 Lateral and directional

| Coefficient | Value | ± | Derivation |
|---|---|---|---|
| C_Yβ | -0.85 /rad | ±0.3 | fin -a_v (S_v/S)(1 + dσ/dβ) = -0.50 (sidewash 1.0 ± 0.2), profile fuselage as a low-AR plate with the wing as end plate -0.25, side force generators -0.10 |
| C_Yp | -0.05 | ±0.05 | fin 2 C_Yβ,v z_v/b |
| C_Yr | +0.6 | ±0.2 | fin -2 C_Yβ,v l_v/b |
| C_Yδr | +0.36 /rad | ±0.12 | a_v (S_v/S) τ_r, τ_r = 0.96 for an 80 % flap times 0.75 |
| C_lβ | -0.03 /rad | ±0.03 | fin C_Yβ,v z_v/b = -0.028, fuselage above the CG -0.006, forward-swept quarter chord (-4°) +0.007; no dihedral, mid wing |
| C_lp | -0.40 | ±0.12 | lifting line for the antisymmetric loading of an elliptic wing, -(a0/8)/(1 + 2a0/(π AR)) = -0.38 with a0 = 2π (strip theory with a_w gives -0.55, an upper bound for this AR), horizontal tail -0.02, fin -0.01 |
| C_lr | +0.12 | ±0.08 | wing C_L/4 at cruise C_L 0.4 = +0.10 (constant here), fin -2 C_Yβ,v z_v l_v / b² = +0.03 |
| C_lδa | +0.35 /rad | ±0.10 | strip theory 2 a_w τ_a ∫c y dy /(S b) = 0.49 (τ_a = 0.76 for a 42 % flap times 0.75, from 10 % of the half span to the tip), times 0.70 for the antisymmetric induced loading (ratio of the lifting-line to the strip roll damping) |
| C_lδr | +0.02 /rad | ±0.02 | C_Yδr z_v/b |
| C_nβ | +0.27 /rad | ±0.10 | fin -C_Yβ,v l_v/b = +0.31, fuselage ahead of the CG -0.033, side force generators -0.007; the propeller's normal force is not modelled |
| C_np | -0.02 | ±0.04 | wing -C_L/8 = -0.05, fin +0.03 |
| C_nr | -0.42 | ±0.12 | fin -2 a_v (S_v/S)(l_v/b)² = -0.38, fuselage behind the CG -0.02, wing -0.02 |
| C_nδa | -0.03 /rad | ±0.03 | adverse yaw, -0.2 C_L C_lδa at cruise |
| C_nδr | -0.22 /rad | ±0.07 | -C_Yδr l_v/b |
| C_Y0, C_l0, C_n0, C_Yδa | 0 | — | symmetric airframe |

Roll checks at 8 m/s: roll-mode time constant I_xx/(-C_lp q S b²/2V) = 37 ms; steady roll rate at full aileron
p = -C_lδa δa/C_lp x 2V/b, about 1200°/s: fast even for a 3D airplane, so the linear aileron likely overpredicts it at
full throw.

## 5. Propulsion and actuators

### 5.1 Motor and propeller [estimate except D, pitch and Kv]

The rotor speed n [rev/s] is the one where the motor torque at the commanded duty d balances the propeller torque at
the current airspeed V (along the body x axis):

- motor: Q_m = k_t ((d U - k_t ω)/R - I0), k_t = 60/(2π Kv) = 2.98e-3 N m/A, ω = 2π n;
- propeller: T = ρ n² D⁴ C_T, C_T = C_T0 (1 - J/J0); Q = ρ n² D⁵ C_P/(2π), C_P = C_P0 (1 - kP J); J = V/(n D).

Q_m = Q is a quadratic in n, solved in closed form (`eratix_propeller`). Values:

| Parameter | Value | ± | Origin |
|---|---|---|---|
| D, pitch | 0.148 m, 35 mm (P/D 0.24) | — | [source] H |
| Kv | 3200 rpm/V | — | [source] M |
| U | 7.6 V | ±0.6 | 2S LiPo between full (8.4 V) and loaded near empty (7.0 V) |
| R | 0.25 Ω | ±0.1 | winding of a 1404 motor about 0.15, two 300 mAh cells 0.1, ESC |
| I0 | 0.5 A | ±0.3 | 1404 class |
| C_T0, C_P0 | 0.060, 0.020 | ±25 %, ±30 % | low-pitch slow-flyer propeller; figure of merit C_T0^1.5/(1.2533 C_P0) = 0.59 |
| J0 | 0.35 | ±0.07 | zero thrust: pitch angle at 0.75 R 5.7° plus the zero-lift angle of a thin cambered blade about 3°, effective pitch 52 mm |
| kP | 2.0 | ±0.5 | C_P falls to 30 % of C_P0 at J0 |
| Rotor inertia | 3e-6 kg m² | ±50 % | 2 g propeller and motor bell |
| Rotation | clockwise seen from behind | — | [assumption], standard tractor propeller: the torque rolls the airplane left |

Results (`eratix_trim`): static thrust at full throttle 2.99 N (305 g) at 17,600 rpm, 8.4 A (the battery's 9 A
rating), 64 W: **T/W = 3.65** (±30 %, so 2.6 to 4.7). A 3D airplane must hover on its propeller: hover needs about
47 % throttle. The manual's hand launch "at 60% power" gives 1.3 N, 1.6 times the weight. At full throttle the thrust
falls to 1.8 N at 8 m/s and 0.8 N at 14 m/s, zero near 18 m/s: level flight tops out near 15.5 m/s.

Also modelled: the reaction torque of the propeller (0.023 N m static, rolls left) and the gyroscopic moment of the
rotor, -ω x H with H = I_rotor 2π n (about 3.4e-3 N m s at cruise), which yaws the airplane right when it pitches up.
Not modelled: the propeller's normal force (P-factor, fin effect), the slipstream over the wing and tail (needed for
hovering and torque rolls), side and down thrust (not published; 0 here).

The thrust randomization factor scales the whole propeller model (thrust and torque): `thrust_max` is the static
thrust at full throttle.

### 5.2 Actuators

| Actuator | Model | Value | ± | Origin |
|---|---|---|---|---|
| Servos (all three surfaces) | first order | τ = 45 ms | ±20 ms | Spektrum 2.3-2.9 g linear servos: 0.13-0.14 s per 60° at 4.2 V (SC); 95 % of a step in that time = 3τ. The A202's own speed is not published; a slew-rate limit is not modelled |
| Throttle (rotor speed) | first order on the duty | 50 ms up, 80 ms down | ±20 ms, ±40 ms | rotor inertia over the torque slopes: 3e-6 / (k_t²/R + 2Q/ω) = 3e-6 / 6e-5 = 50 ms; down with the propeller alone braking |
| Command latency | one control period | 20 ms | — | task default; the real latency of the replacement board is unknown |

## 6. Checks in the simulator

`eratix_trim` (target `eratix_trim`, run locally):

| Airspeed [m/s] | Pitch = α [°] | Elevator [°] | Throttle | Thrust [N] | rpm | C_L | L/D |
|---|---|---|---|---|---|---|---|
| 6.0 | 10.6 | 0.8 | 0.41 | 0.16 | 8,800 | 0.71 | 5.1 |
| 7.0 | 6.7 | 2.9 | 0.45 | 0.15 | 9,700 | 0.52 | 5.5 |
| 8.0 | 5.0 | 2.5 | 0.50 | 0.17 | 10,800 | 0.40 | 4.9 |
| 10.0 | 3.2 | 1.7 | 0.62 | 0.23 | 13,400 | 0.26 | 3.6 |
| 12.0 | 2.2 | 1.2 | 0.75 | 0.32 | 15,900 | 0.18 | 2.6 |
| 15.5 | 1.3 | 0.7 | 1.00 | 0.51 | 20,500 | 0.11 | 1.6 |

- Level trim in attached flow from 6.0 to 15.5 m/s. Best L/D 5.6 at 6.5 m/s; cruise near 8 m/s at half throttle.
  The trim also finds post-stall level flight at 4.5 and 5.5 m/s (α 44° and 30°, "harrier" flight).
- 1 g stall speed: 6.0 m/s (C_L 0.70 at α_stall in the model).
- The elevator trim does not grow with the angle of attack as it would for a stable airplane: the sign of the static
  margin. Measuring the trim elevator against airspeed is therefore the cheapest test of it.
- Linear modes at 8 m/s: longitudinal -28.4, -5.8, -3.0, +0.6 /s (no oscillatory short period; a slow divergence);
  lateral: roll -26.3 /s, Dutch roll -5.9 ± 13.5i (14.7 rad/s, damping 0.40), spiral +0.3 /s. Similar at 7 and 10 m/s.
- Hands-off from trim at 8 m/s (aileron 0.64° and rudder -0.09° added to cancel the propeller torque): within
  0.03 m of altitude and 0.25° of roll for 3 s, then the unstable modes take over (0.4 m and 2.7° at 6 s, a spiral at 10 s).
- No published flight speed exists to compare with. The manual's advice not to fly in winds above 5 mph (2.2 m/s) and
  not indoors fits a cruise near 8 m/s.

### 6.1 Implementation checks (2026-11-04, Windows, MSVC)

- **Existing airplanes unchanged.** The advanced_plane keeps its own `advanced_aerodynamics()` untouched; the Eratix
  code is behind `if constexpr` and `#if FW_AIRPLANE == 2`. `train_sac_c128_b256_tracking_g995_px4plane_i10a20` built
  before and after the change, run for 40k steps with the arguments of `slurm/tracking_px4plane_l2f.slurm`
  (`--checkpoint-interval 20000`): progress.csv (without the wall-time column), checkpoints.csv, config.json, the three
  evaluations, the three flight logs and actor.h are identical.
- **Smoke training**, `train_sac_c128_b256_tracking_g995_eratix_i10a20`, seed 0, 1.5M steps with the arguments of
  `slurm/tracking_px4plane_l2f.slurm` (`--checkpoint-interval 50000`), 571 s on one laptop core
  (a local run): episodes of 42 steps at the start, 1130 at 500k,
  1450-1500 from 800k; curriculum stage 5 by 780k; mean distance to the point 7 m at 1.32M. Validation: 0 % of
  completed flights up to 200k, 92 % at 600k, 95-97 % from 1.05M to 1.3M. Best checkpoint (1.3M): 100/100 normal
  flights completed at 6.1 m RMS position error and 0.19 m/s velocity error, 95/100 random starts at 6.1 m (4 stalls,
  1 altitude), 95/100 with the point ahead. The last actor (1.5M) was a bad swing: 43 %, mostly alpha terminations.

## 7. Task settings for the Eratix

`fixed_wing.h`, under `FW_AIRPLANE == 2` (the Aerosonde and the advanced_plane keep theirs):

| Setting | advanced_plane | Eratix | Why |
|---|---|---|---|
| airspeed_range (trimmed starts) | 12-18 m/s | 7-10 m/s | 1.17 to 1.7 times the stall speed |
| random_airspeed_range | 11-19 m/s | 6-12 m/s | never stalled at the start |
| reference_speed_range | 12-18 m/s | 7-10 m/s | around the 8 m/s cruise; a 45° turn at 7 m/s needs C_L 0.74, above the 0.70 of the stall, so the policy has to speed up for tight turns |
| reference_offset_max | 50 m | 20 m | 2.5 s of flight; a field, flown within sight |
| reference_altitude_offset_max | 15 m | 5 m | |
| reference_climb_max | 1 m/s | 0.5 m/s | |
| reference_altitude_range | 40-140 m | 15-60 m | |
| altitude_range (starts) | 50-120 m | 20-45 m | |
| altitude_floor | 20 m | 5 m | |
| altitude_error_max | 50 m | 25 m | |
| position_error_max | 300 m | 100 m | |
| airspeed_limits | 7-30 m/s | 3.5-25 m/s | 0.6 times the stall speed; past the dive speed |
| alpha_range | -15° to 18° | -14° to 16° | 2-4° past the ±12° stall |
| alpha_soft_min/max (reward margin) | -11°, 14° | -10°, 12° | 4° inside the limits |
| velocity_scale, far_velocity_scale | 4, 15 m/s | 2, 8 m/s | about half the speed |
| velocity_guidance_max | 5 m/s | 2.5 m/s | |
| far_position_scale | 100 m | 50 m | |
| curriculum gate (train_sac.cpp) | 40 m | 15 m | scaled with the start offset |

Unchanged: 50 Hz control, 2 RK4 substeps, one period of latency, 30 s episodes, the observation scalings (for example
the airspeed over 25 m/s, so about 0.3 here).

## 8. Domain randomization

`TaskConfig::randomization` multiplies every factor by U[1 - r, 1 + r]: mass, inertia, thrust (the whole propeller
model), actuator time constants and the 28 linear coefficients, which for the Eratix live in `p.aerodynamics` (the
advanced_plane keeps its coefficients in constants, so its factors only touch mass, inertia, thrust and actuators).
The ± of sections 4 and 5 are mostly 25-35 %, so r = 0.3 covers them, with three exceptions to handle before relying on
randomization:
- C_L0, C_m0, C_Y0, C_l0, C_n0 are 0, so a factor cannot move them; an additive offset (C_m0 ±0.03) would be needed.
- C_mα = +0.65 ± 0.6 can change sign; a factor keeps it positive. An additive range, or randomizing the CG, is needed.
- The stall, post-stall, deflection-drag and motor constants are not randomized.

## 9. Identification flights and tests

Ordered by what they pin down against how uncertain it is:

1. **CG, mass and inertia on the bench**: balance point of the flying airplane; bifilar pendulum about each axis.
   Removes the ±30 % on the inertia and fixes the CG the static margin depends on.
2. **Thrust stand**: thrust, rpm (optical tachometer) and current at 5-6 throttle settings, with a charged and a
   half-charged battery: C_T0, C_P0, R, I0, the static T/W. Then the thrust at speed from level flights at several
   throttles, or a car-top run: J0, kP.
3. **Servo step responses** on the bench (240 fps video of a surface, or the servo's potentiometer): time constant,
   rate limit, dead band; and the end-to-end latency of the new board.
4. **Trimmed level flights at 6.5, 8, 10, 12 m/s** (trim elevator and throttle against airspeed): the sign and size
   of the static margin (C_mα/C_mδe), C_D0 and e with the propeller model. The cheapest flight for the most uncertain
   number.
5. **Elevator doublets (3-2-1-1)** at 7, 9 and 11 m/s: C_mα, C_mq, C_mδe, C_Lα, C_Lδe; the slow pitch divergence.
6. **Power-off glides** at several speeds: the drag polar (C_D0, e) without the propeller.
7. **Aileron doublets**: C_lδa, C_lp (roll time constant about 40 ms, so log the gyro at 500 Hz or more).
8. **Rudder doublets and steady-heading sideslips**: C_nβ, C_nr, C_Yβ from the Dutch roll (about 2.3 Hz), C_lβ and
   C_nδr/C_nβ from the steady sideslip; the spiral from a hands-off bank.
9. **Slow decelerations to the stall**, power off and on, and steady high-α flight: α_stall, C_L,max, the post-stall
   C_N(α) and C_m; the elevator effectiveness at large deflections.
10. **Hover and torque roll**: the slipstream effect on the surfaces, which this model lacks entirely.

The airplane flies at 6-15 m/s and the manual advises against indoor flight, so flights 4-9 need an outdoor field with
an onboard logger (IMU, airspeed, GPS or a vision fix); hover tests fit the motion-capture room.

## 10. What is not verified

- Every aerodynamic coefficient is a textbook estimate; none is measured. The static margin is uncertain in sign.
- The geometry comes from product photos, not from the airframe; the elevator and aileron hinge lines were faint.
- The surface throws are extrapolated from the 860 mm model's manual; the 450 mm manual gives none.
- No thrust data, rpm, current or servo speed exists for these parts; the motor model is generic for a 1404 motor.
- Masses of the motor, receiver board, landing gear and foam parts are estimates (only the total and the battery are
  published).
- Not modelled: propeller slipstream over the surfaces, propeller normal force, the α̇ and β̇ terms (lumped into the
  rate derivatives), servo rate limits, the elastic foam, ground effect, the AS3X/SAFE stabilizer of the stock board
  (the policy is meant to replace it).
