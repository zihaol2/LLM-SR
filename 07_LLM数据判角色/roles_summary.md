# Data-only role identification (deepseek-v4-flash, 5 repeats each)

Only measured statistics were given; no physics description and no equation was requested.

## Topography_Chemotaxis
**f1**  (reference truth: population density)
- seed1: saturation of a diffusive/heat-like field (evidence: values range 0→1, monotonically increasing in time, spatial spectrum becomes smooth/low-k as it saturates, initial localized bump spreads and fills domain, no negative values, rms drift increases as evolution accelerates toward steady state)
- seed2: temperature or concentration field undergoing diffusion/relaxation to a uniform steady state (evidence: initially localized positive bump that spreads and saturates to +1 everywhere; all values nonnegative, monotonic rise, spatial spectrum dominated by k=1 and decaying higher modes, rms drift increasing over time consistent with self-similar diffusion).
- seed3: concentration/temperature-like scalar field undergoing diffusion toward a uniform steady state (evidence: positive-only values in [0,1], initially localized Gaussian-like bump that spreads and saturates to ~1 everywhere, dominant k=1 mode with growth of large-scale component over time, and monotonic increase in RMS in second half indicating relaxation to uniformity)
- seed4: temperature or concentration (evidence: strictly nonnegative, bounded [0,1], smooth spatial modes k=1,2,4, monotone increase toward saturation, drift accelerating over time)
- seed5: scalar temperature or concentration field undergoing diffusion/relaxation (evidence: field remains nonnegative, values grow from 0 to 1 over time, spatial spectrum dominated by low k with smooth increase, late-time flattening to uniform value with small rms drift growth indicating diffusive equilibration)

## Morphogenesis
**f1**  (reference truth: morphogen concentration)
- seed1: probability density (evidence: nonnegative, compact support, smooth localized bump that grows and drifts, spectrum dominated by large scales, increasing drift over time suggests diffusion/advection of a conserved or normalized quantity)
- seed2: Gaussian scalar field (evidence: strictly nonnegative, spatially localized bump with smooth k=1-3 dominant spectrum, amplitude grows over time, no negative values, consistent with squared/complex modulus or passive scalar concentration)
- seed3: evolving localized pulse/heat distribution (evidence: nonnegative field, smooth low-k spatial spectrum, amplitude growing over time, compact support near center, zero at boundaries)
- seed4: <heat-like diffusive pulse amplitude> (evidence: strictly nonnegative, initially zero then growing localized bump in interior, smooth low-k spectrum dominated by k=1-3, monotonic rise with increasing rms drift, no propagation of sharp fronts)
- seed5: concentration of a localized diffusing/scalar quantity (evidence: nonnegative, compact support grows in width, peak amplitude increases slightly, smooth spatial spectrum dominated by low k, and rms drift increases over time; could also be temperature or density perturbation, but nonnegativity and localized Gaussian-like build-up favor concentration)

## Forced_Swift_Hohenberg
**f1**  (reference truth: order parameter)
- seed1: localized wave amplitude (evidence: values bounded near zero with slow growth, dominant low-k spectrum, rms drift increasing 3.06× over time, spatial pattern resembling a growing coherent pulse)
- seed2: amplitude of a growing unstable mode (evidence: values remain small initially, then grow with rms drift increasing 3.06× in second half; spatial spectrum concentrated at low k=2–4, suggesting large-scale pattern; signs and magnitudes at sample rows show coherent growth and spatial structure, consistent with a linear instability or wave amplitude rather than a conserved or diffusive field)
- seed3: spatially localized growing disturbance (evidence: f1 ranges -0.362 to +0.872, mean near 0, negative fraction ~0.5; spectrum dominated by low-wavenumber modes k=2–4; rms drift triples over second half; time samples show a narrow positive pulse at x≈1.25 growing from 0.052 to 0.435 while adjacent points stay small)
- seed4: amplitude of a growing unstable mode (evidence: rms drift ratio 3.06 from first to second half indicates exponential growth; spatial spectrum dominated by low wavenumbers k=2–4 suggests large-scale instability; sign/magnitude range and near-zero mean consistent with a developing wave pattern in a linear instability regime)
- seed5: <most likely a wave amplitude or density perturbation field, e.g., surface elevation or acoustic pressure> (evidence: small zero-mean oscillations with growing amplitude over time, dominant low wavenumbers 2–4 indicative of large-scale wave modes, and rest frame at t=0 with symmetric positive/negative excursions; rms drift second half/first half = 3.06 confirms transient or unstable growth; no clear sign bias, mean +0.017, negative fraction 0.51 suggests oscillatory symmetric field)

## Traffic_Flow_Bottleneck
**f1**  (reference truth: traffic density)
- seed1: scalar density or concentration field undergoing advection-diffusion from a localized source (evidence: strictly positive values with min 0.100 (background), localized Gaussian-like bump growing and spreading in time, dominant low-k spectrum indicating smooth blob, stationary statistics with rms drift ~1.0, no negative values).
- seed2: localized traveling pulse/wave amplitude (evidence: positive-only values, compact support moving through grid with k=1-3 dominant spectrum, stationary rms over time)
- seed3: probability density / concentration (evidence: strictly positive, bounded 0.1–0.7, smooth low-wavenumber spatial spectrum with k=1 dominant, diffusive-like spreading and amplitude growth at central peak over time, no sign change, stationary spectrum drift ratio ≈1)
- seed4: localized propagating pulse/bump (evidence: strictly positive values, bounded range 0.1–0.7, energy concentrated at low wavenumbers with smooth evolution, drift remains statistically stationary, values rise and spread symmetrically from a localized initial perturbation)
- seed5: concentration/temperature-like scalar field advecting and diffusing from a localized source (evidence: strictly positive, bounded between +0.100 and +0.700, mean ≈0.158, smooth low-wavenumber spectrum dominated by k=1–3, stationary late-time RMS drift ≈0.99, and a Gaussian-like bump that develops and spreads symmetrically in time from initial localized peaks).

## Predator_Prey
**f1**  (reference truth: prey density)
- seed1: non-conserved scalar concentration field undergoing relaxation/diffusion toward a uniform state (evidence: values bounded [0,1], monotonic increase from localized initial bump to near-uniform ~0.97, spectrum dominated by k=1 with growing large-scale amplitude, RMS drift increasing over time)
- seed2: amplitude of a slowly evolving dominant mode (evidence: nonnegative, bounded 0–1, mean 0.65, nearly all power at k=1 and 2, monotone increase to saturation, drift ratio >1 indicates ongoing slow growth)
- seed3: concentration or scalar field undergoing diffusion/relaxation to a uniform positive steady state (evidence: values strictly between 0 and 1, initially localized, monotonic increase toward ~0.96–1.0 everywhere, spectrum dominated by k=1 then flattening, rms drift increasing 1.57× as gradients smooth out)
- seed4: concentration/order parameter (evidence: strictly nonnegative, saturates near 1 over time, low-k dominant spectrum, monotone growth toward a uniform state)
- seed5: concentration/temperature-like scalar field (evidence: strictly nonnegative, bounded 0–1, monotonic rise to near-uniform saturation, low-wavenumber-dominated spectrum, persistent growth)

**f2**  (reference truth: predator density)
- seed1: diffusing/conserved passive tracer or density anomaly that spreads and decays in spatial gradients (evidence: small positive values, initial localized peak at x~5.6, amplitude decays strongly over time, spectrum shifts to lower k, RMS drift decreasing by half, weak correlation with f1)
- seed2: secondary/coupled smaller mode (evidence: nonnegative, max 0.40, mean ~0.03, spectrum peaked at low k but broader, peak near mid-domain, strong decay over time, drift ratio <1 indicates relaxation to a quasi-steady configuration)
- seed3: derived quantity such as a flux, gradient magnitude, or reaction rate that is transient and decays (evidence: small positive values, peak at early time near steep f1 gradients, decays to near zero with rms drift decreasing to 0.51, spectrum also low-k but with faster damping)
- seed4: reaction/production rate or secondary field (evidence: small nonnegative peaks localized in space, decays over time, k=1&2 spectrum, weak correlation with f1, rms decreases second half)
- seed5: secondary transported quantity or source/sink field (evidence: nonnegative, small magnitude, peaked structure, decays over time, weak correlation with f1, spectrum more spread to higher k)
