# Recorded development results

These are the frozen results from the research runs, not new live searches of this release. MSE is the all-observation spatial-window residual. Structural support ignores fitted numerical coefficients/free scales according to the saved evaluator.

| Case | MSE | Complexity | Structural Jaccard | API attempts | Reported total tokens |
|---|---:|---:|---:|---:|---:|
| T01 | 1.26155e-12 | 22 | 1.0000 | 21 | 599,072 |
| T02 | 1.49734e-14 | 21 | 1.0000 | 21 | 953,835 |
| T03 | 3.73602e-08 | 29 | 1.0000 | 21 | 866,995 |
| T04 | 1.93113e-14 | 33 | 1.0000 | 21 | 536,784 |
| T05 | 3.32140e-13 | 40 | 1.0000 | 22 | 1,487,365 |
| T2D | 2.97211e-15 | 38 | 1.0000 | 21 | 2,495,585 |

T05 includes an authorized generic existing-subexpression extension and one extra final selection. The original 20-round search was kept. T01–T04 and T2D passed separate offline extension regression, without a new final API or a new search.

Across the six standard cases, recorded usage is 127 API attempts and 6,939,636 reported tokens. This excludes other development runs, initialization and role inference; two failed 2D requests returned no usage, so it is not an exact billing total. T05 had 17/20 usable search policies and T2D 18/20; the remaining cases had 20/20. Programmatic quota repair is not counted as an LLM decision.

## Historical T03 ablations

A drops one-hop neighbor/edge inputs; B quarters the mandatory-test quota; AB combines both. All retain program-side graph/history and the original local budget. These historical runs predate the latest general core refinement, whereas the release ablation entry points share the latest terminal stage.

| Configuration | Declared-choice Jaccard | True-term recovery | Choice/ban consistency | Total tokens | Attempt coverage |
|---|---:|---:|---|---:|---:|
| baseline | 1.0000 | 100% | passed | 866,995 | 100.0% |
| A | 1.0000 | 100% | FAILED | 417,313 | 100.0% |
| B | 0.8333 | 100% | FAILED | 812,649 | 54.1% |
| AB | 0.7143 | 100% | passed | 393,092 | 54.1% |

A recovered the correct five-term expanded support but the LLM selected a model containing terms it also banned. B contained all five true terms plus an extra sinh term, and also failed consistency. Their metrics describe the candidate actually named by the LLM, not accepted final outputs. AB passed selection checks but retained two redundant terms.

A reduced total tokens by about 51.9% in this single T03 run. This is not evidence that all other cases can discard neighbor information. Reduced fitting quota did not necessarily reduce elapsed time: internal parameter counts, optimizer paths and runtime contention matter. One run per setting is insufficient for significance claims.

## Limitations

- Development benchmark feedback was used to improve terminal rules; this is not held-out blind evaluation.
- There is no matched-budget causal demonstration that the LLM, history, or one graph component alone caused the gains.
- All six final supports match under the saved metric; this does not establish unique physical roles, global parameter optima, parameter uncertainty or held-out predictive generalization.
- 3D graph-only diagnostics and unstable physical-role guesses are not presented as successful discovery experiments.
- The release keeps the original 1D numerical path and the guarded 2D acceleration; a microbenchmark speedup is not an end-to-end guarantee.

Complete original fitted coefficients, per-field equations and historical ablation metrics are stored in `gem_gss/assets/published_records.json`. Only the explicit verification/evaluation tools load recorded answers or references.
