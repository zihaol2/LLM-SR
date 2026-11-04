# Method and implementation map

## State and objective

A graph node is a mechanism expression or an internal dependency. Only nodes
marked `candidate` are independently selectable. A hypothesis is a variable-size
set of selectable nodes, fitted as a sum with free external coefficients and
independent per-mechanism internal parameters. Original graph adjacency does not
constrain a hypothesis. Eight is the search/final per-equation mechanism cap.

The numerical target is the spatial-window projection of the time derivative
estimated from observations using cubic splines. Features use quintic-spline
spatial derivatives and the same windows. All observed times are used.

`Score = (MSE / target_mean_square + 1e-14) * exp(0.1 * complexity)`.

Complexity includes operations, internal parameters, external coefficients and
term connections; it is not merely set cardinality. Given internal parameters,
normalized linear least squares solves external coefficients with `rcond=1e-9`.
Internal parameters use bounded `scipy.optimize.least_squares`, at most two
screened starts, bounds [-8,8], full `max_nfev=45` and short budget 12. The start
set uses inherited parameters and [0.25, 0.5, 1, 2]. No global optimum is guaranteed.

## Growing expression graph

`grammar.py` defines the expression syntax. `builder.py` and `constructors.py`
implement the finite original construction rules. `profile_rules.py` rejects
Sinc and Hessian-determinant combinations without reintroducing the old blanket
ban on nonlinear compositions of second derivatives. Limits remain 40
operations, 600 characters and spatial order <=4.

`deduplicate.py` retains domain obligations and distinct nested numerical
derivative paths. Numeric/parameter scale aliases are removed at candidate
level when the external coefficient can absorb the scale. Dependencies needed
to evaluate retained expressions stay in the graph, without automatically
becoming independent candidates. For T01, 2,221 initial nodes include 2,170
candidates and 51 support-only nodes.

Default runs use original frozen graph snapshots, not previous fitted equations.
`initialize` optionally makes one separately logged LLM seed request;
`build-graph` constructs/deduplicates a graph from seed JSON offline.

## Sampling and testing

`rotation.py` partitions initial candidate roots into ten shuffled, fixed groups.
Rounds 1–10 and 11–20 visit the same groups. Each field tests at least half the
current roots. Sweep two first reserves any still-untested initial roots. Under
the executed quota invariant, every initial root receives an attempted fit.
Attempted coverage does not imply numerical validity, successful fitting, or
testing every complementary combination.

The LLM selects roots and their contexts through compact kept-term lists. A
nonjoint group separately adds each root to a retained set; a joint group tests
the specified complementary block together. Programmatic quota repair is logged
separately when a policy is missing or insufficient. A bad old Score does not
permanently remove a mechanism.

## Persistent sets and online growth

`graph.py` validates online proposals and adds/reuses expressions and dependencies.
The same LLM call may propose at most four roots. New roots join local/historical
proposal pools directly; the ten initial groups are not repartitioned. The same
initial-coverage guarantee is not claimed for online additions.

`memory.py` keeps valid fitted sets and edit ancestry. `recombine.py` accesses the
whole tested history for fresh/old pairing, expansion, remove-then-add and group
crossover. Fixed-parameter joint feature probes rank proposals; selected supports
are fully refitted. Ordinary local fits retain add/delete/replace proposals.
There are at most 12 ordinary and 20 historical fits per round, split over fields;
mandatory coverage fits are additional. Search crossover still obeys the eight
mechanism cap, including its existing proposal-only cap handling.

## LLM packet and API behavior

`compression.py` uses a fixed shared expression dictionary, column arrays and
compact historical records before dispatch. It preserves sampled expressions and
the standard packet's one-hop edges/operand positions, but does not claim every
verbose graph attribute is retained. A local >=50% reduction gate blocks an
insufficiently compressed request. The gate can block tiny custom libraries
because fixed instruction text dominates; it is not silently disabled.

`api.py` reserves at most one HTTP attempt per round and persists a matching
request/record. Failed requests are not automatically resent, including after
restart. The original limited syntax-only JSON recovery is retained. Complete
policy failure triggers the ordinary search quota-repair path. Credentials come
only from environment variables and never enter saved requests or manifests.

## Terminal stage

After all 20 rounds are frozen, `terminal.py` selects at most five protected
whole-archive frontier routes, including the Score winner and its simple
same-support representative. For coupled systems, per-field frontier states are
combined and protected as complete systems.

`refinement.py` provides backward deletion and all existing polynomial-prefix
comparisons. Temporary terminal fits may exceed eight terms; final capacity is
checked after pruning. The latest generic core extension compares zero-at-origin
nonlinear wrappers with existing graph cores (including multifield products),
and parameter-modulated products/quotients with simpler existing cores. It is a
tested alternative, not an unconditional linearization.

Each route supplies a best-Score anchor and, when available, its least-error-cost
lower-complexity deletion alternative. The latter may be retained despite worse
Score. Deduplication gives at most ten complete systems. A single final LLM call
selects one frozen candidate, optionally bans field-scoped numerical artifacts,
and must choose a model containing none of those banned terms. A contradiction
is recorded as failure. No automatic extra API or post-choice refit is performed.

`evaluation.py` is a separate explicit post-hoc command; reference equations are
not read by initialization, graph construction, fitting, discovery or selection.

## Numerical scope and changes made for release

The reference 1D path is unchanged. Guarded 2D fusion precombines the same fixed
derivative/window matrices, preserves nested paths, validates features and final
predictions against original operators, and falls back for sensitive models.
It is a floating-point reordering, not an assertion of identical search paths.

Packaging merges old terminal helpers and scalar/system selection into shared
modules, removes path-based dynamic imports, and applies the existing generic
core terminal rule uniformly. It does not open the 51 support-only nodes as new
candidates, change graph rules/budgets, or add a new search strategy. Those would
require distinct experiments. No live 20-round release run was performed during
packaging; offline checks and synthetic mocked integration are reported separately.
