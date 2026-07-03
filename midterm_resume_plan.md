# Midterm resume plan

## Target

Finish a defensible midterm presentation resume in about two weeks. The resume should not claim that HardNet has been implemented yet. The current claim is:

```text
Standard PINNs were first tested on Poisson and then moved to a nonlinear PDE.
For 1D Burgers, a simple hard IC/BC ansatz exactly satisfies initial and boundary conditions.
The first seed shows similar L2 error to soft PINNs, so multi-seed verification is the next required step.
```

## Minimum story for the resume

1. Background: PINNs solve PDEs by minimizing PDE residual and condition losses.
2. Preliminary work: Poisson equation was used to build the baseline workflow.
3. Problem found: Soft-penalty PINNs require balancing PDE/BC/IC losses and may not strictly satisfy constraints.
4. New target: 1D viscous Burgers equation as a first nonlinear PDE.
5. Current method: compare soft PINNs with a hard IC/BC output transform.
6. Result so far: hard IC/BC exactly satisfies IC/BC, but L2 improvement is not yet conclusive.
7. Next direction: multi-seed verification, then HardNet/HardNet++-style constraint enforcement.

## Must-have experiments

| priority | experiment | purpose | output needed |
|---|---|---|---|
| P0 | Burgers soft vs hard-icbc, seeds 0,1,2, 2h per run | check seed stability | summary table, fields plot, smoothed losses |
| P0 | analyze mean/std across seeds | avoid over-claiming from seed 0 | mean L2, mean PDE loss, IC/BC losses |
| P1 | repeat best setting with L-BFGS refinement | see if PDE residual gap changes | comparison table |
| P1 | read HardNet/HardNet++ method sections | clarify what is actually adopted | method summary paragraph |
| P2 | viscosity or collocation ablation | test harder/easier nonlinear regimes | optional table |

## Immediate GPU command

Run this next on the research GPU:

```bash
python run_burgers_constraint_comparison.py --seeds 0,1,2 --runtime-sec 7200 --epochs 1000000 --n-interior 4096 --n-initial 512 --n-boundary 512 --hidden-dim 128 --hidden-layers 5 --lr 0.001 --wandb-mode offline --output-root outputs/burgers_constraint_comparison_2h_seeds012
```

This runs six jobs sequentially:

```text
soft seed 0      up to 2h
hard-icbc seed 0 up to 2h
soft seed 1      up to 2h
hard-icbc seed 1 up to 2h
soft seed 2      up to 2h
hard-icbc seed 2 up to 2h
```

Maximum wall time is about 12 hours.

## Two-week schedule

| date range | task | deliverable |
|---|---|---|
| Days 1-2 | Run seeds 0,1,2 and collect results | `summary.md`, representative figures |
| Days 3-4 | Analyze mean/std and failure region | result paragraph and table |
| Days 5-6 | Add L-BFGS or one controlled follow-up if needed | one extra comparison, only if multi-seed result is unclear |
| Days 7-8 | Summarize HardNet and HardNet++ accurately | method comparison notes |
| Days 9-10 | Draft resume structure | first resume draft |
| Days 11-12 | Insert figures/tables and polish explanation | near-final resume |
| Days 13-14 | Final check and advisor-facing wording | final PDF/Word/LaTeX output |

## Resume outline

1. Title and objective
2. Background: PINNs and hard constraints
3. Preliminary Poisson baseline
4. Nonlinear PDE target: 1D Burgers equation
5. Methods: soft PINNs vs hard IC/BC ansatz
6. Experimental setup
7. Results and discussion
8. Future work: HardNet/HardNet++ and harder flow problems

## Wording guardrails

- Say `hard IC/BC ansatz`, not `HardNet implementation`, for the current code.
- Say `finite-difference reference solution`, not `analytic solution`, for Burgers.
- Say `seed 0 suggests...` until seeds 0,1,2 are complete.
- Separate observed facts from hypotheses.
