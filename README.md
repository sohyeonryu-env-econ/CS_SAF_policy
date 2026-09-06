# CS_SAF_policy

Replication code for the climate-smart (CS) SAF policy paper. The model is a
partial-equilibrium model of the US aviation, road fuel, and crop sectors, used to compare
four policy instruments (carbon tax, volumetric mandate / RFS, CI standard / LCFS, tax
credit) at a common level of GHG abatement.

## Requirements

Julia 1.11. From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs both solvers: PATH (`PATHSolver.jl`) and Ipopt.

## Two engines

The same economic problem is written twice, and the two must agree:

| File | Formulation | Solver |
| --- | --- | --- |
| `main/model_mkt.jl` | market equilibrium, as a mixed complementarity problem | PATH |
| `main/model_sp.jl` | social planner, as a nonlinear program | Ipopt |

The social planner is the default engine. Set `SAF_ENGINE=mcp` to reproduce the same
tables from the MCP instead; the two agree scenario by scenario.

Because the two are separate formulations of one model, **any change to the economics has
to go into both files**, otherwise they stop describing the same problem.

## How to run

Everything downstream is built on one anchor, so run these in order:

```bash
julia --project=. main/unified_benchmark.jl
```

Solves every (policy, case) combination and tunes each one to the same GHG abatement:
the abatement delivered by an RFS at 3 billion gallons of total SAF. Writes
`output/data/results_unified_benchmark.jld2`.

```bash
julia --project=. main/3B_outcomes.jl
```

**This is where the results are.** Reads the jld2 above and writes the tables and figures
into `output/tables/` and `output/figures/`:

## Output

Everything lands in `output/` (`data/`, `tables/`, `figures/`), which is gitignored: a
single jld2 runs to hundreds of MB. Set `SAF_OUTPUT` to write somewhere else:

```bash
SAF_OUTPUT=/path/to/results julia --project=. main/3B_outcomes.jl
```