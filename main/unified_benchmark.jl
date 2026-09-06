# unified_benchmark.jl
#
# Engine: the social-planner IPOPT by default (see section 0).  Set SAF_ENGINE=mcp to reproduce the
# same tables from the market-equilibrium MCP; the two agree scenario by scenario.
#
# Output: results_unified_benchmark.jld2

include(joinpath(@__DIR__, "scenarios.jl"))
using .Scenarios
include(joinpath(@__DIR__, "model_mkt.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "model_sp.jl"))

cd(@__DIR__)
println("Working directory: ", pwd())

using .ModelMkt
using .Analysis
import .ModelMkt: params, run_scenario, extract_solution, is_solved_and_feasible
import .ModelSP: build_planner_model, extract_planner_solution
using JLD2
using JuMP

# =================================================================================
# 0. Engine
# =================================================================================

#   :planner  the social-planner NLP of main/model_sp.jl, solved with Ipopt.
#   :mcp      the market-equilibrium MCP of main/model_mkt.jl, solved with PATH.

const ENGINE = Symbol(get(ENV, "SAF_ENGINE", "planner"))
ENGINE in (:planner, :mcp) || error("SAF_ENGINE must be planner or mcp, got $(ENGINE)")
println("Engine: ", ENGINE == :planner ? "social planner (Ipopt)" : "market equilibrium MCP (PATH)")

# solve_point(config, policy; warm_start=nothing)
# Solve one (config, policy) with the selected engine; returns the solution NamedTuple or
# `nothing` if it did not converge.
# Warm start (planner only): previous accepted bisection iterate, else the MCP solution at the
# same config. The fallback is numerical only; the reported solution is always the planner's.
function solve_point(config, policy; warm_start=nothing)
    if ENGINE == :mcp
        model = build_unified_model(params, config)
        optimize!(model)
        return is_solved_and_feasible(model) ? extract_solution(model, policy) : nothing
    end

    for ws in (warm_start, :from_mcp)
        if ws === :from_mcp
            m = build_unified_model(params, config)
            optimize!(m)
            is_solved_and_feasible(m) || return nothing
            ws = extract_solution(m, policy)
        end
        pm = build_planner_model(params, config; warm_start=ws)
        optimize!(pm)
        st = termination_status(pm)
        if st in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
            return extract_planner_solution(pm, policy, params)
        end
    end
    return nothing
end

include(joinpath(@__DIR__, "paths.jl"))
using .Paths
Paths.setup()

const OUTPUT_DIR = Paths.DATA_DIR

# =================================================================================
# 1. Base scenarios (status quo, first best)
# =================================================================================

base_configs = (
    statusquo=(t=0.0, θ_avi=0.0, σ=0.0, p=0.0, use_ci_threshold=false, recognize_cs=true),
    scc_tax=(t=190.0, θ_avi=0.0, σ=0.0, p=0.0, use_ci_threshold=false, recognize_cs=true),
)

base_results = Dict()
for scenario in [:statusquo, :scc_tax]
    sol = solve_point(getproperty(base_configs, scenario), scenario)
    isnothing(sol) && error("$(scenario) did not solve under engine $(ENGINE)")
    base_results[scenario] = merge(sol, (emissions=calculate_emissions_detail(sol, params),))
    println("✓ $(scenario) solved")
end

const SQ_EMISSIONS = base_results[:statusquo].emissions.total

# =================================================================================
# 2. Case / scenario definitions
# =================================================================================

const TARGET_SAF = 3.0 # billion gallons
const SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

# Canonical case numbering, shared by every table and figure script.
cases = Scenarios.CASES

# Benchmark case: volumetric mandate without CS accounting
const BENCH_CASE = :case1
const BENCH_POLICY = :rfs

# Bisection brackets for the stringency search.
const SEARCH_RANGES = Dict(
    :carbontax => (0.0, 1500.0),
    :rfs => (0.0, 1.0),
    :lcfs => (0.0, 1.0),
    :taxcredit => (0.0, 5000.0),
)

make_config(policy, value, recognize_cs, use_ci_threshold) =
    (t=policy == :carbontax ? value : 0.0,
        θ_avi=policy == :rfs ? value : 0.0,
        σ=policy == :lcfs ? value : 0.0,
        p=policy == :taxcredit ? value : 0.0,
        use_ci_threshold=use_ci_threshold,
        recognize_cs=recognize_cs)

stringency_of(policy, config) =
    policy == :carbontax ? config.t :
    policy == :rfs ? config.θ_avi :
    policy == :lcfs ? config.σ :
    policy == :taxcredit ? config.p : 0.0

total_saf(sol) = sum(sol.q[g] for g in SAF_GOODS)

# =================================================================================
# 3. Bisection search for the stringencies that produce the same emissions reduction
# =================================================================================

function bisect_policy(policy, recognize_cs, use_ci_threshold, metric, target;
    tolerance=1e-9, max_iter=300, label="")

    low, high = SEARCH_RANGES[policy]
    span = high - low
    best = nothing
    warm = nothing

    for iter in 1:max_iter
        (high - low) < 1e-10 * span && break
        mid = (low + high) / 2.0
        config = make_config(policy, mid, recognize_cs, use_ci_threshold)
        sol = solve_point(config, policy; warm_start=warm)
        if isnothing(sol)
            high = mid
            continue
        end
        warm = sol

        m = metric(sol)
        println("  Iter $iter: param = $(round(mid, digits=6)), $label = $(round(m, digits=6))")
        best = (policy_value=mid, sol=sol, config=config, metric_value=m)

        abs(m - target) < tolerance && (println("  ✓ Converged"); break)
        m < target ? (low = mid) : (high = mid)
    end

    return best
end

# =================================================================================
# 4. Step 1, the benchmark: RFS, no CS, no threshold, 3B gal SAF
# =================================================================================

bench_case = cases[findfirst(c -> c.name == BENCH_CASE, cases)]

println("\n" * "="^80)
println("BENCHMARK: $(BENCH_POLICY) | case = $(BENCH_CASE) | recognize_cs=$(bench_case.recognize_cs) | " *
        "use_ci_threshold=$(bench_case.use_ci_threshold) | target SAF = $(TARGET_SAF) B gal")
println("="^80)

bench = bisect_policy(BENCH_POLICY, bench_case.recognize_cs, bench_case.use_ci_threshold,
    total_saf, TARGET_SAF; label="SAF")
isnothing(bench) && error("Benchmark RFS search failed to find any feasible solution.")

bench_sol = bench.sol
bench_em = calculate_emissions_detail(bench_sol, params)

const TARGET_EMISSIONS = bench_em.total
const TARGET_REDUCTION = SQ_EMISSIONS - TARGET_EMISSIONS

println("\n--- Benchmark calibration ---")
println("  θ_avi              = $(round(bench.policy_value, digits=6))")
println("  Total SAF          = $(round(bench.metric_value, digits=6)) B gal")
println("  SQ emissions       = $(round(SQ_EMISSIONS, digits=6)) B ton CO2e")
println("  Benchmark emissions= $(round(TARGET_EMISSIONS, digits=6)) B ton CO2e")
println("  TARGET REDUCTION   = $(round(TARGET_REDUCTION, digits=6)) B ton CO2e " *
        "($(round(TARGET_REDUCTION * 1000, digits=3)) M ton CO2e)")

# =================================================================================
# 5. Step 2: every other scenario matched to TARGET_REDUCTION
# =================================================================================

reduction_metric(sol) = SQ_EMISSIONS - calculate_emissions_detail(sol, params).total

all_case_results = Dict()
match_log = NamedTuple[]   # convergence diagnostics for every scenario

for case in cases
    println("\n" * "="^80)
    println("CASE: $(case.name) | recognize_cs=$(case.recognize_cs) | use_ci_threshold=$(case.use_ci_threshold)")
    println("="^80)

    equivalent_policies = Dict{Symbol,Any}()

    for policy in case.policies
        if case.name == BENCH_CASE && policy == BENCH_POLICY
            equivalent_policies[policy] = merge(bench, (
                actual_saf=bench.metric_value,
                actual_emission=TARGET_EMISSIONS,
                emission_reduction=TARGET_REDUCTION,
            ))
            println("\n── $(policy) ($(case.name)) is the BENCHMARK, reused, no search ──")
            continue
        end

        println("\n── Finding $(policy) in $(case.name) for reduction = " *
                "$(round(TARGET_REDUCTION, digits=6)) B ton CO2e ──")
        res = bisect_policy(policy, case.recognize_cs, case.use_ci_threshold,
            reduction_metric, TARGET_REDUCTION; label="reduction")

        if isnothing(res)
            @warn "No feasible solution found for $(policy) in $(case.name)"
            equivalent_policies[policy] = nothing
            continue
        end

        sol = res.sol
        equivalent_policies[policy] = merge(res, (
            actual_saf=total_saf(sol),
            actual_emission=calculate_emissions_detail(sol, params).total,
            emission_reduction=res.metric_value,
        ))
    end

    # solutions + emissions
    results = Dict(
        pt => begin
            sol = equivalent_policies[pt].sol
            merge(sol, (emissions=calculate_emissions_detail(sol, params),))
        end
        for pt in case.policies if !isnothing(equivalent_policies[pt])
    )

    policy_configs = NamedTuple(
        pt => equivalent_policies[pt].config
        for pt in case.policies if !isnothing(equivalent_policies[pt])
    )

    solved_policies = [pt for pt in case.policies if haskey(results, pt)]

    # convergence diagnostics
    for pt in solved_policies
        ep = equivalent_policies[pt]
        push!(match_log, (
            case=case.name, policy=pt,
            stringency=stringency_of(pt, ep.config),
            saf=ep.actual_saf,
            reduction=ep.emission_reduction,
            gap=ep.emission_reduction - TARGET_REDUCTION,
            at_bracket_top=abs(ep.policy_value - SEARCH_RANGES[pt][2]) < 1e-4,
        ))
    end

    welfare = display_comparison_tables(results, params, policy_configs;
        scenarios=solved_policies,
        title="$(case.name) | recognize_cs=$(case.recognize_cs) | use_ci_threshold=$(case.use_ci_threshold)",
        equivalent_policies=equivalent_policies)

    all_case_results[case.name] = (
        case=case,
        equivalent_policies=equivalent_policies,
        results=results,
        policy_configs=policy_configs,
        welfare=welfare,
        target_emissions=TARGET_EMISSIONS,
        target_reduction=TARGET_REDUCTION,
    )
end

# =================================================================================
# 6. Convergence report
# =================================================================================

println("\n" * "="^100)
println("EQUIVALENCE CHECK: every scenario should hit reduction = " *
        "$(round(TARGET_REDUCTION, digits=6)) B ton CO2e")
println("="^100)
println(rpad("scenario", 26), rpad("stringency", 14), rpad("SAF (Bgal)", 13),
    rpad("reduction", 13), rpad("gap", 13), "at max or min?")
for r in match_log
    tag = string(r.case, "/", r.policy)
    println(rpad(tag, 26),
        rpad(round(r.stringency, digits=5), 14),
        rpad(round(r.saf, digits=4), 13),
        rpad(round(r.reduction, digits=6), 13),
        rpad(round(r.gap, digits=8), 13),
        r.at_bracket_top ? "YES, target may be unreachable" : "")
end
println("="^100)

# =================================================================================
# 7. Save
# =================================================================================

bench_info = (
    case=BENCH_CASE,
    policy=BENCH_POLICY,
    recognize_cs=bench_case.recognize_cs,
    use_ci_threshold=bench_case.use_ci_threshold,
    θ_avi=bench.policy_value,
    total_saf=bench.metric_value,
    sq_emissions=SQ_EMISSIONS,
    target_emissions=TARGET_EMISSIONS,
    target_reduction=TARGET_REDUCTION,
)

@save joinpath(OUTPUT_DIR, "results_unified_benchmark.jld2") all_case_results base_results base_configs bench_info match_log TARGET_SAF

println("\n✓ Saved results_unified_benchmark.jld2")
println("  Next: run extract_results.jl (RESULT_SET = :unified) to write the CSV.")
