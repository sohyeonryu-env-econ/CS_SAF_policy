# case_analysis.jl
# Self-contained: base scenarios + case1~4 equivalent-policy runs.
# (Part 1 of 2 — this part builds `all_case_results` and `base_results` in memory.
#  CSV export and figure are appended after this runs cleanly.)

include(joinpath(@__DIR__, "SAFModel.jl"))   # main-branch SAFModel (non-soy fixed)
include(joinpath(@__DIR__, "analysis.jl"))

cd(@__DIR__)
println("Working directory: ", pwd())

using .SAFModel
using .SAFAnalysis
import .SAFModel: params, run_scenario, extract_solution, is_solved_and_feasible
using JLD2
using JuMP

const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"

# =================================================================================
# 1. Base scenarios (Status quo, first best)
# =================================================================================

base_configs = (
    statusquo=(t=0.0, θ_avi=0.0, σ=0.0, p=0.0, use_ci_threshold=false, recognize_cs=true),
    firstbest=(t=190.0, θ_avi=0.0, σ=0.0, p=0.0, use_ci_threshold=false, recognize_cs=true),
)

base_results = Dict()
for scenario in [:statusquo, :firstbest]
    model = run_scenario(scenario, params, base_configs)
    sol = extract_solution(model, scenario)
    base_results[scenario] = merge(sol, (emissions=calculate_emissions_detail(sol, params),))
    println("✓ $(scenario) solved")
end

@save joinpath(OUTPUT_DIR, "results_base.jld2") base_results base_configs
println("✓ Saved results_base.jld2")

# =================================================================================
# 2. Case Definitions
# =================================================================================

cases = [
    (name=:case1, recognize_cs=true, use_ci_threshold=false, policies=[:carbontax, :rfs, :lcfs, :taxcredit]),
    (name=:case2, recognize_cs=true, use_ci_threshold=true, policies=[:rfs, :lcfs, :taxcredit]),
    (name=:case3, recognize_cs=false, use_ci_threshold=false, policies=[:carbontax, :rfs, :lcfs, :taxcredit]),
    (name=:case4, recognize_cs=false, use_ci_threshold=true, policies=[:rfs, :lcfs, :taxcredit]),
]

const TARGET_SAF = 3.0
const SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

# =================================================================================
# 3. Find Equivalent Policies Function
# =================================================================================

function find_equivalent_policies(target_saf, params, case; tolerance=0.0001)

    recognize_cs = case.recognize_cs
    use_ci_threshold = case.use_ci_threshold
    policies = case.policies

    # ── Step 1: RFS로 target SAF 달성 → target GHG 감축량 계산 ──
    println("\n── Step 1: Finding RFS θ_avi for target SAF = $(target_saf)B ($(case.name)) ──")
    low, high = 0.0, 1.0
    rfs_result = nothing

    for iter in 1:200
        (high - low) < 0.00001 && break
        mid = (low + high) / 2.0
        config = (t=0.0, θ_avi=mid, σ=0.0, p=0.0,
            use_ci_threshold=use_ci_threshold, recognize_cs=recognize_cs)
        model = SAFModel.build_unified_model(params, config)
        optimize!(model)
        !is_solved_and_feasible(model) && (high=mid; continue)
        total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)
        println("  Iter $iter: θ_avi = $(round(mid, digits=6)), SAF = $(round(total_saf, digits=6))")
        rfs_result = (policy_value=mid, model=model, actual_saf=total_saf, config=config)
        abs(total_saf - target_saf) < tolerance && (println("  ✓ Converged"); break)
        total_saf < target_saf ? (low = mid) : (high = mid)
    end

    # RFS solution에서 status quo 대비 GHG 감축량 계산
    rfs_sol = extract_solution(rfs_result.model, :rfs)
    rfs_emissions = calculate_emissions_detail(rfs_sol, params)
    sq_emissions = base_results[:statusquo].emissions  # 이미 저장된 emissions 사용
    target_emissions = rfs_emissions.total
    target_reduction = sq_emissions.total - rfs_emissions.total
    println("  → SQ emissions     = $(round(sq_emissions.total, digits=6)) B ton CO2e")
    println("  → RFS emissions    = $(round(target_emissions, digits=6)) B ton CO2e")
    println("  → Target reduction = $(round(target_reduction, digits=6)) B ton CO2e")

    # ── Step 2: 나머지 정책에서 동일 감축량 달성하는 stringency 탐색 ──
    search_ranges = Dict(
        :carbontax => (0.0, 1500.0),
        :lcfs => (0.0, 1.0),
        :taxcredit => (0.0, 450.0)
    )

    equivalent_policies = Dict{Symbol,Any}(
        :rfs => merge(rfs_result, (actual_emission=target_emissions, emission_reduction=target_reduction))
    )

    for policy_type in filter(p -> p != :rfs, policies)
        println("\n── Step 2: Finding $policy_type for reduction = $(round(target_reduction, digits=6)) ($(case.name)) ──")
        low, high = search_ranges[policy_type]
        best_result = nothing

        for iter in 1:200
            (high - low) < 0.00001 && break
            mid = (low + high) / 2.0
            config = if policy_type == :carbontax
                (t=mid, θ_avi=0.0, σ=0.0, p=0.0,
                    use_ci_threshold=use_ci_threshold, recognize_cs=recognize_cs)
            elseif policy_type == :lcfs
                (t=0.0, θ_avi=0.0, σ=mid, p=0.0,
                    use_ci_threshold=use_ci_threshold, recognize_cs=recognize_cs)
            else # taxcredit
                (t=0.0, θ_avi=0.0, σ=0.0, p=mid,
                    use_ci_threshold=use_ci_threshold, recognize_cs=recognize_cs)
            end

            model = SAFModel.build_unified_model(params, config)
            optimize!(model)
            !is_solved_and_feasible(model) && (high=mid; continue)

            sol = extract_solution(model, policy_type)
            em = calculate_emissions_detail(sol, params)
            reduction = sq_emissions.total - em.total
            println("  Iter $iter: param = $(round(mid, digits=6)), reduction = $(round(reduction, digits=6))")

            best_result = (policy_value=mid, model=model,
                actual_emission=em.total,
                emission_reduction=reduction,
                config=config)
            abs(reduction - target_reduction) < tolerance && (println("  ✓ Converged"); break)
            reduction < target_reduction ? (low = mid) : (high = mid)
        end

        equivalent_policies[policy_type] = best_result
    end

    return equivalent_policies, target_emissions, target_reduction
end

# =================================================================================
# 4. Run All Cases
# =================================================================================

all_case_results = Dict()

for case in cases
    println("\n" * "="^80)
    println("CASE: $(case.name) | recognize_cs=$(case.recognize_cs) | use_ci_threshold=$(case.use_ci_threshold)")
    println("="^80)

    equivalent_policies, target_emissions, target_reduction =
        find_equivalent_policies(TARGET_SAF, params, case)

    # extract solutions + emissions 붙이기
    results = Dict(
        pt => begin
            sol = extract_solution(equivalent_policies[pt].model, pt)
            merge(sol, (
                emissions=calculate_emissions_detail(sol, params),
            ))
        end
        for pt in case.policies
    )

    # policy configs
    policy_configs = NamedTuple(
        pt => equivalent_policies[pt].config
        for pt in case.policies
    )

    # welfare analysis
    welfare = display_comparison_tables(results, params, policy_configs;
        scenarios=case.policies,
        title="$(case.name) | recognize_cs=$(case.recognize_cs) | use_ci_threshold=$(case.use_ci_threshold)",
        equivalent_policies=equivalent_policies)

    all_case_results[case.name] = (
        case=case,
        equivalent_policies=equivalent_policies,
        results=results,
        policy_configs=policy_configs,
        welfare=welfare,
        target_emissions=target_emissions,
        target_reduction=target_reduction
    )

    case_name = case.name
    @save joinpath(OUTPUT_DIR, "results_$(case_name).jld2") results policy_configs equivalent_policies welfare target_emissions target_reduction
    println("✓ Saved results_$(case_name).jld2")
end

# 전체 저장
@save joinpath(OUTPUT_DIR, "results_all_cases.jld2") all_case_results base_results base_configs TARGET_SAF
println("✓ Saved results_all_cases.jld2")

println("\n" * "="^80)
println("✓ Part 1 complete: base_results and all_case_results are in memory.")
println("  Cases solved: ", join(string.(keys(all_case_results)), ", "))
println("="^80)


# figure_threshold.jl
# Requires `all_case_results` in memory (run case_analysis.jl first).
# Plots SAF composition by policy for case1 (Without CI threshold) vs
# case2 (With CI threshold) as horizontal stacked bar charts.

using CairoMakie

# =================================================================================
# Setup
# =================================================================================
# figure_threshold.jl
# Requires `all_case_results` in memory (run case_analysis.jl first).
# Plots SAF composition by policy for case1 (Without threshold) vs
# case2 (With threshold) as horizontal stacked bar charts.
# Matches the original figure: case1 and case2 each show RFS / LCFS / IRA only.

using CairoMakie

# =================================================================================
# Setup
# =================================================================================

const FIGURE_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/figures"

# SAF pathway stacking order and colors (same as original)
saf_components = [
    (:saf_atj_conv, "Conv ATJ-SAF", :blue),
    (:saf_atj_cs, "CS ATJ-SAF", :red),
    (:saf_hefa_conv, "Conv HEFA-SAF", :green),
    (:saf_hefa_cs, "CS HEFA-SAF", :orange),
    (:saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple),
]

# Policies shown in each panel, top to bottom
# Original figure: both panels show RFS / LCFS / IRA (no carbon tax)
panel_policies = [
    (:rfs, "RFS"),
    (:lcfs, "LCFS"),
    (:taxcredit, "IRA"),
]

# Panels: (display title, case key)
panels = [
    ("Without threshold", :case1),
    ("With threshold", :case2),
]

# =================================================================================
# Helper: get SAF quantity directly from all_case_results
# Uses safe accessor matching extract_solution's DenseAxisArray structure
# =================================================================================

function get_saf_quantity(case_key, policy, component)
    case = all_case_results[case_key]
    haskey(case.results, policy) || return 0.0
    sol = case.results[policy]
    isnothing(sol) && return 0.0
    try
        # extract_solution returns q as a JuMP DenseAxisArray;
        # index by Symbol key directly
        arr = sol.q
        # DenseAxisArray: arr[component] works if component is in the axis
        if hasproperty(arr, :data) && hasproperty(arr, :axes)
            idx = findfirst(==(component), arr.axes[1])
            isnothing(idx) && return 0.0
            v = arr.data[idx]
        else
            v = arr[component]
        end
        return (ismissing(v) || !(v isa Number)) ? 0.0 : max(0.0, float(v))
    catch
        return 0.0
    end
end

function target_reduction_mt(case_key)
    all_case_results[case_key].target_reduction * 1000  # B ton CO2e → M ton CO2e
end

# =================================================================================
# Figure
# =================================================================================

begin
    xmax = 4.0
    bar_h = 0.55
    npol = length(panel_policies)
    n_panels = length(panels)

    fig = Figure(size=(1000, 440), fontsize=13)

    for (pi, (ptitle, case_key)) in enumerate(panels)

        red = target_reduction_mt(case_key)
        subtitle = "(GHG reduction $(round(red, digits=2)) M tonCO₂e)"

        ypos = collect(npol:-1:1)          # RFS at top
        ylabels = [p[2] for p in panel_policies]

        ax = Axis(fig[1, pi];
            title=ptitle,
            subtitle=subtitle,
            titlesize=15,
            subtitlesize=12,
            titlefont=:bold,
            xlabel="SAF production (B gal)",
            xticks=0:0.5:xmax,
            yticks=(ypos, ylabels),
            ygridvisible=false,
            topspinevisible=false,
            rightspinevisible=false,
            limits=((0, xmax + 0.5), (0.4, npol + 0.6)),
        )

        for (j, (pol_key, _)) in enumerate(panel_policies)
            y = npol - j + 1
            left = 0.0

            for (comp_sym, _, comp_col) in saf_components
                v = get_saf_quantity(case_key, pol_key, comp_sym)
                v > 1e-10 || continue

                poly!(ax,
                    Rect2f(left, y - bar_h / 2, v, bar_h);
                    color=comp_col,
                    strokecolor=:white,
                    strokewidth=1.0,
                )
                left += v
            end

            # total label at end of bar
            if left > 1e-10
                text!(ax, left + 0.05, y;
                    text=string(round(left, digits=2), "B"),
                    align=(:left, :center),
                    fontsize=12,
                )
            end
        end
    end

    # Shared legend at bottom
    legend_elems = [PolyElement(color=c, strokecolor=:white) for (_, _, c) in saf_components]
    legend_labels = [lab for (_, lab, _) in saf_components]
    Legend(fig[2, 1:n_panels], legend_elems, legend_labels;
        orientation=:horizontal,
        framevisible=false,
        nbanks=1,
    )
    rowsize!(fig.layout, 2, Auto(0.12))
end

display(fig)

# =================================================================================
# Save
# =================================================================================

fig_path = joinpath(FIGURE_DIR, "CI_threshold.png")
save(fig_path, fig; px_per_unit=3)
