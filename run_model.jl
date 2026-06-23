# run_model.jl
include(joinpath(@__DIR__, "SAFModel.jl")) # load SAFModel.jl from the same directory
include(joinpath(@__DIR__, "analysis.jl"))   # load analysis.jl from the same directory

cd(@__DIR__)                               # Set working directory to the script's directory
println("Working directory: ", pwd())

using .SAFModel
using .SAFAnalysis
import .SAFModel: params, run_scenario, extract_solution, is_solved_and_feasible;
import Pkg;
Pkg.add("JLD2");
using JLD2;
using JuMP;

const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"


# =================================================================================
# 1. Base scenarios (Status quo, first best)
# =================================================================================

policy_configs_base = (
    statusquo=(t=0.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation),
    carbontax_first_best_all=(t=190.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:all),
    carbontax_first_best_avi=(t=190.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation)
)

results_base = Dict()
for scenario in [:statusquo, :carbontax_first_best_all, :carbontax_first_best_avi]
    println("\n-- Running: $scenario --")
    results_base[scenario] = extract_solution(run_scenario(scenario, params, policy_configs_base), scenario)
end

welfare_base = display_comparison_tables(results_base, params, policy_configs_base;
    scenarios=[:statusquo, :carbontax_first_best_all, :carbontax_first_best_avi],
    title="Status quo and First Best Carbon Tax RESULTS")

@save joinpath(OUTPUT_DIR, "results_base.jld2") results_base policy_configs_base welfare_base
println("✓ Saved results_base.jld2")

# =================================================================================
# 2. Policies that achieve the same GHG emissions as XX B gallon target RFS
# =================================================================================

function find_equivalent_policies_by_emission(target_saf, params; tolerance=0.0001)
    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    # ── Step 1: RFS로 target SAF 달성 → target emissions 계산 ──
    println("\n── Step 1: Finding RFS θ_avi for target SAF = $(target_saf)B ──")
    low, high = 0.0, 1.0
    rfs_result = nothing
    for iter in 1:200
        (high - low) < 0.00001 && break
        mid = (low + high) / 2.0
        config = (t=0.0, θ_avi=mid, σ=0.0, p=0.0, carbon_tax_scope=:aviation)
        model = SAFModel.build_unified_model(params, config)
        optimize!(model)
        !is_solved_and_feasible(model) && (high=mid; continue)
        total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)
        println("  Iter $iter: θ_avi = $(round(mid, digits=6)), SAF = $(round(total_saf, digits=6))")
        rfs_result = (policy_value=mid, model=model, actual_saf=total_saf,
            config=config)
        abs(total_saf - target_saf) < tolerance && (println("  ✓ Converged"); break)
        total_saf < target_saf ? (low = mid) : (high = mid)
    end

    # RFS solution에서 target emissions 추출
    rfs_sol = extract_solution(rfs_result.model, :rfs)
    rfs_emissions = calculate_emissions_detail(rfs_sol, params)
    target_emissions = rfs_emissions.total
    println("  → Target emissions = $(round(target_emissions, digits=6)) B ton CO2e")

    # ── Step 2: 나머지 정책에서 target emissions 달성하는 stringency 찾기 ──
    search_ranges = Dict(
        :carbontax => (0.0, 1500.0),
        :lcfs => (0.0, 1.0),
        :taxcredit => (0.0, 450.0)
    )

    equivalent_policies = Dict{Symbol,Any}(:rfs => merge(rfs_result, (actual_emission=target_emissions,)))

    for policy_type in [:carbontax, :lcfs, :taxcredit]
        println("\n── Step 2: Finding $policy_type for target emissions = $(round(target_emissions, digits=6)) ──")
        low, high = search_ranges[policy_type]
        best_result = nothing

        for iter in 1:200
            (high - low) < 0.00001 && break
            mid = (low + high) / 2.0
            config = if policy_type == :carbontax
                (t=mid, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation)
            elseif policy_type == :lcfs
                (t=0.0, θ_avi=0.0, σ=mid, p=0.0, carbon_tax_scope=:aviation)
            else
                (t=0.0, θ_avi=0.0, σ=0.0, p=mid, carbon_tax_scope=:aviation)
            end

            model = SAFModel.build_unified_model(params, config)
            optimize!(model)
            !is_solved_and_feasible(model) && (high=mid; continue)

            sol = extract_solution(model, policy_type)
            em = calculate_emissions_detail(sol, params)
            total_em = em.total
            println("  Iter $iter: param = $(round(mid, digits=6)), emissions = $(round(total_em, digits=6))")

            best_result = (policy_value=mid, model=model, actual_emission=total_em, config=config)
            abs(total_em - target_emissions) < tolerance && (println("  ✓ Converged"); break)

            # emissions은 stringency 높을수록 감소
            total_em > target_emissions ? (low = mid) : (high = mid)
        end

        equivalent_policies[policy_type] = best_result
    end

    return equivalent_policies, target_emissions
end

for target_saf in [3.0, 5.0]
    suffix = target_saf == 3.0 ? "" : "_$(Int(target_saf))"

    println("\n" * "="^80)
    println("FINDING EQUIVALENT POLICIES FOR TARGET SAF = $(target_saf)B gallons")
    println("="^80)

    equivalent_policies, target_emissions = find_equivalent_policies_by_emission(target_saf, params)

    results_target = Dict(
        pt => extract_solution(equivalent_policies[pt].model, pt) for pt in POL
    )
    policy_configs_target = NamedTuple(
        pt => equivalent_policies[pt].config for pt in POL
    )

    welfare_target = display_comparison_tables(results_target, params, policy_configs_target;
        scenarios=POL,
        title="EQUIVALENT POLICIES (Target SAF = $(target_saf)B, Target Emissions = $(round(target_emissions, digits=4)) B ton CO2e)",
        equivalent_policies=equivalent_policies)

    @save joinpath(OUTPUT_DIR, "results_target$(suffix).jld2") results_target policy_configs_target equivalent_policies welfare_target target_saf target_emissions
    println("✓ Saved results_target$(suffix).jld2")
end

# =================================================================================
# 3. RFS vs. LCFS acheving 50% CI reduction
# =================================================================================

AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
    :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
δ = params.coeff.delta

# 평균 CI ratio: aviation pool 평균 CI / jet fuel CI
function aviation_ci_ratio(model)
    num = sum(δ[g] * value(model[:q][g]) for g in AVIATION_FUELS)
    den = δ[:jet_fuel] * sum(value(model[:q][g]) for g in AVIATION_FUELS)
    return num / den
end

# LCFS σ=0.5 → 평균 CI를 jet fuel 대비 50% 감소
config_lcfs50 = (t=0.0, θ_avi=0.0, σ=0.5, p=0.0, carbon_tax_scope=:aviation)

# RFS θ_avi를 bisection으로 찾기: aviation_ci_ratio = 0.5
function find_rfs_for_50CI(params; target_ratio=0.5, tolerance=1e-5, high_init=16.0)
    println("\n── Finding RFS θ_avi for aviation CI ratio = $(target_ratio) ──")
    low, high = 0.0, high_init
    rfs_result = nothing
    converged = false
    for iter in 1:200
        (high - low) < 1e-6 && break
        mid = (low + high) / 2.0
        config = (t=0.0, θ_avi=mid, σ=0.0, p=0.0, carbon_tax_scope=:aviation)
        model = SAFModel.build_unified_model(params, config)
        optimize!(model)
        if !is_solved_and_feasible(model)
            high = mid
            continue
        end
        ratio = aviation_ci_ratio(model)
        println("  Iter $iter: θ_avi = $(round(mid, digits=6)), CI ratio = $(round(ratio, digits=6))")
        rfs_result = (policy_value=mid, model=model, actual_ratio=ratio, config=config)
        if abs(ratio - target_ratio) < tolerance
            println("  ✓ Converged")
            converged = true
            break
        end
        # θ_avi 클수록 저탄소 SAF 투입 증가 → ratio 감소
        ratio > target_ratio ? (low = mid) : (high = mid)
    end
    return rfs_result, converged
end

rfs_result, converged = find_rfs_for_50CI(params)

if !converged && rfs_result.actual_ratio > 0.5
    println("\n" * "="^80)
    println("⚠ RFS로는 aviation pool 평균 CI 50% 감소에 도달 불가능합니다.")
    println("  Eligible SAF를 최대로 투입해도 최저 도달 CI ratio = $(round(rfs_result.actual_ratio, digits=6)) > 0.5")
    println("  (Conventional ATJ SAF가 50% CI threshold를 초과하여 mandate에서 제외되기 때문)")
    println("="^80)
end

config_rfs50 = rfs_result.config
println("\n→ RFS θ_avi = $(round(rfs_result.policy_value, digits=6)), achieved CI ratio = $(round(rfs_result.actual_ratio, digits=6))")

# 세 시나리오 solve
policy_configs_50CI = (
    statusquo=SQ_CONFIG,
    rfs=config_rfs50,
    lcfs=config_lcfs50
)

results_50CI = Dict()
for scenario in [:statusquo, :rfs, :lcfs]
    println("\n-- Running: $scenario --")
    results_50CI[scenario] = extract_solution(
        run_scenario(scenario, params, policy_configs_50CI), scenario)
end

welfare_50CI = display_comparison_tables(results_50CI, params, policy_configs_50CI;
    scenarios=[:rfs, :lcfs],
    title="RFS vs LCFS: 50% Aviation CI Reduction")

@save joinpath(OUTPUT_DIR, "results_50CI.jld2") results_50CI policy_configs_50CI welfare_50CI
println("✓ Saved results_50CI.jld2")