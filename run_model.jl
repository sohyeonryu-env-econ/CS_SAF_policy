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

# 4 cases defined by configs
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
        !is_solved_and_feasible(model) && (high = mid; continue)
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
            !is_solved_and_feasible(model) && (high = mid; continue)

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

# run_model.jl의 4번 섹션 수정

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