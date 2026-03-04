# run_model.jl
include(joinpath(@__DIR__, "SAFModel.jl")) # load SAFModel.jl from the same directory
cd(@__DIR__)                               # Set working directory to the script's directory
println("Working directory: ", pwd())

using .SAFModel;
import .SAFModel: params, run_scenario, extract_solution, is_solved_and_feasible;
import Pkg;
Pkg.add("JLD2");
using JLD2;
using JuMP;

const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results_trials/"

# =================================================================================
# 1. Run Base Scenarios (status quo + example policies)
# =================================================================================

# Define policy stringency configuration
#config = (
#    t=t,
#    θ_avi=θ_avi,
#    θ_d6=θ_d6,
#    σ=σ,
#    p=p,
#    D1=D1,
#    D2=D2,
#    use_ccs=use_ccs
#)
policy_configs_base = (
    statusquo=(t=0.0, θ_avi=0.0, θ_d6=0.125, σ=0.0, p=0.0, D1=false, D2=false, use_ccs=false),
    carbontax=(t=190.0, θ_avi=0.0, θ_d6=0.125, σ=0.0, p=0.0, D1=false, D2=false, use_ccs=false),
    rfs=(t=0.0, θ_avi=0.3, θ_d6=0.125, σ=0.0, p=0.0, D1=false, D2=false, use_ccs=false),
    lcfs=(t=0.0, θ_avi=0.0, θ_d6=0.125, σ=0.03, p=0.0, D1=false, D2=false, use_ccs=false),
    taxcredit=(t=0.0, θ_avi=0.0, θ_d6=0.125, σ=0.0, p=10.0, D1=false, D2=false, use_ccs=false)
)

results_base = Dict()

# run
for scenario in [:statusquo, :carbontax, :rfs, :lcfs, :taxcredit]
    println("\n====== Running Base: $scenario ======")
    model = run_scenario(scenario, params, policy_configs_base)
    results_base[scenario] = extract_solution(model, scenario)
end

# save results
@save joinpath(OUTPUT_DIR, "results_base.jld2") results_base policy_configs_base
println("\n✓ Base results saved to results_base.jld2")

# =================================================================================
# 2. Equivalent SAF Analysis (target = 3 billion gallons of total SAF)
# =================================================================================

# Find policy stringency function
function find_policy_for_target_saf(target_saf, params, policy_type; tolerance=0.0001)
    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_atj_conv_ccs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    search_ranges = Dict(
        :carbontax => (0.0, 1500.0, :t),
        :rfs => (0.0, 1.0, :θ_avi),
        :lcfs => (0.0, 1.0, :σ),
        :taxcredit => (0.0, 450.0, :p)
    )

    low, high, param_name = search_ranges[policy_type]
    max_iterations = 200
    iteration = 0

    while iteration < max_iterations && (high - low) > 0.00001
        iteration += 1
        mid = (low + high) / 2.0

        config = if policy_type == :carbontax
            (t=mid, θ_avi=0.0, θ_d6=0.125, σ=0.0, p=0.0, D1=false, D2=false, use_ccs=false)
        elseif policy_type == :rfs
            (t=0.0, θ_avi=mid, θ_d6=0.125, σ=0.0, p=0.0, D1=false, D2=false, use_ccs=false)
        elseif policy_type == :lcfs
            (t=0.0, θ_avi=0.0, θ_d6=0.125, σ=mid, p=0.0, D1=false, D2=false, use_ccs=false)
        else
            (t=0.0, θ_avi=0.0, θ_d6=0.125, σ=0.0, p=mid, D1=false, D2=false, use_ccs=false)
        end

        model = SAFModel.build_unified_model(params, config)
        optimize!(model)

        if !is_solved_and_feasible(model)
            println("  Warning: Model not solved at $param_name = $mid")
            high = mid
            continue
        end

        total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)
        println("  Iteration $iteration: $param_name = $(round(mid, digits=7)), SAF = $(round(total_saf, digits=7))")

        if abs(total_saf - target_saf) < tolerance
            println("  ✓ Found solution: $param_name = $(round(mid, digits=7))")
            return (policy_value=mid, model=model, actual_saf=total_saf, config=config)
        end

        if total_saf < target_saf
            low = mid
        else
            high = mid
        end
    end

    println("  ✗ Max iterations reached. Returning best solution found.")
    mid = (low + high) / 2.0

    config = if policy_type == :carbontax
        (t=mid, θ_avi=0.0, σ=0.0, p=0.0)
    elseif policy_type == :rfs
        (t=0.0, θ_avi=mid, σ=0.0, p=0.0)
    elseif policy_type == :lcfs
        (t=0.0, θ_avi=0.0, σ=mid, p=0.0)
    else
        (t=0.0, θ_avi=0.0, σ=0.0, p=mid)
    end

    model = SAFModel.build_unified_model(params, config)
    optimize!(model)
    total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)

    return (policy_value=mid, model=model, actual_saf=total_saf, config=config)
end

# Multiple Target
# Run target SAF analysis
const TARGET_SAF_VALUES = [3.0, 5.0]

policy_types = [:carbontax, :rfs, :lcfs, :taxcredit]

for target_saf in TARGET_SAF_VALUES
    println("\n" * "="^80)
    println("FINDING POLICY STRINGENCY FOR TARGET SAF = $target_saf billion gallons")
    println("="^80)

    equivalent_policies = Dict()
    for policy_type in policy_types
        println("\n--- Finding $policy_type for $(target_saf)B SAF ---")
        result = find_policy_for_target_saf(target_saf, params, policy_type)
        equivalent_policies[policy_type] = result
    end

    # Extract solutions
    equivalent_solutions = Dict(
        policy_type => extract_solution(result.model, policy_type)
        for (policy_type, result) in equivalent_policies
    )

    # Extract policy configs
    policy_configs_target = NamedTuple(
        policy_type => result.config
        for (policy_type, result) in equivalent_policies
    )

    # Determine output suffix
    suffix = target_saf == 3.0 ? "" : "_$(Int(target_saf))"

    # Save with appropriate naming
    if target_saf == 3.0
        @save joinpath(OUTPUT_DIR, "results_target.jld2") equivalent_policies equivalent_solutions target_saf policy_configs_target

        println("\n✓ Target SAF (3B) results saved to results_target.jld2")
    else
        # 변수명에 suffix 추가
        target_saf_var = target_saf
        equivalent_policies_var = equivalent_policies
        equivalent_solutions_var = equivalent_solutions
        policy_configs_target_var = policy_configs_target

        if target_saf == 5.0
            @save joinpath(OUTPUT_DIR, "results_target_5.jld2") equivalent_policies_5 = equivalent_policies equivalent_solutions_5 = equivalent_solutions target_saf_5 = target_saf policy_configs_target_5 = policy_configs_target
            println("\n✓ Target SAF (5B) results saved to results_target_5.jld2")
        else
            filename = "results_target_$(Int(target_saf)).jld2"
            @save filename equivalent_policies equivalent_solutions target_saf policy_configs_target
            println("\n✓ Target SAF ($(target_saf)B) results saved to $filename")
        end
    end
end

println("\n" * "="^80)
println("ALL TARGET SAF ANALYSES COMPLETED")
println("="^80)

# ================================================================================
# RFS modifications
# ================================================================================


function find_θavi_for_rfs_scenarios(target_saf, params; tolerance=0.0001, max_iterations=200)
    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_atj_conv_ccs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    # 각 시나리오별 base config (θ_avi 제외)
    scenario_configs = (
        status_quo=(t=0.0, θ_d6=0.125, σ=0.0, p=0.0, D1=false, D2=false, use_ccs=false),
        rfs_avi=(t=0.0, θ_d6=0.125, σ=0.0, p=0.0, D1=false, D2=false, use_ccs=false),
        rfs_nested=(t=0.0, θ_d6=0.125, σ=0.0, p=0.0, D1=true, D2=false, use_ccs=false),
        rfs_obligation=(t=0.0, θ_d6=0.125, σ=0.0, p=0.0, D1=true, D2=true, use_ccs=false),
        rfs_obligation_strong=(t=0.0, σ=0.03, p=0.0, D1=true, D2=true, use_ccs=false)
    )

    results = Dict()

    for (scenario_name, base) in pairs(scenario_configs)
        println("\n--- Finding θ_avi for $scenario_name ---")
        if scenario_name == :status_quo
            config = merge(base, (θ_avi=0.0,))
            model = SAFModel.build_unified_model(params, config)
            optimize!(model)
            total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)
            results[scenario_name] = (θ_avi=0.0, model=model, actual_saf=total_saf, config=config)
            println("  ✓ Status quo: θ_avi = 0.0, SAF = $(round(total_saf, digits=4))")
            continue  # binary search 건너뜀
        end

        low, high = 0.0, 1.0
        iteration = 0

        while iteration < max_iterations && (high - low) > 0.00001
            iteration += 1
            mid = (low + high) / 2.0

            # rfs_obligation_strong은 θ_d6 = θ_avi + 0.125
            config = if scenario_name == :rfs_obligation_strong
                merge(base, (θ_avi=mid, θ_d6=mid + 0.125))
            else
                merge(base, (θ_avi=mid,))
            end

            model = SAFModel.build_unified_model(params, config)
            optimize!(model)

            if !is_solved_and_feasible(model)
                println("  Warning: not solved at θ_avi = $mid")
                high = mid
                continue
            end

            total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)
            println("  Iter $iteration: θ_avi = $(round(mid, digits=6)), SAF = $(round(total_saf, digits=4))")

            if abs(total_saf - target_saf) < tolerance
                println("  ✓ Found: θ_avi = $(round(mid, digits=6))")
                results[scenario_name] = (θ_avi=mid, model=model, actual_saf=total_saf, config=config)
                break
            end

            total_saf < target_saf ? (low = mid) : (high = mid)
        end

        if !haskey(results, scenario_name)
            mid = (low + high) / 2.0
            config = if scenario_name == :rfs_obligation_strong
                merge(base, (θ_avi=mid, θ_d6=mid + 0.125))
            else
                merge(base, (θ_avi=mid,))
            end
            model = SAFModel.build_unified_model(params, config)
            optimize!(model)
            total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)
            results[scenario_name] = (θ_avi=mid, model=model, actual_saf=total_saf, config=config)
            println("  ✗ Max iterations. Best θ_avi = $(round(mid, digits=6))")
        end
    end

    return results
end

# 실행
rfs_results = find_θavi_for_rfs_scenarios(3.0, params)
rfs_solutions = Dict(
    scenario => extract_solution(result.model, scenario)
    for (scenario, result) in rfs_results
)
rfs_policy_configs = NamedTuple(
    scenario => result.config
    for (scenario, result) in rfs_results
)
@save joinpath(OUTPUT_DIR, "results_rfs_modification.jld2") rfs_results rfs_solutions rfs_policy_configs
println("\n✓ RFS variation results saved to results_rfs_modification.jld2")