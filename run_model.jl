# run_model.jl
include(joinpath(@__DIR__, "SAFModel.jl"))
using .SAFModel
import .SAFModel: params, run_scenario, extract_solution, is_solved_and_feasible
import Pkg;
Pkg.add("JLD2");
using JLD2
using JuMP

cd(@__DIR__)
println("Working directory: ", pwd())

# =================================================================================
# 1. Run Base Scenarios
# =================================================================================

# Define policy stringency
policy_configs_base = (
    statusquo=(t=0.0, θ_avi=0.0, σ=0.0, p=0.0),
    carbontax=(t=250.0, θ_avi=0.0, σ=0.0, p=0.0),
    rfs=(t=0.0, θ_avi=0.3, σ=0.0, p=0.0),
    lcfs=(t=0.0, θ_avi=0.0, σ=0.03, p=0.0),
    taxcredit=(t=0.0, θ_avi=0.0, σ=0.0, p=10.0)
)

results_base = Dict()

for scenario in [:statusquo, :carbontax, :rfs, :lcfs, :taxcredit]
    println("\n====== Running Base: $scenario ======")
    model = run_scenario(scenario, params, policy_configs_base)  # ⭐ 직접 전달
    results_base[scenario] = extract_solution(model, scenario)
end

# ========== 결과 저장 ==========
@save "results_base.jld2" results_base policy_configs_base
println("\n✓ Base results saved to results_base.jld2")

# =================================================================================
# 2. Target SAF Analysis (3 billion gallons)
# =================================================================================

println("\n" * "="^80)
println("PART 2: TARGET SAF ANALYSIS")
println("="^80)

# Find policy stringency function
function find_policy_for_target_saf(target_saf, params, policy_type; tolerance=0.001)
    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    search_ranges = Dict(
        :carbontax => (0.0, 1500.0, :t),
        :rfs => (0.0, 1.0, :θ_avi),
        :lcfs => (0.0, 1.0, :σ),
        :taxcredit => (0.0, 450.0, :p)
    )

    low, high, param_name = search_ranges[policy_type]
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations && (high - low) > 0.0001
        iteration += 1
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

        if !is_solved_and_feasible(model)
            println("  Warning: Model not solved at $param_name = $mid")
            high = mid
            continue
        end

        total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)
        println("  Iteration $iteration: $param_name = $(round(mid, digits=3)), SAF = $(round(total_saf, digits=3))")

        if abs(total_saf - target_saf) < tolerance
            println("  ✓ Found solution: $param_name = $(round(mid, digits=3))")
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

# Run target SAF analysis
target_saf = 3.0
policy_types = [:carbontax, :rfs, :lcfs, :taxcredit]

println("\nFINDING POLICY STRINGENCY FOR TARGET SAF = $target_saf billion gallons")

equivalent_policies = Dict()
for policy_type in policy_types
    println("\n--- Finding $policy_type ---")
    result = find_policy_for_target_saf(target_saf, params, policy_type)
    equivalent_policies[policy_type] = result
end

# Extract solutions
equivalent_solutions = Dict(
    :carbontax => extract_solution(equivalent_policies[:carbontax].model, :carbontax),
    :rfs => extract_solution(equivalent_policies[:rfs].model, :rfs),
    :lcfs => extract_solution(equivalent_policies[:lcfs].model, :lcfs),
    :taxcredit => extract_solution(equivalent_policies[:taxcredit].model, :taxcredit)
)

policy_configs_target = (
    carbontax=equivalent_policies[:carbontax].config,
    rfs=equivalent_policies[:rfs].config,
    lcfs=equivalent_policies[:lcfs].config,
    taxcredit=equivalent_policies[:taxcredit].config
)

# 결과 저장
@save "results_target.jld2" equivalent_policies equivalent_solutions target_saf policy_configs_target
println("\n✓ Target SAF results saved to results_target_saf.jld2")

# =================================================================================
# Summary
# =================================================================================

println("\n" * "="^80)
println("ANALYSIS COMPLETE!")
println("="^80)
println("Saved files:")
println("  1. results_base.jld2 - Base scenario results")
println("  2. results_target_saf.jld2 - Target SAF analysis results")