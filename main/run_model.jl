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
    statusquo=(t=0.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation, use_ci_threshold=false, recognize_cs=true),
    carbontax_first_best_all=(t=190.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:all, use_ci_threshold=false, recognize_cs=true),
    carbontax_first_best_avi=(t=190.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation, use_ci_threshold=false, recognize_cs=true)
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


# =================================================================================
# 2. Policies that achieve the same GHG emissions as XX B gallon target RFS
# =================================================================================

function find_equivalent_policies_by_emission(target_saf, params; tolerance=0.0001)
    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    # Step 1: RFS target SAF → calculate the target emissions
    println("\n── Step 1: Finding RFS θ_avi for target SAF = $(target_saf)B ──")
    low, high = 0.0, 1.0
    rfs_result = nothing
    for iter in 1:200
        (high - low) < 0.00001 && break
        mid = (low + high) / 2.0
        config = (t=0.0, θ_avi=mid, σ=0.0, p=0.0, carbon_tax_scope=:aviation, use_ci_threshold=true, recognize_cs=true)
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

    # Find target emissions from the RFS solution
    rfs_sol = extract_solution(rfs_result.model, :rfs)
    rfs_emissions = calculate_emissions_detail(rfs_sol, params)
    target_emissions = rfs_emissions.total
    println("  → Target emissions = $(round(target_emissions, digits=6)) B ton CO2e")

    # Step 2: Find the stringency for each remaining policy that achieves the target emissions
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
                (t=mid, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation,
                    use_ci_threshold=true, recognize_cs=true)
            elseif policy_type == :lcfs
                (t=0.0, θ_avi=0.0, σ=mid, p=0.0, carbon_tax_scope=:aviation,
                    use_ci_threshold=false, recognize_cs=true)
            else  # taxcredit
                (t=0.0, θ_avi=0.0, σ=0.0, p=mid, carbon_tax_scope=:aviation,
                    use_ci_threshold=true, recognize_cs=true)
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

            total_em > target_emissions ? (low = mid) : (high = mid)
        end

        equivalent_policies[policy_type] = best_result
    end

    return equivalent_policies, target_emissions
end

# Find equivalent policies for different target SAF levels
for target_saf in [3.0, 6.0]
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