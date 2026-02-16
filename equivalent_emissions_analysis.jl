# equivalent_emissions_analysis.jl
cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "SAFModel.jl"))
using .SAFModel
import .SAFModel: params, build_unified_model, extract_solution
using JLD2
using JuMP

# =================================================================================
# 1. Load Target SAF Results (3B target analysis)
# =================================================================================
const TARGET_SAF_VALUE = [3.0, 5.0]

@load "results_target.jld2" equivalent_policies equivalent_solutions target_saf policy_configs_target

# =================================================================================
# 2. Calculate Target Emissions from RFS (3B target)
# =================================================================================

# Get RFS solution from 3B target analysis
rfs_solution = equivalent_solutions[:rfs]

# Use the existing function from analysis.jl
rfs_emissions = calculate_emissions_detail(rfs_solution, params)
target_total_emission = rfs_emissions.total  # billion ton CO2e

println("\n--- Target Emissions from RFS (3B SAF) ---")
println("  Aviation: $(round(rfs_emissions.aviation, digits=3)) Billion ton CO2e")
println("  Road: $(round(rfs_emissions.road, digits=3)) Billion ton CO2e")
println("  Food: $(round(rfs_emissions.food, digits=3)) Billion ton CO2e")
println("  TOTAL: $(round(target_total_emission, digits=3)) Billion ton CO2e")


# =================================================================================
# 3. Find Policy Stringencies for Target Emissions
# =================================================================================

function find_policy_for_target_emissions(target_emission, params, policy_type;
    tolerance=0.0001, max_iterations=100)
    """
    Find policy stringency that achieves target total emissions

    target_emission: target total emissions (billion ton CO2e)
    policy_type: :carbontax, :rfs, :lcfs, or :taxcredit
    """

    # Define search ranges - wider than SAF target search
    search_ranges = Dict(
        :carbontax => (0.0, 300.0, :t),
        :rfs => (0.0, 0.5, :θ_avi),
        :lcfs => (0.0, 0.5, :σ),
        :taxcredit => (0.0, 500.0, :p)
    )

    low, high, param_name = search_ranges[policy_type]
    iteration = 0

    while iteration < max_iterations && (high - low) > 0.0001
        iteration += 1
        mid = (low + high) / 2.0

        # Build config
        config = if policy_type == :carbontax
            (t=mid, θ_avi=0.0, σ=0.0, p=0.0)
        elseif policy_type == :rfs
            (t=0.0, θ_avi=mid, σ=0.0, p=0.0)
        elseif policy_type == :lcfs
            (t=0.0, θ_avi=0.0, σ=mid, p=0.0)
        else
            (t=0.0, θ_avi=0.0, σ=0.0, p=mid)
        end

        # Build and solve model
        model = build_unified_model(params, config)
        optimize!(model)

        if !is_solved_and_feasible(model)
            println("  Warning: Model not solved at $param_name = $mid")
            high = mid
            continue
        end

        # Extract solution and calculate emissions using existing function
        solution = extract_solution(model, policy_type)
        emissions = calculate_emissions_detail(solution, params)
        total_em = emissions.total

        println("  Iteration $iteration: $param_name = $(round(mid, digits=4)), " *
                "Emissions = $(round(total_em, digits=3)) Billion ton CO2e")

        # Check convergence
        if abs(total_em - target_emission) < tolerance
            println("  ✓ Found solution: $param_name = $(round(mid, digits=4))")
            return (policy_value=mid, model=model, actual_emission=total_em, config=config)
        end

        # Update search bounds
        # Higher emissions → need stricter policy → increase policy stringency
        if total_em > target_emission
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

    model = build_unified_model(params, config)
    optimize!(model)

    solution = extract_solution(model, policy_type)
    emissions = calculate_emissions_detail(solution, params)
    total_em = emissions.total

    return (policy_value=mid, model=model, actual_emission=total_em, config=config)
end

# =================================================================================
# 4. Run Analysis for Each Policy Type
# =================================================================================

policy_types = [:carbontax, :rfs, :lcfs, :taxcredit]

println("\n" * "="^130)
println("FINDING POLICY STRINGENCIES FOR TARGET EMISSIONS = $(round(target_total_emission, digits=3)) Billion ton CO2e")
println("="^130)

equivalent_emission_policies = Dict()

for policy_type in policy_types
    println("\n--- Finding $policy_type ---")
    result = find_policy_for_target_emissions(target_total_emission, params, policy_type)
    equivalent_emission_policies[policy_type] = result
end

# =================================================================================
# 5. Extract Solutions and Save Results
# =================================================================================

# Extract solutions
equivalent_emission_solutions = Dict(
    policy_type => extract_solution(result.model, policy_type)
    for (policy_type, result) in equivalent_emission_policies
)

# Extract policy configs
policy_configs_emission = NamedTuple(
    policy_type => result.config
    for (policy_type, result) in equivalent_emission_policies
)

# Save results
@save "results_equivalent_emissions.jld2" equivalent_emission_policies equivalent_emission_solutions target_total_emission policy_configs_emission

# =================================================================================
# 6. Print Summary
# =================================================================================

println("\n" * "="^130)
println("SUMMARY: Policy Stringencies for Target Emissions")
println("="^130)

using DataFrames
using Printf

summary_df = DataFrame(
    Policy=String[],
    Parameter=String[],
    Value=Float64[],
    Actual_Emissions_BilliontonC02=Float64[],
    Difference_Bton=Float64[]
)

param_labels = Dict(
    :carbontax => "Carbon Tax (\$/ton CO2e)",
    :rfs => "RFS Mandate Share",
    :lcfs => "LCFS CI Reduction (σ)",
    :taxcredit => "Tax Credit (\$/gal)"
)

for policy_type in policy_types
    result = equivalent_emission_policies[policy_type]
    param_label = param_labels[policy_type]

    push!(summary_df, (
        String(policy_type),
        param_label,
        result.policy_value,
        result.actual_emission,
        (result.actual_emission - target_total_emission)
    ))
end

show(summary_df, allrows=true)

println("\n" * "="^130)

# =================================================================================
# 7. Detailed Analysis and Comparison Tables
# =================================================================================

println("\n" * "="^130)
println("DETAILED ANALYSIS: Equivalent Emissions Policies")
println("="^130)

# Display comparison tables using existing function from analysis.jl
display_comparison_tables(
    equivalent_emission_solutions,
    params,
    policy_configs_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT EMISSIONS POLICY COMPARISON (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)",
    show_policy_params=true,
    equivalent_policies=equivalent_emission_policies
)

# =================================================================================
# 9. Welfare Analysis for Equivalent Emissions Policies
# =================================================================================

println("\n" * "="^130)
println("WELFARE ANALYSIS: Equivalent Emissions Policies")
println("="^130)

include(joinpath(@__DIR__, "welfare.jl"))

# Load status quo for welfare calculation
@load "results_base_analysis.jld2" results_base_analysis
status_quo = results_base_analysis[:statusquo]

# Calculate implicit taxes and emissions for equivalent emission solutions
println("\nCalculating implicit taxes and emissions...")
implicit_taxes_equiv_emission = calculate_all_implicit_taxes(
    equivalent_emission_solutions,
    params,
    policy_configs_emission
)

emissions_equiv_emission = Dict()
for (scenario, solution) in equivalent_emission_solutions
    emissions_equiv_emission[scenario] = calculate_emissions_detail(solution, params)
end

# Add implicit taxes and emissions to solutions
results_equiv_emission_analysis = Dict()
for (scenario, solution) in equivalent_emission_solutions
    results_equiv_emission_analysis[scenario] = merge(
        solution,
        (
            implicit_taxes=implicit_taxes_equiv_emission[scenario],
            emissions=emissions_equiv_emission[scenario]
        )
    )
end

# 1. Consumer Surplus Changes
println("\n" * "="^80)
println("1. CONSUMER SURPLUS ANALYSIS")
println("="^80)

cs_changes_equiv_emission = calculate_cs_changes(
    results_equiv_emission_analysis,
    status_quo,
    params;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_cs_changes(
    cs_changes_equiv_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT EMISSIONS: CONSUMER SURPLUS CHANGES (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e, billion \$)"
)

# 2. Land Producer Surplus Changes
println("\n" * "="^80)
println("2. LAND PRODUCER SURPLUS ANALYSIS")
println("="^80)

ps_land_equiv_emission = calculate_ps_land_changes(
    results_equiv_emission_analysis,
    status_quo,
    params;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_ps_land_changes(
    ps_land_equiv_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT EMISSIONS: LAND PRODUCER SURPLUS CHANGES (billion \$)"
)

# 3. Government Revenue Changes
println("\n" * "="^80)
println("3. GOVERNMENT REVENUE ANALYSIS")
println("="^80)

gr_changes_equiv_emission = calculate_gr_changes(
    results_equiv_emission_analysis;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_gr_changes(
    gr_changes_equiv_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT EMISSIONS: GOVERNMENT REVENUE CHANGES (billion \$)"
)

# 4. Environmental Benefits
println("\n" * "="^80)
println("4. ENVIRONMENTAL BENEFIT ANALYSIS")
println("="^80)

const SCC = 190.0  # EPA 2023 central estimate ($/ton CO2e)

env_benefits_equiv_emission = calculate_environmental_benefit(
    results_equiv_emission_analysis,
    status_quo,
    SCC;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_environmental_benefits(
    env_benefits_equiv_emission,
    SCC;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT EMISSIONS: ENVIRONMENTAL BENEFITS (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)"
)

# 5. Total Welfare Summary
println("\n" * "="^80)
println("5. TOTAL WELFARE SUMMARY")
println("="^80)

welfare_summary_equiv_emission = calculate_total_welfare(
    cs_changes_equiv_emission,
    ps_land_equiv_emission,
    gr_changes_equiv_emission,
    env_benefits_equiv_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_welfare_summary(
    welfare_summary_equiv_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT EMISSIONS: WELFARE SUMMARY (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)"
)

# Detailed CS Breakdown
cs_breakdown_df = make_cs_change_table(cs_changes_equiv_emission; scenarios=policy_types)
println("\n--- Consumer Surplus Breakdown by Sector (billion \$) ---")
show(cs_breakdown_df, allrows=true)


# 5. Total Welfare Summary
println("\n" * "="^80)
println("5. TOTAL WELFARE SUMMARY")
println("="^80)

welfare_summary_equiv_emission = calculate_total_welfare(
    cs_changes_equiv_emission,
    ps_land_equiv_emission,
    gr_changes_equiv_emission,
    env_benefits_equiv_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_welfare_summary(
    welfare_summary_equiv_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT EMISSIONS: WELFARE SUMMARY (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)"
)

# Convert welfare summary to DataFrame
welfare_comparison_df = DataFrame(
    Policy=String[],
    CS_Change=Float64[],
    PS_Land_Change=Float64[],
    Gov_Revenue=Float64[],
    Env_Benefit=Float64[],
    Social_Welfare=Float64[]
)

for policy_type in [:carbontax, :rfs, :lcfs, :taxcredit]
    welfare = welfare_summary_equiv_emission[policy_type]

    push!(welfare_comparison_df, (
        String(policy_type),
        welfare.cs_change,
        welfare.ps_land_change,
        welfare.gr_change,
        welfare.env_benefit,
        welfare.total_welfare
    ))
end

println("\n--- Welfare Comparison DataFrame (billion \$) ---")
show(welfare_comparison_df, allrows=true)

# Detailed CS Breakdown
cs_breakdown_df = make_cs_change_table(cs_changes_equiv_emission; scenarios=policy_types)
println("\n--- Consumer Surplus Breakdown by Sector (billion \$) ---")
show(cs_breakdown_df, allrows=true)


# =================================================================================
# Average Abatement Cost Calculation
# =================================================================================
aac_equiv_emission = calculate_average_abatement_cost(
    welfare_summary_equiv_emission,
    results_equiv_emission_analysis,
    status_quo;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_aac_analysis(
    aac_equiv_emission;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT EMISSIONS: AVERAGE ABATEMENT COST (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)"
)

# =================================================================================
# 10. Create All Comparison DataFrames
# =================================================================================

println("\n" * "="^130)
println("COMPREHENSIVE COMPARISON TABLES")
println("="^130)

# 1. Policy Parameters DataFrame
policy_params_df = DataFrame(
    Policy=String[],
    Parameter=String[],
    Value=Float64[],
    Actual_Emissions_Bton=Float64[],
    Diff_from_Target_Bton=Float64[]
)

param_labels_full = Dict(
    :carbontax => "Carbon Tax (\$/ton CO2e)",
    :rfs => "RFS Mandate Share",
    :lcfs => "LCFS CI Reduction (σ)",
    :taxcredit => "Tax Credit (\$/gal)"
)

for policy_type in policy_types
    result = equivalent_emission_policies[policy_type]

    push!(policy_params_df, (
        String(policy_type),
        param_labels_full[policy_type],
        result.policy_value,
        result.actual_emission,
        result.actual_emission - target_total_emission
    ))
end

println("\n--- Policy Parameters ---")
show(policy_params_df, allrows=true)

# 2. SAF Production DataFrame
saf_goods = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
saf_production_df = DataFrame(
    Policy=String[],
    saf_atj_conv=Float64[],
    saf_atj_cs=Float64[],
    saf_hefa_conv=Float64[],
    saf_hefa_cs=Float64[],
    saf_hefa_nonsoy=Float64[],
    Total_SAF=Float64[]
)

for policy_type in policy_types
    solution = equivalent_emission_solutions[policy_type]

    total_saf = sum(solution.q[g] for g in saf_goods)

    push!(saf_production_df, (
        String(policy_type),
        solution.q[:saf_atj_conv],
        solution.q[:saf_atj_cs],
        solution.q[:saf_hefa_conv],
        solution.q[:saf_hefa_cs],
        solution.q[:saf_hefa_nonsoy],
        total_saf
    ))
end

println("\n--- SAF Production (billion gallons) ---")
show(saf_production_df, allrows=true)

# 3. Emissions Comparison DataFrame
emissions_comparison_df = DataFrame(
    Policy=String[],
    Aviation_Bton=Float64[],
    Road_Bton=Float64[],
    Food_Bton=Float64[],
    Total_Bton=Float64[],
    Diff_from_Target_Bton=Float64[]
)

for policy_type in policy_types
    em = emissions_equiv_emission[policy_type]

    push!(emissions_comparison_df, (
        String(policy_type),
        em.aviation,
        em.road,
        em.food,
        em.total,
        em.total - target_total_emission
    ))
end

println("\n--- Emissions (billion ton CO2e) ---")
show(emissions_comparison_df, allrows=true)

# 4. Land Use DataFrame
land_comparison_df = DataFrame(
    Policy=String[],
    Conv_Land=Float64[],
    CS_Land=Float64[],
    Total_Land=Float64[],
    Land_Rent=Float64[]
)

for policy_type in policy_types
    solution = equivalent_emission_solutions[policy_type]

    push!(land_comparison_df, (
        String(policy_type),
        solution.l_n * 1000,  # million acres
        solution.l_cs * 1000,
        (solution.l_n + solution.l_cs) * 1000,
        solution.duals.r_land
    ))
end

println("\n--- Land Use (million acres) ---")
show(land_comparison_df, allrows=true)

println("\n" * "="^130)

# =================================================================================
# 11. Save Complete Results
# =================================================================================

# Save complete results
@save "results_equivalent_emissions_complete.jld2" equivalent_emission_policies equivalent_emission_solutions results_equiv_emission_analysis target_total_emission policy_configs_emission cs_changes_equiv_emission ps_land_equiv_emission gr_changes_equiv_emission env_benefits_equiv_emission welfare_summary_equiv_emission aac_equiv_emission SCC

println("\n✓ Complete results with welfare analysis saved to results_equivalent_emissions_complete.jld2")

# Save comparison tables
comparison_tables = Dict(
    :policy_params => policy_params_df,
    :saf_production => saf_production_df,
    :emissions => emissions_comparison_df,
    :welfare => welfare_comparison_df,
    :welfare_cs_breakdown => cs_breakdown_df,
    :land_use => land_comparison_df
)

@save "results_equivalent_emissions_tables.jld2" comparison_tables

println("✓ Comparison tables saved to results_equivalent_emissions_tables.jld2")

# =================================================================================
# equivalent_emissions_analysis.jl
cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "SAFModel.jl"))
using .SAFModel
import .SAFModel: params, build_unified_model, extract_solution
using JLD2
using JuMP
using DataFrames
using Printf

# =================================================================================
# 0. Configuration
# =================================================================================

# 분석할 타겟 SAF 목표치 설정 (run_model.jl의 TARGET_SAF_VALUES와 일치해야 함)
const TARGET_SAF_VALUES = [3.0, 5.0]  # 여러 타겟 분석 가능

# =================================================================================
# Helper Functions
# =================================================================================
include(joinpath(@__DIR__, "analysis.jl"))

function find_policy_for_target_emissions(target_emission, params, policy_type;
    tolerance=0.0001, max_iterations=100)

    search_ranges = Dict(
        :carbontax => (0.0, 300.0, :t),
        :rfs => (0.0, 0.5, :θ_avi),
        :lcfs => (0.0, 0.5, :σ),
        :taxcredit => (0.0, 500.0, :p)
    )

    low, high, param_name = search_ranges[policy_type]
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

        model = build_unified_model(params, config)
        optimize!(model)

        if !is_solved_and_feasible(model)
            println("  Warning: Model not solved at $param_name = $mid")
            high = mid
            continue
        end

        solution = extract_solution(model, policy_type)
        emissions = calculate_emissions_detail(solution, params)
        total_em = emissions.total

        println("  Iteration $iteration: $param_name = $(round(mid, digits=4)), " *
                "Emissions = $(round(total_em, digits=3)) Billion ton CO2e")

        if abs(total_em - target_emission) < tolerance
            println("  ✓ Found solution: $param_name = $(round(mid, digits=4))")
            return (policy_value=mid, model=model, actual_emission=total_em, config=config)
        end

        if total_em > target_emission
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

    model = build_unified_model(params, config)
    optimize!(model)

    solution = extract_solution(model, policy_type)
    emissions = calculate_emissions_detail(solution, params)
    total_em = emissions.total

    return (policy_value=mid, model=model, actual_emission=total_em, config=config)
end

# =================================================================================
# Load Status Quo (once, used for all targets)
# =================================================================================

include(joinpath(@__DIR__, "welfare.jl"))
@load "results_base_analysis.jld2" results_base_analysis
status_quo = results_base_analysis[:statusquo]

policy_types = [:carbontax, :rfs, :lcfs, :taxcredit]

# =================================================================================
# Main Analysis Loop - Process Each Target
# =================================================================================

for target_saf in TARGET_SAF_VALUES
    println("\n" * "="^130)
    println("PROCESSING TARGET SAF = $(target_saf)B gallons")
    println("="^130)

    # Determine file suffix
    suffix = target_saf == 3.0 ? "" : "_$(Int(target_saf))"

    # =================================================================================
    # 1. Load Target SAF Results
    # =================================================================================

    if target_saf == 3.0
        @load "results_target.jld2" equivalent_policies equivalent_solutions target_saf policy_configs_target
    elseif target_saf == 5.0
        @load "results_target_5.jld2" equivalent_policies_5 equivalent_solutions_5 target_saf_5 policy_configs_target_5
        equivalent_policies = equivalent_policies_5
        equivalent_solutions = equivalent_solutions_5
        target_saf = target_saf_5
        policy_configs_target = policy_configs_target_5
    else
        filename = "results_target_$(Int(target_saf)).jld2"
        @load filename equivalent_policies equivalent_solutions target_saf policy_configs_target
    end

    # =================================================================================
    # 2. Calculate Target Emissions from RFS
    # =================================================================================

    rfs_solution = equivalent_solutions[:rfs]
    rfs_emissions = calculate_emissions_detail(rfs_solution, params)
    target_total_emission = rfs_emissions.total

    println("\n--- Target Emissions from RFS ($(target_saf)B SAF) ---")
    println("  Aviation: $(round(rfs_emissions.aviation, digits=3)) Billion ton CO2e")
    println("  Road: $(round(rfs_emissions.road, digits=3)) Billion ton CO2e")
    println("  Food: $(round(rfs_emissions.food, digits=3)) Billion ton CO2e")
    println("  TOTAL: $(round(target_total_emission, digits=3)) Billion ton CO2e")

    # =================================================================================
    # 3. Find Policy Stringencies for Target Emissions
    # =================================================================================

    println("\n" * "="^130)
    println("FINDING POLICY STRINGENCIES FOR TARGET EMISSIONS = $(round(target_total_emission, digits=3)) Billion ton CO2e")
    println("="^130)

    equivalent_emission_policies = Dict()

    for policy_type in policy_types
        println("\n--- Finding $policy_type ---")
        result = find_policy_for_target_emissions(target_total_emission, params, policy_type)
        equivalent_emission_policies[policy_type] = result
    end

    # =================================================================================
    # 4. Extract Solutions
    # =================================================================================

    equivalent_emission_solutions = Dict(
        policy_type => extract_solution(result.model, policy_type)
        for (policy_type, result) in equivalent_emission_policies
    )

    policy_configs_emission = NamedTuple(
        policy_type => result.config
        for (policy_type, result) in equivalent_emission_policies
    )

    # Save initial results
    @save "results_equivalent_emissions$(suffix).jld2" equivalent_emission_policies equivalent_emission_solutions target_total_emission policy_configs_emission

    # =================================================================================
    # 5. Print Summary
    # =================================================================================

    println("\n" * "="^130)
    println("SUMMARY: Policy Stringencies for Target Emissions")
    println("="^130)

    summary_df = DataFrame(
        Policy=String[],
        Parameter=String[],
        Value=Float64[],
        Actual_Emissions_BilliontonCO2=Float64[],
        Difference_Bton=Float64[]
    )

    param_labels = Dict(
        :carbontax => "Carbon Tax (\$/ton CO2e)",
        :rfs => "RFS Mandate Share",
        :lcfs => "LCFS CI Reduction (σ)",
        :taxcredit => "Tax Credit (\$/gal)"
    )

    for policy_type in policy_types
        result = equivalent_emission_policies[policy_type]
        param_label = param_labels[policy_type]

        push!(summary_df, (
            String(policy_type),
            param_label,
            result.policy_value,
            result.actual_emission,
            (result.actual_emission - target_total_emission)
        ))
    end

    show(summary_df, allrows=true)
    println("\n" * "="^130)

    # =================================================================================
    # 6. Detailed Comparison Tables
    # =================================================================================

    println("\n" * "="^130)
    println("DETAILED ANALYSIS: Equivalent Emissions Policies")
    println("="^130)

    display_comparison_tables(
        equivalent_emission_solutions,
        params,
        policy_configs_emission;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
        title="EQUIVALENT EMISSIONS POLICY COMPARISON (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)",
        show_policy_params=true,
        equivalent_policies=equivalent_emission_policies
    )

    # =================================================================================
    # 7. Welfare Analysis
    # =================================================================================

    println("\n" * "="^130)
    println("WELFARE ANALYSIS: Equivalent Emissions Policies")
    println("="^130)

    # Calculate implicit taxes and emissions
    println("\nCalculating implicit taxes and emissions...")
    implicit_taxes_equiv_emission = calculate_all_implicit_taxes(
        equivalent_emission_solutions,
        params,
        policy_configs_emission
    )

    emissions_equiv_emission = Dict()
    for (scenario, solution) in equivalent_emission_solutions
        emissions_equiv_emission[scenario] = calculate_emissions_detail(solution, params)
    end

    # Add implicit taxes and emissions to solutions
    results_equiv_emission_analysis = Dict()
    for (scenario, solution) in equivalent_emission_solutions
        results_equiv_emission_analysis[scenario] = merge(
            solution,
            (
                implicit_taxes=implicit_taxes_equiv_emission[scenario],
                emissions=emissions_equiv_emission[scenario]
            )
        )
    end

    # Consumer Surplus
    println("\n" * "="^80)
    println("1. CONSUMER SURPLUS ANALYSIS")
    println("="^80)

    cs_changes_equiv_emission = calculate_cs_changes(
        results_equiv_emission_analysis,
        status_quo,
        params;
        scenarios=policy_types
    )

    display_cs_changes(
        cs_changes_equiv_emission;
        scenarios=policy_types,
        title="EQUIVALENT EMISSIONS: CONSUMER SURPLUS CHANGES (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e, billion \$)"
    )

    # Land Producer Surplus
    println("\n" * "="^80)
    println("2. LAND PRODUCER SURPLUS ANALYSIS")
    println("="^80)

    ps_land_equiv_emission = calculate_ps_land_changes(
        results_equiv_emission_analysis,
        status_quo,
        params;
        scenarios=policy_types
    )

    display_ps_land_changes(
        ps_land_equiv_emission;
        scenarios=policy_types,
        title="EQUIVALENT EMISSIONS: LAND PRODUCER SURPLUS CHANGES (billion \$)"
    )

    # Government Revenue
    println("\n" * "="^80)
    println("3. GOVERNMENT REVENUE ANALYSIS")
    println("="^80)

    gr_changes_equiv_emission = calculate_gr_changes(
        results_equiv_emission_analysis;
        scenarios=policy_types
    )

    display_gr_changes(
        gr_changes_equiv_emission;
        scenarios=policy_types,
        title="EQUIVALENT EMISSIONS: GOVERNMENT REVENUE CHANGES (billion \$)"
    )

    # Environmental Benefits
    println("\n" * "="^80)
    println("4. ENVIRONMENTAL BENEFIT ANALYSIS")
    println("="^80)

    env_benefits_equiv_emission = calculate_environmental_benefit(
        results_equiv_emission_analysis,
        status_quo,
        SCC;
        scenarios=policy_types
    )

    display_environmental_benefits(
        env_benefits_equiv_emission,
        SCC;
        scenarios=policy_types,
        title="EQUIVALENT EMISSIONS: ENVIRONMENTAL BENEFITS (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)"
    )

    # Total Welfare Summary
    println("\n" * "="^80)
    println("5. TOTAL WELFARE SUMMARY")
    println("="^80)

    welfare_summary_equiv_emission = calculate_total_welfare(
        cs_changes_equiv_emission,
        ps_land_equiv_emission,
        gr_changes_equiv_emission,
        env_benefits_equiv_emission;
        scenarios=policy_types
    )

    display_welfare_summary(
        welfare_summary_equiv_emission;
        scenarios=policy_types,
        title="EQUIVALENT EMISSIONS: WELFARE SUMMARY (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)"
    )

    # Welfare Comparison DataFrame
    welfare_comparison_df = DataFrame(
        Policy=String[],
        CS_Change=Float64[],
        PS_Land_Change=Float64[],
        Gov_Revenue=Float64[],
        Env_Benefit=Float64[],
        Social_Welfare=Float64[]
    )

    for policy_type in policy_types
        welfare = welfare_summary_equiv_emission[policy_type]

        push!(welfare_comparison_df, (
            String(policy_type),
            welfare.cs_change,
            welfare.ps_land_change,
            welfare.gr_change,
            welfare.env_benefit,
            welfare.social_welfare
        ))
    end

    println("\n--- Welfare Comparison DataFrame (billion \$) ---")
    show(welfare_comparison_df, allrows=true)

    # CS Breakdown
    cs_breakdown_df = make_cs_change_table(cs_changes_equiv_emission; scenarios=policy_types)
    println("\n--- Consumer Surplus Breakdown by Sector (billion \$) ---")
    show(cs_breakdown_df, allrows=true)

    # Average Abatement Cost
    aac_equiv_emission = calculate_average_abatement_cost(
        welfare_summary_equiv_emission,
        results_equiv_emission_analysis,
        status_quo;
        scenarios=policy_types
    )

    display_aac_analysis(
        aac_equiv_emission;
        scenarios=policy_types,
        title="EQUIVALENT EMISSIONS: AVERAGE ABATEMENT COST (Target = $(round(target_total_emission, digits=3)) Billion ton CO2e)"
    )

    # =================================================================================
    # 8. Create Comparison DataFrames
    # =================================================================================

    println("\n" * "="^130)
    println("COMPREHENSIVE COMPARISON TABLES")
    println("="^130)

    # Policy Parameters
    policy_params_df = DataFrame(
        Policy=String[],
        Parameter=String[],
        Value=Float64[],
        Actual_Emissions_Bton=Float64[],
        Diff_from_Target_Bton=Float64[]
    )

    param_labels_full = Dict(
        :carbontax => "Carbon Tax (\$/ton CO2e)",
        :rfs => "RFS Mandate Share",
        :lcfs => "LCFS CI Reduction (σ)",
        :taxcredit => "Tax Credit (\$/gal)"
    )

    for policy_type in policy_types
        result = equivalent_emission_policies[policy_type]

        push!(policy_params_df, (
            String(policy_type),
            param_labels_full[policy_type],
            result.policy_value,
            result.actual_emission,
            result.actual_emission - target_total_emission
        ))
    end

    println("\n--- Policy Parameters ---")
    show(policy_params_df, allrows=true)

    # SAF Production
    saf_goods = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    saf_production_df = DataFrame(
        Policy=String[],
        saf_atj_conv=Float64[],
        saf_atj_cs=Float64[],
        saf_hefa_conv=Float64[],
        saf_hefa_cs=Float64[],
        saf_hefa_nonsoy=Float64[],
        Total_SAF=Float64[]
    )

    for policy_type in policy_types
        solution = equivalent_emission_solutions[policy_type]
        total_saf = sum(solution.q[g] for g in saf_goods)

        push!(saf_production_df, (
            String(policy_type),
            solution.q[:saf_atj_conv],
            solution.q[:saf_atj_cs],
            solution.q[:saf_hefa_conv],
            solution.q[:saf_hefa_cs],
            solution.q[:saf_hefa_nonsoy],
            total_saf
        ))
    end

    println("\n--- SAF Production (billion gallons) ---")
    show(saf_production_df, allrows=true)

    # Emissions Comparison
    emissions_comparison_df = DataFrame(
        Policy=String[],
        Aviation_Bton=Float64[],
        Road_Bton=Float64[],
        Food_Bton=Float64[],
        Total_Bton=Float64[],
        Diff_from_Target_Bton=Float64[]
    )

    for policy_type in policy_types
        em = emissions_equiv_emission[policy_type]

        push!(emissions_comparison_df, (
            String(policy_type),
            em.aviation,
            em.road,
            em.food,
            em.total,
            em.total - target_total_emission
        ))
    end

    println("\n--- Emissions (billion ton CO2e) ---")
    show(emissions_comparison_df, allrows=true)

    # Land Use
    land_comparison_df = DataFrame(
        Policy=String[],
        Conv_Land=Float64[],
        CS_Land=Float64[],
        Total_Land=Float64[],
        Land_Rent=Float64[]
    )

    for policy_type in policy_types
        solution = equivalent_emission_solutions[policy_type]

        push!(land_comparison_df, (
            String(policy_type),
            solution.l_n * 1000,
            solution.l_cs * 1000,
            (solution.l_n + solution.l_cs) * 1000,
            solution.duals.r_land
        ))
    end

    println("\n--- Land Use (million acres) ---")
    show(land_comparison_df, allrows=true)

    println("\n" * "="^130)

    # =================================================================================
    # 9. Save Complete Results
    # =================================================================================

    @save "results_equivalent_emissions_complete$(suffix).jld2" equivalent_emission_policies equivalent_emission_solutions results_equiv_emission_analysis target_total_emission policy_configs_emission cs_changes_equiv_emission ps_land_equiv_emission gr_changes_equiv_emission env_benefits_equiv_emission welfare_summary_equiv_emission aac_equiv_emission SCC

    println("\n✓ Complete results saved to results_equivalent_emissions_complete$(suffix).jld2")

    # Save comparison tables
    comparison_tables = Dict(
        :policy_params => policy_params_df,
        :saf_production => saf_production_df,
        :emissions => emissions_comparison_df,
        :welfare => welfare_comparison_df,
        :welfare_cs_breakdown => cs_breakdown_df,
        :land_use => land_comparison_df
    )

    @save "results_equivalent_emissions_tables$(suffix).jld2" comparison_tables

    println("✓ Comparison tables saved to results_equivalent_emissions_tables$(suffix).jld2")
end

println("\n" * "="^130)
println("ALL TARGET EMISSIONS ANALYSES COMPLETED")
println("="^130)
