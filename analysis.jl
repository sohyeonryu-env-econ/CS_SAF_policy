# analysis.jl
include(joinpath(@__DIR__, "SAFModel.jl"))
using .SAFModel
using JLD2
using DataFrames
using Printf


# =====================
# Display Comparison Tables and individual Tables
# =====================
function display_comparison_tables(solutions, params, policy_configs;
    scenarios=nothing,
    title="RESULTS",
    show_policy_params=false,  # ⭐ Target 분석용 옵션
    equivalent_policies=nothing)  # ⭐ Target 분석용 데이터

    println("\n" * "="^130)
    println(title)
    println("="^130)

    # ⭐ Policy Parameters (Target 분석에만 표시)
    if show_policy_params && !isnothing(equivalent_policies)
        policy_labels = Dict(
            :carbontax => ("Carbon Tax", "Carbon Tax (\$/ton CO2e)"),
            :rfs => ("RFS Aviation", "Mandate Share"),
            :lcfs => ("LCFS", "CI Reduction (σ)"),
            :taxcredit => ("Tax Credit", "Rate (\$/gal)")
        )

        println("\n--- Policy Parameters ---")
        param_df = DataFrame(
            Policy=String[],
            Parameter_Name=String[],
            Parameter_Value=Float64[],
            Actual_SAF=Float64[]
        )

        for policy_type in [:carbontax, :rfs, :lcfs, :taxcredit]
            if haskey(equivalent_policies, policy_type)
                result = equivalent_policies[policy_type]
                config = result.config
                label, param_name = policy_labels[policy_type]

                param_value = if policy_type == :carbontax
                    config.t
                elseif policy_type == :rfs
                    config.θ_avi
                elseif policy_type == :lcfs
                    config.σ
                else
                    config.p
                end

                push!(param_df, (label, param_name, param_value, result.actual_saf))
            end
        end

        show(param_df, allrows=true)
    end

    # Calculate implicit taxes
    implicit_taxes = calculate_all_implicit_taxes(solutions, params, policy_configs)

    tables = [
        ("Implicit Taxes/Subsidies (\$/gallon)", make_implicit_tax_table(implicit_taxes, params; scenarios=scenarios, exclude_statusquo=true)),
        ("Production", make_production_table(solutions, params; scenarios=scenarios)),
        ("Demand", make_demand_table(solutions; scenarios=scenarios)),
        ("Prices", make_price_table(solutions; scenarios=scenarios)),
        ("Land Use (Million Acres)", make_land_table(solutions, params; scenarios=scenarios)),
        ("Emissions (MMT CO2e)", make_emissions_table(solutions, params; scenarios=scenarios)),
        ("Dual Variables", make_duals_table(solutions; scenarios=scenarios))
    ]

    for (table_name, table) in tables
        println("\n--- $table_name ---")
        show(table, allrows=true)
    end

    println("\n" * "="^130)
end

# =====================
# 1) Implicit Tax/Subsidy Calculation
# =====================

function calculate_implicit_taxes(solution, params, config)
    if isnothing(solution)
        return nothing
    end

    # Tuple unpacking
    t = config.t
    θ_avi = config.θ_avi
    σ = config.σ
    p = config.p
    #t, θ_avi, σ, p = config

    # Get coefficients
    delta = params.coeff.delta
    delta_mj = params.coeff.delta_mj
    baselineCI = params.coeff.baselineCI

    # Get dual variables
    duals = solution.duals
    γ_avi = duals.λ_rfs_avi  # RFS aviation dual
    μ = duals.λ_lcfs          # LCFS dual

    # Aviation fuels
    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    # Initialize results
    implicit_tax = Dict(g => Dict(
        :carbon_tax => 0.0,
        :rfs_avi => 0.0,
        :lcfs => 0.0,
        :tax_credit => 0.0,
        :total => 0.0
    ) for g in AVIATION_FUELS)

    for g in AVIATION_FUELS
        # 1. Carbon tax component
        implicit_tax[g][:carbon_tax] = t * delta[g]

        # 2. RFS aviation component
        if g == :jet_fuel
            implicit_tax[g][:rfs_avi] = γ_avi * θ_avi
        else  # SAF goods
            implicit_tax[g][:rfs_avi] = (delta[g] <= 0.5 * delta[:jet_fuel]) ? -γ_avi * 1.6 : 0.0
        end

        # 3. LCFS component
        implicit_tax[g][:lcfs] = -μ * ((1 - σ) * delta[:jet_fuel] - delta[g])

        # 4. Tax credit component (SAF only)
        if g != :jet_fuel
            implicit_tax[g][:tax_credit] = -tax_credit_rate(delta_mj[g], baselineCI, p)
        end

        # Total
        implicit_tax[g][:total] = (
            implicit_tax[g][:carbon_tax] +
            implicit_tax[g][:rfs_avi] +
            implicit_tax[g][:lcfs] +
            implicit_tax[g][:tax_credit]
        )
    end

    return implicit_tax
end

"""
Calculate implicit taxes for all solutions
"""
function calculate_all_implicit_taxes(solutions, params, policy_configs)
    implicit_taxes = Dict()
    for (scenario, solution) in solutions
        if !isnothing(solution)
            config = getproperty(policy_configs, scenario)
            implicit_taxes[scenario] = calculate_implicit_taxes(solution, params, config)
        end
    end
    return implicit_taxes
end

"""
Create implicit tax table
"""
function make_implicit_tax_table(implicit_taxes, params; scenarios=nothing, exclude_statusquo=true)
    scenario_list = isnothing(scenarios) ? collect(keys(implicit_taxes)) : scenarios
    labels = params.meta[:process_labels]

    if exclude_statusquo
        scenario_list = filter(s -> s != :statusquo, scenario_list)
    end

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    # Define which tax component to show for each scenario
    scenario_tax_type = Dict(
        :statusquo => :total,
        :carbontax => :carbon_tax,
        :rfs => :rfs_avi,
        :lcfs => :lcfs,
        :taxcredit => :tax_credit
    )

    df = DataFrame(Fuel=String[])
    for scenario in scenario_list
        df[!, String(scenario)] = Float64[]
    end

    for g in AVIATION_FUELS
        push!(df.Fuel, labels[g])
        for scenario in scenario_list
            if haskey(implicit_taxes, scenario) && !isnothing(implicit_taxes[scenario])
                tax_type = get(scenario_tax_type, scenario, :total)
                value = implicit_taxes[scenario][g][tax_type]
                push!(df[!, String(scenario)], value)
            else
                push!(df[!, String(scenario)], 0.0)
            end
        end
    end

    return df
end

"""
Display implicit taxes with custom formatting
"""
function display_implicit_taxes(implicit_taxes, params; scenarios=nothing, show_statusquo=false)
    labels = params.meta[:process_labels]

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    scenario_tax_type = Dict(
        :carbontax => :carbon_tax,
        :rfs => :rfs_avi,
        :lcfs => :lcfs,
        :taxcredit => :tax_credit
    )

    scenario_labels = Dict(
        :carbontax => "Carbon Tax",
        :rfs => "RFS",
        :lcfs => "LCFS",
        :taxcredit => "Tax Credit"
    )

    # Determine scenarios to display
    if isnothing(scenarios)
        display_scenarios = collect(keys(implicit_taxes))
    else
        display_scenarios = scenarios
    end

    # Exclude statusquo if requested
    if !show_statusquo
        display_scenarios = filter(s -> s != :statusquo, display_scenarios)
    end

    println("\n" * "="^90)
    println("IMPLICIT TAXES/SUBSIDIES BY POLICY (\$/gallon)")
    println("="^90)

    # Header
    @printf("%-30s", "Fuel")
    for scenario in display_scenarios
        label = get(scenario_labels, scenario, String(scenario))
        @printf("%15s", label)
    end
    println()
    println("-"^90)

    # Data rows
    for g in AVIATION_FUELS
        @printf("%-30s", labels[g])
        for scenario in display_scenarios
            if haskey(implicit_taxes, scenario) && !isnothing(implicit_taxes[scenario])
                tax_type = get(scenario_tax_type, scenario, :total)
                value = implicit_taxes[scenario][g][tax_type]
                @printf("%15.4f", value)
            else
                @printf("%15s", "N/A")
            end
        end
        println()
    end

    println("="^90)
end

# =====================
# 2) Production, Demand, Price, Land Use, Emissions, Duals Tables
# =====================
function make_production_table(solutions, params; scenarios=nothing)
    labels = params.meta[:process_labels]
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(Product=String[])
    for scenario in scenario_list
        df[!, scenario] = Float64[]
    end

    # Fuel goods
    for g in FUEL_GOODS
        push!(df.Product, labels[g])
        for scenario in scenario_list
            push!(df[!, scenario], solutions[scenario].q[g])
        end
    end

    # Feedstock
    feedstock_goods = [:feedstock_corn_n, :feedstock_corn_cs, :feedstock_soy_n, :feedstock_soy_cs]
    feedstock_keys = [:corn_n, :corn_cs, :soy_n, :soy_cs]

    for (g, key) in zip(feedstock_goods, feedstock_keys)
        push!(df.Product, labels[g])
        for scenario in scenario_list
            push!(df[!, scenario], solutions[scenario].q_feedstock[key])
        end
    end

    # DDGS
    push!(df.Product, "DDGS")
    for scenario in scenario_list
        push!(df[!, scenario], solutions[scenario].ddgs)
    end

    # Soymeal
    push!(df.Product, "Soymeal Production (MMT)")
    for scenario in scenario_list
        push!(df[!, scenario], solutions[scenario].soymeal_produ)
    end

    return df
end

function make_demand_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(
        Scenario=String[],
        Aviation=Float64[],
        Gasoline=Float64[],
        Diesel=Float64[],
        Corn=Float64[],
        Corn_excl_DDGS=Float64[],
        Soyoil=Float64[],
        Soymeal=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]
        push!(df, (
            String(scenario),
            sol.x[:avi],
            sol.x[:gas],
            sol.x[:die],
            sol.x[:corn],
            sol.x[:corn] - sol.ddgs,
            sol.x[:soyoil],
            sol.x[:soymeal]
        ))
    end

    return df
end

function make_price_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(
        Scenario=String[],
        Aviation=Float64[],
        Gasoline=Float64[],
        Diesel=Float64[],
        ConvCorn=Float64[],
        CSCorn=Float64[],
        ConvSoy=Float64[],
        CSSoy=Float64[],
        Soymeal=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]
        push!(df, (
            String(scenario),
            sol.p_c[:avi],
            sol.p_c[:gas],
            sol.p_c[:die],
            sol.p_f[:feedstock_corn_n],
            sol.p_f[:feedstock_corn_cs],
            sol.p_f[:feedstock_soy_n],
            sol.p_f[:feedstock_soy_cs],
            sol.p_c[:soymeal]
        ))
    end

    return df
end

function make_land_table(solutions, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    omega = params.coeff.omega

    df = DataFrame(
        Scenario=String[],
        Corn_N=Float64[],
        Corn_CS=Float64[],
        Corn_Total=Float64[],
        Soy_N=Float64[],
        Soy_CS=Float64[],
        Soy_Total=Float64[],
        Total_N=Float64[],
        Total_CS=Float64[],
        Grand_Total=Float64[],
        Land_Rent=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]

        corn_land_n = omega * sol.l_n
        soy_land_n = (1 - omega) * sol.l_n
        corn_land_cs = omega * sol.l_cs
        soy_land_cs = (1 - omega) * sol.l_cs

        total_corn = corn_land_n + corn_land_cs
        total_soy = soy_land_n + soy_land_cs
        total_n = sol.l_n
        total_cs = sol.l_cs
        total = sol.l_n + sol.l_cs

        push!(df, (
            String(scenario),
            1000 * corn_land_n,
            1000 * corn_land_cs,
            1000 * total_corn,
            1000 * soy_land_n,
            1000 * soy_land_cs,
            1000 * total_soy,
            1000 * total_n,
            1000 * total_cs,
            1000 * total,
            sol.duals.r_land
        ))
    end

    return df
end

# Emissions
function calculate_emissions_detail(solution, params)
    """Calculate detailed emissions by sector"""
    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel,
        :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    FOOD_GOODS = [:corn, :soyoil]

    delta = params.coeff.delta

    # Sector emissions (billion ton CO2e)
    avi_emission = sum(delta[g] * solution.q[g] for g in AVIATION_FUELS)
    road_emission = sum(delta[g] * solution.q[g] for g in ROAD_FUELS)
    food_emission = sum(delta[g] * solution.x[g] for g in FOOD_GOODS)
    total_emission = avi_emission + road_emission + food_emission

    # Detailed fuel-level emissions
    fuel_emissions = Dict(
        g => delta[g] * solution.q[g] for g in union(AVIATION_FUELS, ROAD_FUELS)
    )

    food_emissions = Dict(
        g => delta[g] * solution.x[g] for g in FOOD_GOODS
    )

    return (
        aviation=avi_emission,
        road=road_emission,
        food=food_emission,
        total=total_emission,
        by_fuel=fuel_emissions,
        by_food=food_emissions
    )
end

function make_emissions_table(solutions, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    FOOD_GOODS = [:corn, :soyoil]

    delta = params.coeff.delta

    df = DataFrame(
        Scenario=String[],
        Aviation=Float64[],
        Road=Float64[],
        Food=Float64[],
        Total=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]

        avi_emission = sum(delta[g] * sol.q[g] for g in AVIATION_FUELS)
        road_emission = sum(delta[g] * sol.q[g] for g in ROAD_FUELS)
        food_emission = sum(delta[g] * sol.x[g] for g in FOOD_GOODS)

        push!(df, (
            String(scenario),
            avi_emission * 1000,
            road_emission * 1000,
            food_emission * 1000,
            (avi_emission + road_emission + food_emission) * 1000
        ))
    end

    return df
end

function make_duals_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(
        Scenario=String[],
        λ_rfs=Float64[],
        λ_rfs_avi=Float64[],
        λ_lcfs=Float64[],
        r_land=Float64[],
        λ_blendwall_ethanol=Float64[],
        λ_blendwall_biodiesel=Float64[],
        λ_nonsoy_capacity=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]
        push!(df, (
            String(scenario),
            sol.duals.λ_rfs,
            sol.duals.λ_rfs_avi,
            sol.duals.λ_lcfs,
            sol.duals.r_land,
            sol.duals.λ_blendwall_ethanol,
            sol.duals.λ_blendwall_biodiesel,
            sol.duals.λ_nonsoy_capacity
        ))
    end

    return df
end


# =================================================================================
# RUN ANALYSIS
# =================================================================================

# PART 1: BASE SCENARIOS
println("\n" * "="^130)
println("LOADING BASE SCENARIOS")
println("="^130)

@load "results_base.jld2" results_base policy_configs_base
println("✓ Loaded base results")

display_comparison_tables(
    results_base,
    params,
    policy_configs_base;
    scenarios=[:statusquo, :carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIO RESULTS"
)

# PART 2: TARGET SAF
println("\n" * "="^130)
println("LOADING TARGET SAF ANALYSIS")
println("="^130)

@load "results_target.jld2" equivalent_policies equivalent_solutions target_saf policy_configs_target
println("✓ Loaded target SAF results")

display_comparison_tables(
    equivalent_solutions,
    params,
    policy_configs_target;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="EQUIVALENT POLICY COMPARISON (Target SAF = $(target_saf) billion gallons)",
    show_policy_params=true,  # ⭐ Policy parameters 표시
    equivalent_policies=equivalent_policies  # ⭐ Policy 데이터 전달
)

println("\n" * "="^130)
println("ANALYSIS COMPLETE")
println("="^130)


# =================================================================================
# SAVE RESULTS WITH IMPLICIT TAXES FOR WELFARE ANALYSIS
# =================================================================================

println("\n" * "="^80)
println("SAVING RESULTS WITH IMPLICIT TAXES")
println("="^80)

# =====================
# Base scenarios 분석 데이터 저장
# =====================
implicit_taxes_base = calculate_all_implicit_taxes(results_base, params, policy_configs_base)

results_base_analysis = Dict()
for (scenario, solution) in results_base
    emissions = calculate_emissions_detail(solution, params)

    results_base_analysis[scenario] = merge(
        solution,  # 기존 solution
        (
            implicit_taxes=implicit_taxes_base[scenario],
            emissions=emissions
            # 향후 추가될 수 있는 분석 데이터...
        )
    )
end

@save "results_base_analysis.jld2" results_base_analysis policy_configs_base
println("✓ Saved results_base_analysis.jld2 (with implicit taxes + emissions)")

# =====================
# Target/Equivalent scenarios 분석 데이터 저장
# =====================
implicit_taxes_target = calculate_all_implicit_taxes(equivalent_solutions, params, policy_configs_target)

results_equivalent_analysis = Dict()
for (scenario, solution) in equivalent_solutions
    emissions = calculate_emissions_detail(solution, params)

    results_equivalent_analysis[scenario] = merge(
        solution,
        (
            implicit_taxes=implicit_taxes_target[scenario],
            emissions=emissions
            # 향후 추가될 수 있는 분석 데이터...
        )
    )
end

@save "results_equivalent_analysis.jld2" results_equivalent_analysis policy_configs_target equivalent_policies target_saf
println("✓ Saved results_equivalent_analysis.jld2 (with implicit taxes + emissions)")

println("\n" * "="^80)