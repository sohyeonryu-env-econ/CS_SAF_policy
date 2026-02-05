# extended_grid.jl
cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "SAFModel.jl"))
include(joinpath(@__DIR__, "analysis.jl"))  # implicit tax, emissions
include(joinpath(@__DIR__, "welfare.jl"))   # welfare
import .SAFModel: params, build_unified_model, extract_solution,
    FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS
using JLD2
using DataFrames
using Printf
using Plots
using JuMP

# =================================================================================
# 1. Load Status Quo from Base Analysis
# =================================================================================

@load "results_base_complete.jld2" status_quo

println("\nStatus quo emissions (billion ton CO2e):")
println("  Aviation: ", status_quo.emissions.aviation)
println("  Road: ", status_quo.emissions.road)
println("  Food: ", status_quo.emissions.food)
println("  Total: ", status_quo.emissions.total)

# =================================================================================
# 2. Extended Policy Parameter Grid
# =================================================================================

# =====================
# 1) Define and solve extended policy grid scenarios
# =====================
# Define policy grid
const POLICY_RANGES = (
    t=0:0.912:456,
    θ_avi=0:0.001:0.574,
    σ=0.0:0.00027:0.1332,
    p=0:0.052:26.0
)

function create_policy_scenarios()
    scenarios = Dict()

    # 1. Carbon tax scenarios (varying t)
    for t in POLICY_RANGES.t
        t_int = round(Int, t)
        scenarios[Symbol("carbontax_$t_int")] = (t=Float64(t), θ_avi=0.0, σ=0.0, p=0.0)
    end

    # 2. RFS aviation scenarios (varying θ_avi)
    for θ in POLICY_RANGES.θ_avi
        θ_int = round(Int, θ * 1000)
        scenarios[Symbol("rfs_$θ_int")] = (t=0.0, θ_avi=Float64(θ), σ=0.0, p=0.0)
    end

    # 3. LCFS scenarios (varying σ)
    for σ in POLICY_RANGES.σ
        σ_int = round(Int, σ * 1000)
        scenarios[Symbol("lcfs_$σ_int")] = (t=0.0, θ_avi=0.0, σ=Float64(σ), p=0.0)
    end

    # 4. Tax credit scenarios (varying p)
    for p in POLICY_RANGES.p
        p_int = round(Int, p * 100)
        scenarios[Symbol("taxcredit_$p_int")] = (t=0.0, θ_avi=0.0, σ=0.0, p=Float64(p))
    end

    # Add status quo
    scenarios[:statusquo] = (t=0.0, θ_avi=0.0, σ=0.0, p=0.0)
    return scenarios
end

const EXTENDED_POLICY_MATRIX = create_policy_scenarios()

# Run the extended grid scenarios
function run_extended_analysis(params, policy_configs; verbose=true)
    results = Dict()
    solutions = Dict()
    total_scenarios = length(policy_configs)
    current = 0
    solved_count = 0
    failed_count = 0

    for (scenario_name, config) in policy_configs
        current += 1
        try
            model = build_unified_model(params, config)
            optimize!(model)
            if is_solved_and_feasible(model)
                results[scenario_name] = model
                sol = extract_solution(model, scenario_name)
                emissions = calculate_emissions_detail(sol, params)
                sol = merge(sol, (emissions=emissions,))
                if scenario_name != :statusquo
                    implicit_taxes = calculate_implicit_taxes(sol, params, config)
                    sol = merge(sol, (implicit_taxes=implicit_taxes,))
                end
                solutions[scenario_name] = sol
                solved_count += 1
            else
                if verbose
                    println("  ✗ [$current/$total_scenarios] $scenario_name: Failed to solve")
                end
                results[scenario_name] = nothing
                solutions[scenario_name] = nothing
                failed_count += 1
            end
        catch e
            if verbose
                println("  ✗ [$current/$total_scenarios] $scenario_name: Error - $e")
            end
            results[scenario_name] = nothing
            solutions[scenario_name] = nothing
            failed_count += 1
        end
    end

    # 최종 요약만 출력
    if verbose
        println("\n=== Extended Analysis Summary ===")
        @printf("  Solved: %d / %d\n", solved_count, total_scenarios)
        @printf("  Failed: %d / %d\n", failed_count, total_scenarios)
    end

    return results, solutions
end


# =====================
# 2) Calculate full welfare analysis for extended grid
# =====================

function calculate_extended_welfare_analysis(solutions, policy_configs, params; scc=190.0)
    valid_solutions = filter(p -> !isnothing(p.second), solutions)

    if isempty(valid_solutions)
        println("No valid solutions to analyze!")
        return nothing
    end

    status_quo = valid_solutions[:statusquo]

    # Get scenario lists by policy type
    carbontax_scenarios = [k for k in keys(valid_solutions) if startswith(String(k), "carbontax_")]
    rfs_scenarios = [k for k in keys(valid_solutions) if startswith(String(k), "rfs_")]
    lcfs_scenarios = [k for k in keys(valid_solutions) if startswith(String(k), "lcfs_")]
    taxcredit_scenarios = [k for k in keys(valid_solutions) if startswith(String(k), "taxcredit_")]

    # use functions from welfare.jl
    println("Calculating consumer surplus changes...")
    cs_changes = calculate_cs_changes(valid_solutions, status_quo, params)

    println("Calculating producer surplus changes...")
    ps_land_changes = calculate_ps_land_changes(valid_solutions, status_quo, params)

    println("Calculating government revenue changes...")
    gr_changes = calculate_gr_changes(valid_solutions)

    println("Calculating environmental benefits...")
    env_benefits = calculate_environmental_benefit(valid_solutions, status_quo, scc)

    println("Calculating total welfare...")
    welfare_summary = calculate_total_welfare(cs_changes, ps_land_changes,
        gr_changes, env_benefits)

    return (
        solutions=valid_solutions,
        cs_changes=cs_changes,
        ps_land_changes=ps_land_changes,
        gr_changes=gr_changes,
        env_benefits=env_benefits,
        welfare_summary=welfare_summary,
        scenario_groups=(
            carbontax=carbontax_scenarios,
            rfs=rfs_scenarios,
            lcfs=lcfs_scenarios,
            taxcredit=taxcredit_scenarios
        )
    )
end

# =====================
# 3) RUN EXTENDED ANALYSIS
# =====================

# solve
all_results, all_solutions = run_extended_analysis(params, EXTENDED_POLICY_MATRIX, verbose=true);

# welfare
results_extended_analysis = calculate_extended_welfare_analysis(
    all_solutions,
    EXTENDED_POLICY_MATRIX,
    params,
    scc=190.0
);

# =================================================================================
# 3. MARGINAL ABATEMENT COST CALCULATION FOR EXTENDED GRID
# =================================================================================
# calculate
function calculate_mac_extended(results_extended_analysis)
    welfare_summary = results_extended_analysis.welfare_summary
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    mac_results = Dict()

    for (policy_type, scenario_list) in pairs(scenario_groups)
        println("\nCalculating MAC for $(policy_type)...")

        sorted_scenarios = sort(scenario_list, by=s -> begin
            if policy_type == :carbontax
                parse(Int, split(String(s), "_")[2])
            elseif policy_type == :rfs
                parse(Int, split(String(s), "_")[2])
            elseif policy_type == :lcfs
                parse(Int, split(String(s), "_")[2])
            elseif policy_type == :taxcredit
                parse(Int, split(String(s), "_")[2])
            end
        end)

        n = length(sorted_scenarios)
        mac_data = []

        for i in 2:n
            scenario_i = sorted_scenarios[i]
            scenario_i_minus_1 = sorted_scenarios[i-1]

            # Get welfare components
            welfare_i = welfare_summary[scenario_i]
            welfare_i_minus_1 = welfare_summary[scenario_i_minus_1]

            # Get emissions
            emission_i = solutions[scenario_i].emissions.total
            emission_i_minus_1 = solutions[scenario_i_minus_1].emissions.total

            # Calculate changes
            Δemission = emission_i - emission_i_minus_1

            # Skip if no emission change
            if abs(Δemission) < 1e-6
                continue
            end

            # Private surplus and social welfare (already in welfare_summary)
            private_surplus_i = welfare_i.private_surplus
            private_surplus_i_minus_1 = welfare_i_minus_1.private_surplus
            Δprivate_surplus = private_surplus_i - private_surplus_i_minus_1

            social_welfare_i = welfare_i.social_welfare
            social_welfare_i_minus_1 = welfare_i_minus_1.social_welfare
            Δsocial_welfare = social_welfare_i - social_welfare_i_minus_1

            # MAC calculation (negative because we want cost per unit reduction)
            mac_private = Δprivate_surplus / Δemission
            mac_social = Δsocial_welfare / Δemission

            push!(mac_data, (
                scenario=scenario_i,
                scenario_prev=scenario_i_minus_1,
                emission=emission_i,
                emission_prev=emission_i_minus_1,
                Δemission=Δemission,
                private_surplus=private_surplus_i,
                private_surplus_prev=private_surplus_i_minus_1,
                Δprivate_surplus=Δprivate_surplus,
                social_welfare=social_welfare_i,
                social_welfare_prev=social_welfare_i_minus_1,
                Δsocial_welfare=Δsocial_welfare,
                mac_private=mac_private,
                mac_social=mac_social
            ))
        end

        mac_results[policy_type] = mac_data
    end

    return mac_results
end

# Run MAC calculation
mac_extended = calculate_mac_extended(results_extended_analysis)

# Display summary
using Statistics
for (policy_type, mac_data) in pairs(mac_extended)
    println("\n$(uppercase(String(policy_type))):")
    println("  Number of grid points: $(length(mac_data))")
    if !isempty(mac_data)
        macs_private = [d.mac_private for d in mac_data]
        macs_social = [d.mac_social for d in mac_data]
        println("  MAC Private: min=$(minimum(macs_private)), max=$(maximum(macs_private)), mean=$(mean(macs_private))")
        println("  MAC Social: min=$(minimum(macs_social)), max=$(maximum(macs_social)), mean=$(mean(macs_social))")
    end
end

# Plotting
using Plots
function plot_mac_comparison(results_extended_analysis, mac_extended)
    solutions = results_extended_analysis.solutions
    statusquo_emission = solutions[:statusquo].emissions.total

    max_abatement = 0.04  # Billion tons

    p = plot(layout=(1, 2), size=(1600, 600),
        plot_title="Marginal Abatement Cost Curves Comparison",
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    policy_info = Dict(
        :carbontax => (label="Carbon Tax", color=:blue, marker=:circle),
        :rfs => (label="RFS Aviation", color=:red, marker=:diamond),
        :lcfs => (label="LCFS", color=:green, marker=:square),
        :taxcredit => (label="Tax Credit", color=:purple, marker=:utriangle)
    )

    for policy_type in [:carbontax, :rfs, :lcfs, :taxcredit]
        mac_data = mac_extended[policy_type]

        if isempty(mac_data)
            continue
        end

        abatement_values = Float64[]
        mac_private_values = Float64[]
        mac_social_values = Float64[]

        for d in mac_data
            abatement = statusquo_emission - d.emission
            if abatement <= max_abatement
                push!(abatement_values, abatement)
                push!(mac_private_values, d.mac_private)
                push!(mac_social_values, d.mac_social)
            end
        end

        if isempty(abatement_values)
            continue
        end

        sorted_indices = sortperm(abatement_values)
        abatement_sorted = abatement_values[sorted_indices]
        mac_private_sorted = mac_private_values[sorted_indices]
        mac_social_sorted = mac_social_values[sorted_indices]

        info = policy_info[policy_type]

        # Private MAC
        plot!(p[1], abatement_sorted, mac_private_sorted,
            label=info.label,
            #marker=info.marker,
            #markersize=5,
            linewidth=2.5,
            color=info.color,
            xlabel="Cumulative Abatement (Billion tons CO2)",
            ylabel="MAC (\$/ton CO2)",
            title="Private MAC",
            legend=:topleft,
            xlims=(0, max_abatement),
            ylims=(-100, 700))

        # Social MAC
        plot!(p[2], abatement_sorted, mac_social_sorted,
            label=info.label,
            #marker=info.marker,
            #markersize=5,
            linewidth=2.5,
            color=info.color,
            xlabel="Cumulative Abatement (Billion tons CO2)",
            ylabel="MAC (\$/ton CO2)",
            title="Social MAC",
            legend=:topleft,
            xlims=(0, max_abatement),
            ylims=(-500, 500))
    end

    # Add zero line
    hline!(p[1], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)
    hline!(p[2], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)

    return p
end

# Generate plot
p_mac_comparison = plot_mac_comparison(results_extended_analysis, mac_extended)

# plot private and social MAC combined
function plot_mac_combined(results_extended_analysis, mac_extended)
    solutions = results_extended_analysis.solutions
    statusquo_emission = solutions[:statusquo].emissions.total

    max_abatement = 0.1  # Billion tons

    p = plot(size=(1000, 700),
        title="Marginal Abatement Cost Curves",
        xlabel="Cumulative Abatement (Billion tons CO2)",
        ylabel="MAC (\$/ton CO2)",
        legend=:topleft,
        xlims=(0, max_abatement),
        ylims=(-500, 1700),
        left_margin=5Plots.mm,
        bottom_margin=8Plots.mm)  # 아래 여백 증가

    policy_info = Dict(
        :carbontax => (label="Carbon Tax", color=:blue),
        :rfs => (label="RFS Aviation", color=:red),
        :lcfs => (label="LCFS", color=:green),
        :taxcredit => (label="Tax Credit", color=:purple)
    )

    for policy_type in [:carbontax, :rfs, :lcfs, :taxcredit]
        mac_data = mac_extended[policy_type]

        if isempty(mac_data)
            continue
        end

        abatement_values = Float64[]
        mac_private_values = Float64[]
        mac_social_values = Float64[]

        for d in mac_data
            abatement = statusquo_emission - d.emission
            if abatement <= max_abatement
                push!(abatement_values, abatement)
                push!(mac_private_values, d.mac_private)
                push!(mac_social_values, d.mac_social)
            end
        end

        if isempty(abatement_values)
            continue
        end

        sorted_indices = sortperm(abatement_values)
        abatement_sorted = abatement_values[sorted_indices]
        mac_private_sorted = mac_private_values[sorted_indices]
        mac_social_sorted = mac_social_values[sorted_indices]

        info = policy_info[policy_type]

        # Private MAC - 실선
        plot!(p, abatement_sorted, mac_private_sorted,
            label=info.label,  # 정책명만
            linewidth=2.5,
            linestyle=:solid,
            color=info.color)

        # Social MAC - 점선
        plot!(p, abatement_sorted, mac_social_sorted,
            label="",  # label 없음
            linewidth=2.5,
            linestyle=:dash,
            color=info.color)
    end

    # Add zero line
    hline!(p, [0], color=:gray, linestyle=:dot, label="", alpha=0.5)

    # Add note at the bottom
    annotate!(p, 0.15, -400,
        text("Note: Solid lines = Private MAC, Dashed lines = Social MAC",
            :center, 9, :gray))

    return p
end

# Generate plot
p_mac_combined = plot_mac_combined(results_extended_analysis, mac_extended)


# =====================
# Plot the extended grid results
# =====================
# Convert solutions and welfare to comprehensive DataFrame
function results_to_dataframe(extended_analysis, policy_configs)
    solutions = extended_analysis.solutions
    cs_changes = extended_analysis.cs_changes
    ps_land_changes = extended_analysis.ps_land_changes
    gr_changes = extended_analysis.gr_changes
    env_benefits = extended_analysis.env_benefits
    welfare_summary = extended_analysis.welfare_summary

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel,
        :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    SECTORS = [:avi, :gas, :die, :corn, :soyoil, :soymeal]

    results_df = DataFrame(
        scenario=String[],
        policy_type=String[],
        t=Float64[],
        θ_avi=Float64[],
        σ=Float64[],
        p=Float64[]
    )

    # Quantities
    for fuel in vcat(AVIATION_FUELS, ROAD_FUELS)
        results_df[!, Symbol("q_$fuel")] = Float64[]
    end

    # Demand
    for sector in SECTORS
        results_df[!, Symbol("x_$sector")] = Float64[]
    end

    # Prices
    results_df[!, :p_avi] = Float64[]
    results_df[!, :p_gas] = Float64[]
    results_df[!, :p_die] = Float64[]
    results_df[!, :r_land] = Float64[]

    # Emissions (MMT CO2e)
    results_df[!, :emission_avi] = Float64[]
    results_df[!, :emission_road] = Float64[]
    results_df[!, :emission_food] = Float64[]
    results_df[!, :emission_total] = Float64[]

    # ⭐ Welfare components (billion $)
    results_df[!, :cs_change] = Float64[]
    results_df[!, :ps_land_change] = Float64[]
    results_df[!, :gr_change] = Float64[]
    results_df[!, :env_benefit] = Float64[]
    results_df[!, :private_surplus] = Float64[]
    results_df[!, :social_welfare] = Float64[]

    # Fill DataFrame
    for (scenario_name, sol) in solutions
        if isnothing(sol)
            continue
        end

        scenario_str = String(scenario_name)
        config = policy_configs[scenario_name]

        policy_type = if occursin("carbontax", scenario_str)
            "carbontax"
        elseif occursin("rfs", scenario_str)
            "rfs"
        elseif occursin("lcfs", scenario_str)
            "lcfs"
        elseif occursin("taxcredit", scenario_str)
            "taxcredit"
        else
            "statusquo"
        end

        row = [scenario_str, policy_type, config.t, config.θ_avi, config.σ, config.p]

        # Quantities
        for fuel in vcat(AVIATION_FUELS, ROAD_FUELS)
            push!(row, sol.q[fuel])
        end

        # Demand
        for sector in SECTORS
            push!(row, sol.x[sector])
        end

        # Prices
        append!(row, [sol.p_c[:avi], sol.p_c[:gas], sol.p_c[:die], sol.duals.r_land])

        # Emissions
        append!(row, [
            sol.emissions.aviation * 1000,
            sol.emissions.road * 1000,
            sol.emissions.food * 1000,
            sol.emissions.total * 1000
        ])

        # ⭐ Welfare (statusquo는 0)
        if scenario_name == :statusquo
            append!(row, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else
            w = welfare_summary[scenario_name]
            append!(row, [
                w.cs_change,
                w.ps_land_change,
                w.gr_change,
                w.env_benefit,
                w.private_surplus,
                w.social_welfare
            ])
        end

        push!(results_df, row)
    end

    return results_df
end


# Create comprehensive DataFrame
results_df = results_to_dataframe(results_extended_analysis, EXTENDED_POLICY_MATRIX)


# Plot fuel production by policy grid
function plot_fuel_production(results_df, fuel_config)

    policies = [
        ("carbontax", :t, "Carbon Tax (\$ / ton CO2e)", "Carbon Tax Policy", false),
        ("rfs", :θ_avi, "RFS Aviation Mandate (θ_avi)", "RFS Aviation Policy", false),
        ("lcfs", :σ, "LCFS Standard (σ)", "LCFS Policy", true),
        ("taxcredit", :p, "Tax Credit Rate (\$ / gallon)", "Tax Credit Policy", false)
    ]

    plots = []

    for (i, (policy_type, xcol, xlabel, title, reverse_sort)) in enumerate(policies)
        df = filter(row -> row.policy_type == policy_type, results_df)
        sort!(df, xcol, rev=reverse_sort)

        x_min = minimum(df[!, xcol])
        x_max = maximum(df[!, xcol])

        # Main fuel plot
        p_main = plot(
            xlabel="",
            ylabel="$(fuel_config.main_fuel_label)\n(billion gallons)",
            title=title,
            titlefontsize=16,
            titlefontweight=:bold,
            legend=false,
            grid=true,
            xlims=(x_min, x_max),
            ylims=get(fuel_config, :main_fuel_ylims, :auto),
            xticks=:none,
            bottom_margin=-1Plots.mm,
            top_margin=10Plots.mm,
            left_margin=12Plots.mm
        )

        plot!(p_main, df[!, xcol], df[!, fuel_config.main_fuel],
            linewidth=2, color=:black
        )

        # Biofuel plot
        p_bio = plot(
            xlabel=xlabel,
            ylabel="$(fuel_config.biofuel_ylabel)\n(billion gallons)",
            legend=false,
            grid=true,
            xlims=(x_min, x_max),
            top_margin=-1Plots.mm,
            bottom_margin=45Plots.mm,
            left_margin=12Plots.mm,
            guidefontsize=11
        )

        for (col, label, color) in fuel_config.biofuel_types
            plot!(p_bio, df[!, xcol], df[!, col],
                linewidth=2, color=color
            )
        end

        p_combined = plot(p_main, p_bio,
            layout=grid(2, 1, heights=[0.4, 0.6]),
            link=:x
        )

        push!(plots, p_combined)
    end

    # Legend
    p_legend = plot(
        legend=:top,
        legendcolumns=fuel_config.legendcolumns,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1),
        framestyle=:none,
        legendfontsize=10,
        size=(2200, 80)
    )

    plot!(p_legend, [NaN], [NaN],
        label=fuel_config.main_fuel_label,
        linewidth=2, color=:black)

    for (col, label, color) in fuel_config.biofuel_types
        plot!(p_legend, [NaN], [NaN], label=label, linewidth=2, color=color)
    end

    final_plot = plot(
        plot(plots..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2500, 1800),
        plot_title=fuel_config.plot_title,
        plot_titlefontsize=22,
        plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end

# =====================
# Run and Plot the extended grid results
# =====================

# Aviation Fuel
aviation_config = (
    main_fuel=:q_jet_fuel,
    main_fuel_label="Jet Fuel",
    biofuel_types=[
        (:q_saf_atj_conv, "Conventional ATJ-SAF", :blue),
        (:q_saf_atj_cs, "Climate-Smart ATJ-SAF", :red),
        (:q_saf_hefa_conv, "Conventional HEFA-SAF", :green),
        (:q_saf_hefa_cs, "Climate-Smart HEFA-SAF", :orange),
        (:q_saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple)
    ],
    biofuel_ylabel="SAF Production",
    plot_title="Aviation Fuel Production by Policy Stringency",
    filename="aviation_fuel_tight_layout.png",
    legendcolumns=6
)

aviation_plot = plot_fuel_production(results_df, aviation_config)
#savefig(aviation_plot, aviation_config.filename)
#display(aviation_plot)

# Road Gasoline
gasoline_config = (
    main_fuel=:q_gasoline,
    main_fuel_label="Gasoline",
    biofuel_types=[
        (:q_ethanol, "Ethanol", :red)
    ],
    biofuel_ylabel="Ethanol Production",
    plot_title="Road Gasoline Fuel Production by Policy Stringency",
    filename="road_gasoline_fuel_tight_layout.png",
    legendcolumns=2
)

gasoline_plot = plot_fuel_production(results_df, gasoline_config)
#savefig(gasoline_plot, gasoline_config.filename)
#display(gasoline_plot)

# Diesel
diesel_config = (
    main_fuel=:q_diesel,
    main_fuel_label="Diesel",
    main_fuel_ylims=(43.2, 45),
    biofuel_types=[
        (:q_rd_soy, "Soy Renewable Diesel", :red),
        (:q_rd_nonsoy, "Non-soy Renewable Diesel", :orange),
        (:q_biodiesel_soy, "Soy Biodiesel", :blue),
        (:q_biodiesel_nonsoy, "Non-soy Biodiesel", :purple)
    ],
    biofuel_ylabel="Renewable Diesel & Biodiesel",
    plot_title="Diesel Fuel Production by Policy Stringency",
    filename="diesel_fuel_tight_layout.png",
    legendcolumns=5
)

diesel_plot = plot_fuel_production(results_df, diesel_config)

function plot_food_production(results_extended_analysis)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    policies = [
        (:carbontax, "Carbon Tax (\$/ton CO2)", "Carbon Tax Policy"),
        (:rfs, "RFS Aviation Mandate (θ_avi)", "RFS Aviation Policy"),
        (:lcfs, "LCFS Standard (σ)", "LCFS Policy"),
        (:taxcredit, "Tax Credit (\$/gal)", "Tax Credit Policy")
    ]

    # Helper function for policy value
    function get_policy_value(scenario::Symbol, policy_type::Symbol)
        int_val = parse(Int, split(String(scenario), "_")[2])
        if policy_type == :carbontax
            return Float64(int_val)
        elseif policy_type in [:rfs, :lcfs]
            return int_val / 1000.0
        elseif policy_type == :taxcredit
            return int_val / 100.0
        end
    end

    # 고정된 축 범위
    corn_ylim = (5, 8.6)
    soy_oil_ylim = (10, 14.5)
    soy_meal_ylim = (62, 67)

    plots = []

    for (policy_type, xlabel, title) in policies
        scenario_list = scenario_groups[policy_type]
        sorted_scenarios = sort(scenario_list, by=s -> parse(Int, split(String(s), "_")[2]))

        # Extract data
        x_vals = [get_policy_value(s, policy_type) for s in sorted_scenarios]

        total_corn_food = [solutions[s].x[:corn] for s in sorted_scenarios]
        ddgs = [0.092 * solutions[s].q[:ethanol] +
                0.159 * (solutions[s].q[:saf_atj_conv] + solutions[s].q[:saf_atj_cs])
                for s in sorted_scenarios]
        corn_feedstock = total_corn_food .- ddgs
        soy_oil = [solutions[s].x[:soyoil] for s in sorted_scenarios]
        soy_meal = [solutions[s].x[:soymeal] for s in sorted_scenarios]

        # --- Corn Plot ---
        p_corn = plot(
            xlabel="",
            ylabel="Corn Supply\n(billion bushels)",
            title=title,
            titlefontsize=14,
            titlefontweight=:bold,
            legend=false,
            grid=true,
            xlims=(x_vals[1], x_vals[end]),
            ylims=corn_ylim,
            xticks=:none,
            bottom_margin=-2Plots.mm,
            top_margin=8Plots.mm,
            left_margin=18Plots.mm,
            right_margin=25Plots.mm  # ⭐ 오른쪽 여백 추가
        )
        plot!(p_corn, x_vals, corn_feedstock,
            linewidth=2, color=:gold)
        plot!(p_corn, x_vals, total_corn_food,
            linewidth=2, color=:darkorange)

        # --- Soybean Plot (dual axis) ---
        p_soy = plot(
            xlabel=xlabel,
            ylabel="Soybean Oil\n(billion lbs)",
            title="",
            legend=false,
            grid=true,
            xlims=(x_vals[1], x_vals[end]),
            ylims=soy_oil_ylim,
            top_margin=-2Plots.mm,
            bottom_margin=8Plots.mm,
            left_margin=18Plots.mm,
            right_margin=25Plots.mm,  # ⭐ 오른쪽 여백 추가
            guidefontsize=10
        )
        plot!(p_soy, x_vals, soy_oil,
            linewidth=2, color=:darkgreen)

        # Soybean meal (right axis)
        p_soy_twin = twinx(p_soy)
        plot!(p_soy_twin, x_vals, soy_meal,
            linewidth=2, color=:brown,
            ylabel="Soybean Meal\n(million metric tons)",
            legend=false,
            grid=false,
            ylims=soy_meal_ylim,
            right_margin=25Plots.mm  # ⭐ 오른쪽 여백
        )

        # Combine
        p_combined = plot(p_corn, p_soy,
            layout=grid(2, 1, heights=[0.5, 0.5]),
            link=:x,
            size=(550, 800)
        )

        push!(plots, p_combined)
    end

    # Legend plot
    p_legend = plot(
        legend=:top,
        legendcolumns=4,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1),
        framestyle=:none,
        legendfontsize=11,
        size=(2500, 80)
    )
    plot!(p_legend, [NaN], [NaN], label="Corn Feedstock", linewidth=2, color=:gold)
    plot!(p_legend, [NaN], [NaN], label="Total Corn (+ DDGS)", linewidth=2, color=:darkorange)
    plot!(p_legend, [NaN], [NaN], label="Soybean Oil", linewidth=2, color=:darkgreen)
    plot!(p_legend, [NaN], [NaN], label="Soybean Meal", linewidth=2, color=:brown)

    # Final layout
    final_plot = plot(
        plot(plots..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2600, 1800),  # ⭐ 가로 크기 증가
        plot_title="Food Production by Policy Stringency",
        plot_titlefontsize=22,
        plot_titlefontweight=:bold,
        margin=5Plots.mm
    )

    return final_plot
end

# 실행
p_food = plot_food_production(results_extended_analysis)
display(p_food)
savefig(p_food, "food_production_by_policy.png")




# 결과 저장
@save "extended_policy_results.jld2" all_results all_solutions results_df

# welfare summary plot by policy type
function plot_welfare_summary_by_policy(results_extended_analysis)
    welfare_summary = results_extended_analysis.welfare_summary
    scenario_groups = results_extended_analysis.scenario_groups

    COLOR_CS = :steelblue
    COLOR_PS = :orange
    COLOR_GR = :green
    COLOR_PRIV = :purple
    COLOR_SOC = :red

    p = plot(layout=(2, 2), size=(1400, 1000),
        plot_title="Welfare Components by Policy Type",
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    policy_info = Dict(
        :carbontax => (label="Carbon Tax", xlabel="Tax (\$/ton CO2)"),
        :rfs => (label="RFS Aviation", xlabel="Mandate Share"),
        :lcfs => (label="LCFS", xlabel="Carbon Intensity Limit"),
        :taxcredit => (label="Tax Credit", xlabel="Credit (\$/gal)")
    )

    for (idx, policy_type) in enumerate([:carbontax, :rfs, :lcfs, :taxcredit])
        scenario_list = scenario_groups[policy_type]
        if isempty(scenario_list)
            continue
        end

        sorted_scenarios = sort(scenario_list, by=s -> parse(Int, split(String(s), "_")[2]))

        grid_values = Float64[]
        cs_values = Float64[]
        ps_values = Float64[]
        gr_values = Float64[]
        private_surplus_values = Float64[]
        social_welfare_values = Float64[]

        for scenario in sorted_scenarios
            if !haskey(welfare_summary, scenario)
                continue
            end

            config = EXTENDED_POLICY_MATRIX[scenario]
            if policy_type == :carbontax
                push!(grid_values, config.t)
            elseif policy_type == :rfs
                push!(grid_values, config.θ_avi)
            elseif policy_type == :lcfs
                push!(grid_values, config.σ)
            elseif policy_type == :taxcredit
                push!(grid_values, config.p)
            end

            welfare = welfare_summary[scenario]
            push!(cs_values, welfare.cs_change)
            push!(ps_values, welfare.ps_land_change)
            push!(gr_values, welfare.gr_change)
            push!(private_surplus_values, welfare.private_surplus)
            push!(social_welfare_values, welfare.social_welfare)
        end

        info = policy_info[policy_type]

        # CS
        plot!(p[idx], grid_values, cs_values,
            label="CS Change", linewidth=2, color=COLOR_CS,
            xlabel=info.xlabel,
            ylabel="Welfare Change (Billion \$)",
            title=info.label,
            legend=:best)

        # PS
        plot!(p[idx], grid_values, ps_values,
            label="PS Change", linewidth=2, color=COLOR_PS)

        # GR: carbontax, taxcredit
        if policy_type in [:carbontax, :taxcredit]
            plot!(p[idx], grid_values, gr_values,
                label="Govt Revenue", linewidth=2, color=COLOR_GR)
        end

        # Private Surplus
        plot!(p[idx], grid_values, private_surplus_values,
            label="Private Surplus", linewidth=3, linestyle=:dot, color=COLOR_PRIV)

        # Social Welfare
        plot!(p[idx], grid_values, social_welfare_values,
            label="Social Welfare", linewidth=3, linestyle=:dot, color=COLOR_SOC)

        # Zero line
        hline!(p[idx], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)
    end

    return p
end

# Generate plot
p_welfare_summary = plot_welfare_summary_by_policy(results_extended_analysis)

# emissions plot by policy type
function plot_emissions_by_policy(results_extended_analysis)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    # Create 2x2 subplot
    p = plot(layout=(2, 2), size=(1400, 1000),
        plot_title="Emissions by Policy Type",
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    policy_info = Dict(
        :carbontax => (label="Carbon Tax", xlabel="Tax (\$/ton CO2)"),
        :rfs => (label="RFS Aviation", xlabel="Mandate Share"),
        :lcfs => (label="LCFS", xlabel="Carbon Intensity Limit"),
        :taxcredit => (label="Tax Credit", xlabel="Credit (\$/ton)")
    )

    for (idx, policy_type) in enumerate([:carbontax, :rfs, :lcfs, :taxcredit])
        scenario_list = scenario_groups[policy_type]

        if isempty(scenario_list)
            continue
        end

        # Sort scenarios by policy intensity
        sorted_scenarios = sort(scenario_list, by=s -> begin
            parse(Int, split(String(s), "_")[2])
        end)

        grid_values = Float64[]
        emission_values = Float64[]

        for scenario in sorted_scenarios
            if !haskey(solutions, scenario) || isnothing(solutions[scenario])
                continue
            end

            # Extract grid value
            scenario_str = String(scenario)
            config = EXTENDED_POLICY_MATRIX[scenario]
            if policy_type == :carbontax
                push!(grid_values, config.t)
            elseif policy_type == :rfs
                push!(grid_values, config.θ_avi)
            elseif policy_type == :lcfs
                push!(grid_values, config.σ)
            elseif policy_type == :taxcredit
                push!(grid_values, config.p)
            end

            # Get emission
            push!(emission_values, solutions[scenario].emissions.total)
        end

        info = policy_info[policy_type]

        # Plot emissions
        plot!(p[idx], grid_values, emission_values,
            label="Total Emissions",
            #marker=:circle,
            linewidth=2,
            xlabel=info.xlabel,
            ylabel="Emissions (tons CO2)",
            title=info.label,
            legend=:best,
            color=:red)
    end

    return p
end

# Generate plot
p_emissions = plot_emissions_by_policy(results_extended_analysis)





# extended_grid.jl에 추가하여 실행

# 1. 특정 시나리오의 상세 비용 분석
function diagnose_diesel_fuels(scenario_name, solution, params, config)
    println("\n" * "="^80)
    println("DIESEL FUEL COST ANALYSIS: $scenario_name")
    println("="^80)

    sol = solution
    coeff = params.coeff
    supply = params.supply
    fuel_cost = supply.fuel

    # RFS dual
    λ_rfs = sol.duals.λ_rfs

    # 생산량
    q_rd_soy = sol.q[:rd_soy]
    q_rd_nonsoy = sol.q[:rd_nonsoy]
    q_bd_soy = sol.q[:biodiesel_soy]
    q_bd_nonsoy = sol.q[:biodiesel_nonsoy]

    total_rd = q_rd_soy + q_rd_nonsoy
    total_bd = q_bd_soy + q_bd_nonsoy
    total_hefa = total_rd + sol.q[:saf_hefa_conv] + sol.q[:saf_hefa_cs] + sol.q[:saf_hefa_nonsoy]

    println("\nProduction:")
    println("  RD (soy):        $(round(q_rd_soy, digits=4))")
    println("  RD (nonsoy):     $(round(q_rd_nonsoy, digits=4))")
    println("  BD (soy):        $(round(q_bd_soy, digits=4))")
    println("  BD (nonsoy):     $(round(q_bd_nonsoy, digits=4))")
    println("  Total HEFA:      $(round(total_hefa, digits=4))")

    # HEFA 공정 비용
    v_hefa = fuel_cost[:saf_hefa_shared].v
    c0_hefa = fuel_cost[:saf_hefa_shared].c0
    c2_hefa = fuel_cost[:saf_hefa_shared].c2

    process_mc_hefa = c0_hefa + c2_hefa * max(0, total_hefa - v_hefa)^2

    println("\nHEFA Process Cost:")
    println("  v (kink point):  $(round(v_hefa, digits=4))")
    println("  c0:              $(round(c0_hefa, digits=4))")
    println("  c2:              $(round(c2_hefa, digits=4))")
    println("  Total HEFA:      $(round(total_hefa, digits=4))")
    println("  Over kink:       $(round(max(0, total_hefa - v_hefa), digits=4))")
    println("  MC_hefa:         $(round(process_mc_hefa, digits=4))")

    # Biodiesel 비용
    v_bd = fuel_cost[:biodiesel_soy].v
    c0_bd = fuel_cost[:biodiesel_soy].c0
    c2_bd = fuel_cost[:biodiesel_soy].c2

    mc_bd_soy = c0_bd + c2_bd * max(0, q_bd_soy - v_bd)^2
    mc_bd_nonsoy = c0_bd + c2_bd * max(0, q_bd_nonsoy - v_bd)^2

    println("\nBiodiesel Cost:")
    println("  v (kink point):  $(round(v_bd, digits=4))")
    println("  c0:              $(round(c0_bd, digits=4))")
    println("  c2:              $(round(c2_bd, digits=4))")
    println("  MC_bd_soy:       $(round(mc_bd_soy, digits=4))")
    println("  MC_bd_nonsoy:    $(round(mc_bd_nonsoy, digits=4))")

    # 피드스톡 가격
    p_soy = sol.p_f[:feedstock_soy_n]
    nonsoy_price = coeff.nonsoy_feedstock_price
    alpha_soy = coeff.alpha[:rd_soy]

    println("\nFeedstock:")
    println("  Soy price:       $(round(p_soy, digits=4)) \$/lb")
    println("  Non-soy price:   $(round(nonsoy_price, digits=4)) \$/lb")
    println("  Alpha (lb/gal):  $(round(alpha_soy, digits=4))")

    # Total cost 계산
    hefa_premium = coeff.hefa_saf_premium

    total_cost_rd_soy = process_mc_hefa - hefa_premium + alpha_soy * p_soy - 1.7 * λ_rfs
    total_cost_rd_nonsoy = process_mc_hefa - hefa_premium + alpha_soy * nonsoy_price - 1.7 * λ_rfs
    total_cost_bd_soy = mc_bd_soy + alpha_soy * p_soy - 1.5 * λ_rfs
    total_cost_bd_nonsoy = mc_bd_nonsoy + alpha_soy * nonsoy_price - 1.5 * λ_rfs

    println("\nNet Cost (MC + Feedstock - RFS Credit):")
    println("  RD soy:          $(round(total_cost_rd_soy, digits=4))")
    println("  RD nonsoy:       $(round(total_cost_rd_nonsoy, digits=4))")
    println("  BD soy:          $(round(total_cost_bd_soy, digits=4))")
    println("  BD nonsoy:       $(round(total_cost_bd_nonsoy, digits=4))")

    println("\nCost Comparison:")
    println("  BD_soy - RD_soy:     $(round(total_cost_bd_soy - total_cost_rd_soy, digits=4))")
    println("  BD_nonsoy - RD_nonsoy: $(round(total_cost_bd_nonsoy - total_cost_rd_nonsoy, digits=4))")

    println("\nλ_rfs = $(round(λ_rfs, digits=4))")
    println("="^80)
end

# 2. 여러 시나리오 비교
scenarios_to_check = [:statusquo, :carbontax_100, :carbontax_200, :rfs_100, :lcfs_50, :taxcredit_500]

for scenario in scenarios_to_check
    if haskey(all_solutions, scenario) && !isnothing(all_solutions[scenario])
        config = EXTENDED_POLICY_MATRIX[scenario]
        diagnose_diesel_fuels(scenario, all_solutions[scenario], params, config)
    end
end

function diagnose_spillover_effect(scenario_name, solution, params)
    println("\n" * "="^80)
    println("SPILLOVER ANALYSIS: $scenario_name")
    println("="^80)

    sol = solution

    # Aviation sector
    total_saf = sum(sol.q[g] for g in [:saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy])
    jet_fuel = sol.q[:jet_fuel]
    saf_share = total_saf / (total_saf + jet_fuel)

    # Road sector  
    total_rd = sol.q[:rd_soy] + sol.q[:rd_nonsoy]
    total_bd = sol.q[:biodiesel_soy] + sol.q[:biodiesel_nonsoy]
    diesel = sol.q[:diesel]
    road_biofuel_share = (total_rd + total_bd) / (total_rd + total_bd + diesel)

    # Feedstock
    p_soy = sol.p_f[:feedstock_soy_n]

    # Total soy to aviation vs road
    soy_to_avi = (sol.q[:saf_hefa_conv] + sol.q[:saf_hefa_cs]) * 8.0  # alpha
    soy_to_road = (sol.q[:rd_soy] + sol.q[:biodiesel_soy]) * 7.55

    println("\nAviation Sector:")
    println("  SAF: $(round(total_saf, digits=3)) billion gal")
    println("  Jet fuel: $(round(jet_fuel, digits=3)) billion gal")
    println("  SAF share: $(round(saf_share*100, digits=2))%")

    println("\nRoad Diesel Sector:")
    println("  RD: $(round(total_rd, digits=3)) billion gal")
    println("  BD: $(round(total_bd, digits=3)) billion gal")
    println("  Fossil diesel: $(round(diesel, digits=3)) billion gal")
    println("  Biofuel share: $(round(road_biofuel_share*100, digits=2))%")

    println("\nFeedstock Allocation:")
    println("  Soy price: $(round(p_soy, digits=3)) \$/lb")
    println("  Soy to aviation: $(round(soy_to_avi, digits=3)) billion lbs")
    println("  Soy to road: $(round(soy_to_road, digits=3)) billion lbs")
    println("  Aviation share: $(round(soy_to_avi/(soy_to_avi+soy_to_road)*100, digits=2))%")

    println("="^80)
end

# Compare scenarios
diagnose_spillover_effect(:statusquo, all_solutions[:statusquo], params)
diagnose_spillover_effect(:rfs_100, all_solutions[:rfs_100], params)
diagnose_spillover_effect(:rfs_300, all_solutions[:rfs_300], params)
diagnose_spillover_effect(:lcfs_50, all_solutions[:lcfs_50], params)


# =================================================================================
# Non-soy Capacity Constraint Check
# =================================================================================

function analyze_nonsoy_bottleneck(all_solutions, params)
    # 1. 시나리오별 Non-soy 사용량 계산
    alpha_val = params.coeff.alpha[:rd_nonsoy] # 7.55
    capacity_limit = 30.0

    analysis_data = []

    for (name, sol) in all_solutions
        isnothing(sol) && continue

        # Non-soy 연료들 합계 (원료 기준)
        usage = alpha_val * (
            sol.q[:biodiesel_nonsoy] +
            sol.q[:rd_nonsoy] +
            sol.q[:saf_hefa_nonsoy]
        )

        # Dual variable (Shadow price) 추출
        # PATHSolver 결과에서 λ_nonsoy_capacity의 값을 가져옵니다.
        shadow_price = sol.duals.λ_nonsoy_capacity

        push!(analysis_data, (
            scenario=name,
            policy=occursin("carbontax", String(name)) ? "CarbonTax" :
                   occursin("rfs", String(name)) ? "RFS" :
                   occursin("lcfs", String(name)) ? "LCFS" :
                   occursin("taxcredit", String(name)) ? "TaxCredit" : "StatusQuo",
            usage=usage,
            slack=capacity_limit - usage,
            shadow_price=shadow_price
        ))
    end

    df_check = DataFrame(analysis_data)

    # 2. 결과 요약 출력
    println("\n=== Non-soy Capacity Analysis (Limit: $capacity_limit) ===")

    # 용량의 99.9% 이상을 사용 중인 시나리오 필터링
    bottlenecked = filter(row -> row.usage >= capacity_limit * 0.999, df_check)

    println("Total scenarios checked: ", nrow(df_check))
    println("Scenarios hitting capacity: ", nrow(bottlenecked))

    if nrow(bottlenecked) > 0
        println("\nMax Shadow Price (Value of additional 1 unit of Non-soy):")
        # 정책별로 가장 병목이 심한(shadow price가 높은) 지점 출력
        for p in unique(df_check.policy)
            p_df = filter(row -> row.policy == p, bottlenecked)
            if !isempty(p_df)
                max_row = p_df[argmax(p_df.shadow_price), :]
                @printf("  %-10s: Max Shadow Price = %.4f (at %s)\n", p, max_row.shadow_price, max_row.scenario)
            end
        end
    end

    # 3. 시각화 (Shadow Price가 0보다 큰 구간 확인)
    p_check = plot(layout=(2, 1), size=(1000, 800))

    # Usage plot
    scatter!(p_check[1], df_check.usage, group=df_check.policy,
        title="Non-soy Feedstock Total Usage", ylabel="Usage (Million Units)",
        legend=:outerright)
    hline!(p_check[1], [30.0], color=:red, linestyle=:dash, label="Capacity Limit")

    # Shadow Price plot
    scatter!(p_check[2], df_check.shadow_price, group=df_check.policy,
        title="Shadow Price (λ_nonsoy_capacity)", ylabel="Price (\$)",
        legend=:outerright)

    return df_check, p_check
end

# 실행
df_nonsoy, p_nonsoy = analyze_nonsoy_bottleneck(all_solutions, params)
display(p_nonsoy)

# 구체적으로 어떤 연료가 Non-soy를 점유하는지 보고 싶을 때 (예: LCFS 강한 구간)
function check_composition(scenario_name, all_solutions)
    sol = all_solutions[scenario_name]
    println("\nComposition of $scenario_name:")
    println("  Non-soy BD: ", sol.q[:biodiesel_nonsoy])
    println("  Non-soy RD: ", sol.q[:rd_nonsoy])
    println("  Non-soy SAF: ", sol.q[:saf_hefa_nonsoy])
end