include(joinpath(@__DIR__, "SAFModel.jl"))
include(joinpath(@__DIR__, "analysis.jl"))  # ⭐ implicit tax, emissions
include(joinpath(@__DIR__, "welfare.jl"))   # ⭐ welfare
import .SAFModel: params, build_unified_model, extract_solution,
    FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS
using JLD2
using DataFrames
using Printf
using Plots

cd(@__DIR__)
println("Working directory: ", pwd())

# =================================================================================
# LOAD STATUS QUO FROM COMPLETE BASE ANALYSIS
# =================================================================================
println("\n" * "="^80)
println("LOADING STATUS QUO FROM BASE ANALYSIS")
println("="^80)

@load "results_base_complete.jld2" status_quo
println("✓ Loaded status quo from results_base_complete.jld2")

# Validate status quo
println("\nStatus quo validation:")
println("  Has q (quantities): ", haskey(status_quo, :q))
println("  Has x (demand): ", haskey(status_quo, :x))
println("  Has emissions: ", haskey(status_quo, :emissions))
println("  Has duals: ", haskey(status_quo, :duals))

if !haskey(status_quo, :emissions)
    error("❌ Status quo missing emissions! Please run analysis.jl and welfare.jl first.")
end

println("\nStatus quo emissions (billion ton CO2e):")
println("  Aviation: ", status_quo.emissions.aviation)
println("  Road: ", status_quo.emissions.road)
println("  Food: ", status_quo.emissions.food)
println("  Total: ", status_quo.emissions.total)

println("="^80)

# =================================================================================
# 5. Extended Policy Parameter Grid
# =================================================================================

# Define policy grid
const POLICY_RANGES = (
    t=0:75:1500,           # Carbon tax: 0, 10, 20, ..., 400
    θ_avi=0:0.05:1.0,     # RFS aviation: 0, 0.05, 0.1, ..., 0.5
    σ=0.0:0.05:1.0,      # LCFS: 1.0, 0.99, 0.98, ..., 0.5
    p=0:22:450             # Tax credit: 0, 1, 2, ..., 100
)

# Create policy scenarios based on which parameter varies
function create_policy_scenarios()
    scenarios = Dict()

    # 1. Carbon tax scenarios (varying t)
    for t in POLICY_RANGES.t
        scenarios[Symbol("carbontax_$(Int(t))")] = (t=Float64(t), θ_avi=0.0, σ=0.0, p=0.0)
    end

    # 2. RFS aviation scenarios (varying θ_avi)
    for θ in POLICY_RANGES.θ_avi
        # Use round to avoid floating point precision issues
        θ_int = round(Int, θ * 100)
        scenarios[Symbol("rfs_$θ_int")] = (t=0.0, θ_avi=Float64(θ), σ=0.0, p=0.0)
    end

    # 3. LCFS scenarios (varying σ)
    for σ in POLICY_RANGES.σ
        # Use round to avoid floating point precision issues
        σ_int = round(Int, σ * 100)
        scenarios[Symbol("lcfs_$σ_int")] = (t=0.0, θ_avi=0.0, σ=Float64(σ), p=0.0)
    end

    # 4. Tax credit scenarios (varying p)
    for p in POLICY_RANGES.p
        scenarios[Symbol("taxcredit_$(Int(p))")] = (t=0.0, θ_avi=0.0, σ=0.0, p=Float64(p))
    end

    # Add status quo
    scenarios[:statusquo] = (t=0.0, θ_avi=0.0, σ=0.0, p=0.0)
    return scenarios
end

const EXTENDED_POLICY_MATRIX = create_policy_scenarios()

# =====================
# Run the extended grid scenarios
# =====================

function run_extended_analysis(params, policy_configs; verbose=true)
    results = Dict()
    solutions = Dict()
    total_scenarios = length(policy_configs)
    current = 0

    for (scenario_name, config) in policy_configs
        current += 1
        if verbose
            println("\n[$current/$total_scenarios] Running $scenario_name...")
        end

        try
            model = build_unified_model(params, config)
            optimize!(model)

            if is_solved_and_feasible(model)
                results[scenario_name] = model

                # Extract base solution
                sol = extract_solution(model, scenario_name)

                # ⭐ analysis.jl의 함수 사용해서 emissions 추가
                emissions = calculate_emissions_detail(sol, params)
                sol = merge(sol, (emissions=emissions,))

                # ⭐ analysis.jl의 함수 사용해서 implicit_taxes 추가 (statusquo 제외)
                if scenario_name != :statusquo
                    implicit_taxes = calculate_implicit_taxes(sol, params, config)
                    sol = merge(sol, (implicit_taxes=implicit_taxes,))
                end

                solutions[scenario_name] = sol

                if verbose
                    println("  ✓ Solved successfully")
                end
            else
                if verbose
                    println("  ✗ Failed to solve")
                end
                results[scenario_name] = nothing
                solutions[scenario_name] = nothing
            end
        catch e
            if verbose
                println("  ✗ Error: $e")
            end
            results[scenario_name] = nothing
            solutions[scenario_name] = nothing
        end
    end

    return results, solutions
end

# =====================
# Calculate full welfare analysis for extended grid
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

    # ⭐ welfare.jl의 함수들 사용 (모두 solution.emissions 필요)
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
# RUN EXTENDED ANALYSIS
# =====================
println("\n" * "="^80)
println("RUNNING EXTENDED POLICY GRID ANALYSIS")
println("="^80)

all_results, all_solutions = run_extended_analysis(params, EXTENDED_POLICY_MATRIX, verbose=true)

println("\n" * "="^80)
println("CALCULATING WELFARE ANALYSIS FOR EXTENDED GRID")
println("="^80)

results_extended_analysis = calculate_extended_welfare_analysis(
    all_solutions,
    EXTENDED_POLICY_MATRIX,
    params,
    scc=190.0
)


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
println("\n" * "="^80)
println("MARGINAL ABATEMENT COST SUMMARY")
println("="^80)

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

using Plots




# =====================
# SAVE RESULTS
# =====================
println("\nSaving extended analysis results...")
@save "results_extended_analysis.jld2" results_extended_analysis EXTENDED_POLICY_MATRIX

println("\n✓ Extended analysis complete!")
println("Results saved to: results_extended_analysis.jld2")


function plot_mac_comparison(results_extended_analysis, mac_extended)
    solutions = results_extended_analysis.solutions
    statusquo_emission = solutions[:statusquo].emissions.total

    max_abatement = 0.3  # Billion tons

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
            ylims=(-500, 2000))

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
            ylims=(-500, 2000))
    end

    # Add zero line
    hline!(p[1], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)
    hline!(p[2], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)

    return p
end

# Generate plot
p_mac_comparison = plot_mac_comparison(results_extended_analysis, mac_extended)
display(p_mac_comparison)

using Plots

function plot_mac_combined(results_extended_analysis, mac_extended)
    solutions = results_extended_analysis.solutions
    statusquo_emission = solutions[:statusquo].emissions.total

    max_abatement = 0.3  # Billion tons

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
display(p_mac_combined)

# Save plot
savefig(p_mac_combined, "mac_combined.png")
println("\nPlot saved as mac_combined.png")


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
println("\nCreating results DataFrame...")
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
            xticks=:none,
            bottom_margin=-1Plots.mm,
            top_margin=10Plots.mm,
            left_margin=12Plots.mm
        )

        plot!(p_main, df[!, xcol], df[!, fuel_config.main_fuel],
            linewidth=2, marker=:circle, markersize=3, color=:black, markerstrokewidth=0
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
                linewidth=2, marker=:circle, markersize=3, color=color, markerstrokewidth=0
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
        linewidth=2, color=:black, marker=:circle)

    for (col, label, color) in fuel_config.biofuel_types
        plot!(p_legend, [NaN], [NaN], label=label, linewidth=2, color=color, marker=:circle)
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
#println("\nRunning extended policy analysis...")
#all_results, all_solutions = run_all_scenarios(params, verbose=true)

#results_df = solutions_to_dataframe(all_solutions, params)

println("\nGenerating figures...")

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
#savefig(diesel_plot, diesel_config.filename)
#display(diesel_plot)

#println("\nFigures saved!")

# 결과 저장
@save "extended_policy_results.jld2" all_results all_solutions results_df
println("\n✓ Extended policy results saved to extended_policy_results.jld2")


function plot_welfare_summary_by_policy(results_extended_analysis)
    welfare_summary = results_extended_analysis.welfare_summary
    scenario_groups = results_extended_analysis.scenario_groups

    # Create 2x2 subplot
    p = plot(layout=(2, 2), size=(1400, 1000),
        plot_title="Welfare Components by Policy Type",
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
        cs_values = Float64[]
        ps_values = Float64[]
        private_surplus_values = Float64[]
        social_welfare_values = Float64[]

        for scenario in sorted_scenarios
            if !haskey(welfare_summary, scenario)
                continue
            end

            # Extract grid value
            scenario_str = String(scenario)
            grid_val = parse(Int, split(scenario_str, "_")[2])

            # Adjust for RFS and LCFS (they use 0-100 scale)
            if policy_type in [:rfs, :lcfs]
                push!(grid_values, Float64(grid_val) / 100.0)
            else
                push!(grid_values, Float64(grid_val))
            end

            # Get welfare values
            welfare = welfare_summary[scenario]
            push!(cs_values, welfare.cs_change)
            push!(ps_values, welfare.ps_land_change)
            push!(private_surplus_values, welfare.private_surplus)
            push!(social_welfare_values, welfare.social_welfare)
        end

        info = policy_info[policy_type]

        # Plot CS
        plot!(p[idx], grid_values, cs_values,
            label="CS Change",
            marker=:circle,
            linewidth=2,
            xlabel=info.xlabel,
            ylabel="Welfare Change (Million \$)",
            title=info.label,
            legend=:best)

        # Plot PS
        plot!(p[idx], grid_values, ps_values,
            label="PS Change",
            marker=:diamond,
            linewidth=2)

        # Plot Private Surplus
        plot!(p[idx], grid_values, private_surplus_values,
            label="Private Surplus",
            marker=:square,
            linewidth=3,
            linestyle=:dash)

        # Plot Social Welfare
        plot!(p[idx], grid_values, social_welfare_values,
            label="Social Welfare",
            marker=:hexagon,
            linewidth=3,
            linestyle=:dashdot,
            color=:red)

        # Add zero line
        hline!(p[idx], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)
    end

    return p
end

# Generate plot
p_welfare_summary = plot_welfare_summary_by_policy(results_extended_analysis)

using Plots

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
            grid_val = parse(Int, split(scenario_str, "_")[2])

            # Adjust for RFS and LCFS (they use 0-100 scale)
            if policy_type in [:rfs, :lcfs]
                push!(grid_values, Float64(grid_val) / 100.0)
            else
                push!(grid_values, Float64(grid_val))
            end

            # Get emission
            push!(emission_values, solutions[scenario].emissions.total)
        end

        info = policy_info[policy_type]

        # Plot emissions
        plot!(p[idx], grid_values, emission_values,
            label="Total Emissions",
            marker=:circle,
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
display(p_emissions)

# Save plot
savefig(p_emissions, "emissions_by_policy.png")
println("\nPlot saved as emissions_by_policy.png")