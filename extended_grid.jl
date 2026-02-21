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
const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"

# =================================================================================
# 1. Load Status Quo from Base Analysis
# =================================================================================

@load joinpath(OUTPUT_DIR, "results_base_welfare.jld2") status_quo

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
    θ_avi=0:0.001:0.9,
    σ=0.0:0.00027:0.5,
    p=0:0.052:50.0
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

# Plotting
using Plots

# ── ep_3B, ep_6B 먼저 로드 ──────────────────────────────────────────────
@load joinpath(OUTPUT_DIR, "results_equivalent_emissions.jld2") equivalent_emission_policies
ep_3B = equivalent_emission_policies

@load joinpath(OUTPUT_DIR, "results_equivalent_emissions_6.jld2") equivalent_emission_policies
ep_6B = equivalent_emission_policies

# ── vlines 계산 ──────────────────────────────────────────────────────────
statusquo_em = results_extended_analysis.solutions[:statusquo].emissions.total

abatement_3B = statusquo_em - ep_3B[:rfs].actual_emission
abatement_6B = statusquo_em - ep_6B[:rfs].actual_emission

vlines_abatement = [(abatement_3B, "3B"), (abatement_6B, "6B")]

vlines_data = Dict(
    :carbontax => [(ep_3B[:carbontax].policy_value, "3B"), (ep_6B[:carbontax].policy_value, "6B")],
    :rfs => [(ep_3B[:rfs].policy_value, "3B"), (ep_6B[:rfs].policy_value, "6B")],
    :lcfs => [(ep_3B[:lcfs].policy_value, "3B"), (ep_6B[:lcfs].policy_value, "6B")],
    :taxcredit => [(ep_3B[:taxcredit].policy_value, "3B"), (ep_6B[:taxcredit].policy_value, "6B")],
)

# ── MAC plot 함수 정의 (vlines_abatement 버전 하나만 유지) ────────────────
function plot_mac_comparison(results_extended_analysis, mac_extended; vlines_abatement=nothing)
    solutions = results_extended_analysis.solutions
    statusquo_emission = solutions[:statusquo].emissions.total
    max_abatement = 0.15

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

        data_iter = policy_type == :taxcredit ? mac_data[2:end] : mac_data

        for d in data_iter
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

        plot!(p[1], abatement_sorted, mac_private_sorted,
            label=info.label, linewidth=2.5, color=info.color,
            xlabel="Cumulative Abatement (Billion tons CO2)",
            ylabel="MAC (\$/ton CO2)", title="Private MAC",
            legend=:topleft, xlims=(0, max_abatement), ylims=(-100, 1000))

        plot!(p[2], abatement_sorted, mac_social_sorted,
            label=info.label, linewidth=2.5, color=info.color,
            xlabel="Cumulative Abatement (Billion tons CO2)",
            ylabel="MAC (\$/ton CO2)", title="Social MAC",
            legend=:topleft, xlims=(0, max_abatement), ylims=(-500, 1000))
    end

    hline!(p[1], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)
    hline!(p[2], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)

    if !isnothing(vlines_abatement)
        vline_colors = [:darkred, :darkblue]
        for (j, (abatement_val, vlabel)) in enumerate(vlines_abatement)
            c = vline_colors[min(j, length(vline_colors))]
            vline!(p[1], [abatement_val], color=c, linestyle=:dash, linewidth=1.8, label="")
            vline!(p[2], [abatement_val], color=c, linestyle=:dash, linewidth=1.8, label="")
            annotate!(p[1], abatement_val, 650, text(vlabel, c, :center, 9))
            annotate!(p[2], abatement_val, 460, text(vlabel, c, :center, 9))
        end
    end

    return p
end

# ── Generate MAC plot ─────────────────────────────────────────────────────
p_mac_comparison = plot_mac_comparison(results_extended_analysis, mac_extended;
    vlines_abatement=vlines_abatement)

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
function plot_fuel_production(results_df, fuel_config; vlines=nothing)

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

        p_main = plot(
            xlabel="",
            ylabel="$(fuel_config.main_fuel_label)\n(billion gallons)",
            title=title,
            titlefontsize=16, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=(x_min, x_max),
            ylims=get(fuel_config, :main_fuel_ylims, :auto),
            xticks=:none,
            bottom_margin=-1Plots.mm, top_margin=10Plots.mm, left_margin=12Plots.mm
        )
        plot!(p_main, df[!, xcol], df[!, fuel_config.main_fuel], linewidth=2, color=:black)

        p_bio = plot(
            xlabel=xlabel,
            ylabel="$(fuel_config.biofuel_ylabel)\n(billion gallons)",
            legend=false, grid=true,
            xlims=(x_min, x_max),
            top_margin=-1Plots.mm, bottom_margin=45Plots.mm,
            left_margin=12Plots.mm, guidefontsize=11
        )
        for (col, label, color) in fuel_config.biofuel_types
            plot!(p_bio, df[!, xcol], df[!, col], linewidth=2, color=color)
        end

        if !isnothing(vlines) && haskey(vlines, Symbol(policy_type))
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[Symbol(policy_type)])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p_main, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                vline!(p_bio, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
            end
            # annotate는 p_main ylims 확정 후 별도 처리
            cur_ylims = get(fuel_config, :main_fuel_ylims, nothing)
            if !isnothing(cur_ylims)
                for (j, (xval, vlabel)) in enumerate(vlines[Symbol(policy_type)])
                    c = vline_colors[min(j, length(vline_colors))]
                    annotate!(p_main, xval, cur_ylims[2] * 0.95, text(vlabel, c, :center, 9))
                end
            end
        end

        p_combined = plot(p_main, p_bio, layout=grid(2, 1, heights=[0.4, 0.6]), link=:x)
        push!(plots, p_combined)
    end

    p_legend = plot(
        legend=:top, legendcolumns=fuel_config.legendcolumns,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=10, size=(2200, 80)
    )
    plot!(p_legend, [NaN], [NaN], label=fuel_config.main_fuel_label, linewidth=2, color=:black)
    for (col, label, color) in fuel_config.biofuel_types
        plot!(p_legend, [NaN], [NaN], label=label, linewidth=2, color=color)
    end

    final_plot = plot(
        plot(plots..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2500, 1800),
        plot_title=fuel_config.plot_title,
        plot_titlefontsize=22, plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end

# =====================
# Run and Plot the extended grid results
# =====================
@load "results_equivalent_emissions.jld2" equivalent_emission_policies
ep_3B = equivalent_emission_policies

@load "results_equivalent_emissions_6.jld2" equivalent_emission_policies
ep_6B = equivalent_emission_policies

vlines_data = Dict(
    :carbontax => [(ep_3B[:carbontax].policy_value, "3B"), (ep_6B[:carbontax].policy_value, "6B")],
    :rfs => [(ep_3B[:rfs].policy_value, "3B"), (ep_6B[:rfs].policy_value, "6B")],
    :lcfs => [(ep_3B[:lcfs].policy_value, "3B"), (ep_6B[:lcfs].policy_value, "6B")],
    :taxcredit => [(ep_3B[:taxcredit].policy_value, "3B"), (ep_6B[:taxcredit].policy_value, "6B")],
)

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

aviation_plot = plot_fuel_production(results_df, aviation_config; vlines=vlines_data)
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

gasoline_plot = plot_fuel_production(results_df, gasoline_config; vlines=vlines_data)
#savefig(gasoline_plot, gasoline_config.filename)
#display(gasoline_plot)

# Diesel
diesel_config = (
    main_fuel=:q_diesel,
    main_fuel_label="Diesel",
    main_fuel_ylims=(43.2, 45),
    biofuel_types=[
        (:q_rd_soy, "Soy Renewable Diesel", :red),
        (:q_rd_nonsoy, "Non-soy Renewable Diesel", :green),
        (:q_biodiesel_soy, "Soy Biodiesel", :blue),
        (:q_biodiesel_nonsoy, "Non-soy Biodiesel", :purple)
    ],
    biofuel_ylabel="Renewable Diesel & Biodiesel",
    plot_title="Diesel Fuel Production by Policy Stringency",
    filename="diesel_fuel_tight_layout.png",
    legendcolumns=5
)

diesel_plot = plot_fuel_production(results_df, diesel_config; vlines=vlines_data)
function plot_food_production(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    policies = [
        (:carbontax, "Carbon Tax (\$/ton CO2)", "Carbon Tax Policy"),
        (:rfs, "RFS Aviation Mandate (θ_avi)", "RFS Aviation Policy"),
        (:lcfs, "LCFS Standard (σ)", "LCFS Policy"),
        (:taxcredit, "Tax Credit (\$/gal)", "Tax Credit Policy")
    ]

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

    corn_ylim = (5, 8.6)
    soy_oil_ylim = (10, 16)
    soy_meal_ylim = (62, 67)

    plots = []

    for (policy_type, xlabel, title) in policies
        scenario_list = scenario_groups[policy_type]
        sorted_scenarios = sort(scenario_list, by=s -> parse(Int, split(String(s), "_")[2]))

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
            xlabel="", ylabel="Corn Supply\n(billion bushels)",
            title=title, titlefontsize=14, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=(x_vals[1], x_vals[end]), ylims=corn_ylim,
            xticks=:none,
            bottom_margin=-2Plots.mm, top_margin=8Plots.mm,
            left_margin=18Plots.mm, right_margin=25Plots.mm
        )
        plot!(p_corn, x_vals, corn_feedstock, linewidth=2, color=:gold)
        plot!(p_corn, x_vals, total_corn_food, linewidth=2, color=:darkorange)

        # --- Soybean Plot ---
        p_soy = plot(
            xlabel=xlabel, ylabel="Soybean Oil\n(billion lbs)",
            title="", legend=false, grid=true,
            xlims=(x_vals[1], x_vals[end]), ylims=soy_oil_ylim,
            top_margin=-2Plots.mm, bottom_margin=8Plots.mm,
            left_margin=18Plots.mm, right_margin=25Plots.mm,
            guidefontsize=10
        )
        plot!(p_soy, x_vals, soy_oil, linewidth=2, color=:darkgreen)

        p_soy_twin = twinx(p_soy)
        plot!(p_soy_twin, x_vals, soy_meal,
            linewidth=2, color=:brown,
            ylabel="Soybean Meal\n(million metric tons)",
            legend=false, grid=false,
            ylims=soy_meal_ylim, right_margin=25Plots.mm
        )

        # ── 세로선 추가 ───────────────────────────────────────────────────
        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p_corn, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                vline!(p_soy, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p_corn, xval, corn_ylim[2] * 0.95, text(vlabel, c, :center, 9))
            end
        end
        # ─────────────────────────────────────────────────────────────────

        p_combined = plot(p_corn, p_soy,
            layout=grid(2, 1, heights=[0.5, 0.5]),
            link=:x, size=(550, 800)
        )
        push!(plots, p_combined)
    end

    p_legend = plot(
        legend=:top, legendcolumns=4,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=11, size=(2500, 80)
    )
    plot!(p_legend, [NaN], [NaN], label="Corn for food (Feedstock)", linewidth=2, color=:gold)
    plot!(p_legend, [NaN], [NaN], label="Total Corn (+ DDGS)", linewidth=2, color=:darkorange)
    plot!(p_legend, [NaN], [NaN], label="Soybean Oil", linewidth=2, color=:darkgreen)
    plot!(p_legend, [NaN], [NaN], label="Soybean Meal", linewidth=2, color=:brown)

    final_plot = plot(
        plot(plots..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2600, 1800),
        plot_title="Food Production by Policy Stringency",
        plot_titlefontsize=22, plot_titlefontweight=:bold,
        margin=5Plots.mm
    )

    return final_plot
end

# 실행
p_food = plot_food_production(results_extended_analysis; vlines=vlines_data)
display(p_food)
savefig(p_food, "food_production_by_policy.png")

# 결과 저장
@save joinpath(OUTPUT_DIR, "extended_policy_results.jld2") all_results all_solutions results_df

# welfare summary plot by policy type
function plot_welfare_summary_by_policy(results_extended_analysis; vlines=nothing)
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

        plot!(p[idx], grid_values, cs_values,
            label="CS Change", linewidth=2, color=COLOR_CS,
            xlabel=info.xlabel, ylabel="Welfare Change (Billion \$)",
            title=info.label, legend=:best)

        plot!(p[idx], grid_values, ps_values,
            label="PS Change", linewidth=2, color=COLOR_PS)

        if policy_type in [:carbontax, :taxcredit]
            plot!(p[idx], grid_values, gr_values,
                label="Govt Revenue", linewidth=2, color=COLOR_GR)
        end

        plot!(p[idx], grid_values, private_surplus_values,
            label="Private Surplus", linewidth=3, linestyle=:dot, color=COLOR_PRIV)

        plot!(p[idx], grid_values, social_welfare_values,
            label="Social Welfare", linewidth=3, linestyle=:dot, color=COLOR_SOC)

        hline!(p[idx], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)

        # ── 세로선 추가 ───────────────────────────────────────────────────
        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            y_all = [cs_values; ps_values; gr_values; private_surplus_values; social_welfare_values]
            y_top = isempty(y_all) ? 1.0 : maximum(y_all)
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p[idx], [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p[idx], xval, y_top * 0.95, text(vlabel, c, :center, 9))
            end
        end
        # ─────────────────────────────────────────────────────────────────
    end

    return p
end

# Generate plot
p_welfare_summary = plot_welfare_summary_by_policy(results_extended_analysis; vlines=vlines_data)

# emissions plot by policy type
function plot_emissions_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

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

        sorted_scenarios = sort(scenario_list, by=s -> parse(Int, split(String(s), "_")[2]))

        grid_values = Float64[]
        emission_values = Float64[]

        for scenario in sorted_scenarios
            if !haskey(solutions, scenario) || isnothing(solutions[scenario])
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

            push!(emission_values, solutions[scenario].emissions.total)
        end

        info = policy_info[policy_type]

        plot!(p[idx], grid_values, emission_values,
            label="Total Emissions",
            linewidth=2,
            xlabel=info.xlabel,
            ylabel="Emissions (Billion ton CO2e)",
            title=info.label,
            legend=:best,
            color=:red)

        # ── 세로선 추가 ───────────────────────────────────────────────────
        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            y_top = isempty(emission_values) ? 1.0 : maximum(emission_values)
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p[idx], [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p[idx], xval, y_top * 0.98, text(vlabel, c, :center, 9))
            end
        end
        # ─────────────────────────────────────────────────────────────────
    end

    return p
end

# Generate plot
p_emissions = plot_emissions_by_policy(results_extended_analysis; vlines=vlines_data)


# Stacked quantity figure
# Plot fuel production by policy grid with stacked area charts
using Statistics
function plot_fuel_production_stacked(results_df, fuel_config; vlines=nothing)

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
        x_vals = df[!, xcol]

        biofuel_sorted = sort(fuel_config.biofuel_types,
            by=x -> mean(df[!, x[1]]),
            rev=true
        )

        p = plot(
            xlabel=xlabel,
            ylabel="Quantity (billion gallons)",
            title=title,
            titlefontsize=16, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=(x_min, x_max),
            ylims=fuel_config.ylims,
            margin=10Plots.mm, guidefontsize=11
        )

        main_fuel_vals = df[!, fuel_config.main_fuel]
        plot!(p, x_vals, main_fuel_vals,
            fillrange=fuel_config.ylims[1],
            fillalpha=0.7, fillcolor=:lightgray,
            linewidth=1.5, color=:black, label=""
        )

        cumsum_vals = copy(main_fuel_vals)
        for (col, label, color) in biofuel_sorted
            biofuel_vals = df[!, col]
            new_cumsum = cumsum_vals .+ biofuel_vals
            plot!(p, x_vals, new_cumsum,
                fillrange=cumsum_vals,
                fillalpha=0.7, fillcolor=color,
                linewidth=1.5, color=color, label=""
            )
            cumsum_vals = new_cumsum
        end

        # ── 세로선 추가 ───────────────────────────────────────────────────
        if !isnothing(vlines) && haskey(vlines, Symbol(policy_type))
            vline_colors = [:darkred, :darkblue]
            y_top = fuel_config.ylims[2]
            for (j, (xval, vlabel)) in enumerate(vlines[Symbol(policy_type)])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p, xval, y_top * 0.97, text(vlabel, c, :center, 9))
            end
        end
        # ─────────────────────────────────────────────────────────────────

        push!(plots, p)
    end

    p_legend = plot(
        legend=:top, legendcolumns=fuel_config.legendcolumns,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=10, size=(2200, 80)
    )
    plot!(p_legend, [NaN], [NaN],
        label=fuel_config.main_fuel_label,
        linewidth=3, color=:black, fillalpha=0.7, fillcolor=:lightgray
    )
    for (col, label, color) in fuel_config.biofuel_types
        plot!(p_legend, [NaN], [NaN],
            label=label, linewidth=3, color=color,
            fillalpha=0.7, fillcolor=color
        )
    end

    final_plot = plot(
        plot(plots..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2500, 1800),
        plot_title=fuel_config.plot_title,
        plot_titlefontsize=22, plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end

# =====================
# Aviation Fuel Configuration
# =====================
aviation_config = (
    main_fuel=:q_jet_fuel,
    main_fuel_label="Jet Fuel",
    ylims=(0, 22),
    biofuel_types=[
        (:q_saf_atj_conv, "Conventional ATJ-SAF", :blue),
        (:q_saf_atj_cs, "Climate-Smart ATJ-SAF", :red),
        (:q_saf_hefa_conv, "Conventional HEFA-SAF", :green),
        (:q_saf_hefa_cs, "Climate-Smart HEFA-SAF", :orange),
        (:q_saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple)
    ],
    plot_title="Aviation Fuel Production by Policy Stringency",
    filename="aviation_fuel_stacked.png",
    legendcolumns=6
)

aviation_plot = plot_fuel_production_stacked(results_df, aviation_config; vlines=vlines_data)
#savefig(aviation_plot, aviation_config.filename)
#display(aviation_plot)

# =====================
# Road Gasoline Configuration
# =====================
gasoline_config = (
    main_fuel=:q_gasoline,
    main_fuel_label="Gasoline",
    ylims=(120, 150),
    biofuel_types=[
        (:q_ethanol, "Ethanol", :red)
    ],
    plot_title="Road Gasoline Fuel Production by Policy Stringency",
    filename="road_gasoline_fuel_stacked.png",
    legendcolumns=2
)

gasoline_plot = plot_fuel_production_stacked(results_df, gasoline_config; vlines=vlines_data)
#savefig(gasoline_plot, gasoline_config.filename)
#display(gasoline_plot)

# =====================
# Diesel Configuration
# =====================
diesel_config = (
    main_fuel=:q_diesel,
    main_fuel_label="Diesel",
    ylims=(42, 50),
    biofuel_types=[
        (:q_rd_soy, "Soy Renewable Diesel", :red),
        (:q_rd_nonsoy, "Non-soy Renewable Diesel", :green),
        (:q_biodiesel_soy, "Soy Biodiesel", :blue),
        (:q_biodiesel_nonsoy, "Non-soy Biodiesel", :purple)
    ],
    plot_title="Diesel Fuel Production by Policy Stringency",
    filename="diesel_fuel_stacked.png",
    legendcolumns=5
)

diesel_plot = plot_fuel_production_stacked(results_df, diesel_config; vlines=vlines_data)
#savefig(diesel_plot, diesel_config.filename)
#display(diesel_plot)