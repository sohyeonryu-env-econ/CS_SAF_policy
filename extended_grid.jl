include(joinpath(@__DIR__, "SAFModel.jl"))
include(joinpath(@__DIR__, "analysis.jl"))  # ⭐ implicit tax 함수들
include(joinpath(@__DIR__, "welfare.jl"))   # ⭐ welfare 함수들
import .SAFModel: params, build_unified_model, extract_solution,
    FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS
using JLD2
using DataFrames
using Printf
using Plots

cd(@__DIR__)
println("Working directory: ", pwd())

# =================================================================================
# 5. Extended Policy Parameter Grid
# =================================================================================

# Define policy ranges
const POLICY_RANGES = (
    t=0:100:1500,           # Carbon tax: 0, 10, 20, ..., 400
    θ_avi=0:0.05:1.0,     # RFS aviation: 0, 0.05, 0.1, ..., 0.5
    σ=0.0:0.05:1.0,      # LCFS: 1.0, 0.99, 0.98, ..., 0.5
    p=0:15:450             # Tax credit: 0, 1, 2, ..., 100
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

function run_all_scenarios(params; verbose=true)
    results = Dict()
    solutions = Dict()

    total_scenarios = length(EXTENDED_POLICY_MATRIX)
    current = 0

    for (scenario_name, config) in EXTENDED_POLICY_MATRIX
        current += 1

        if verbose
            println("\n[$current/$total_scenarios] Running $scenario_name...")
        end

        try
            model = build_unified_model(params, config)
            optimize!(model)

            if is_solved_and_feasible(model)
                results[scenario_name] = model
                solutions[scenario_name] = extract_solution(model, scenario_name)

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

# Run all scenarios
println("\nRunning extended policy analysis...")
all_results, all_solutions = run_all_scenarios(params, verbose=true)


# =====================
# Plot the extended grid results
# =====================

# Convert solutions to DataFrame
function solutions_to_dataframe(all_solutions, params)

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    FOOD_GOODS = [:corn, :soyoil]
    delta = params.coeff.delta

    results_df = DataFrame(
        scenario=String[],
        policy_type=String[],
        t=Float64[],
        θ_avi=Float64[],
        σ=Float64[],
        p=Float64[]
    )

    # Add fuel quantity columns
    for fuel in AVIATION_FUELS
        results_df[!, Symbol("q_$fuel")] = Float64[]
    end

    for fuel in ROAD_FUELS
        results_df[!, Symbol("q_$fuel")] = Float64[]
    end

    # Add demand columns
    for sector in [:avi, :gas, :die, :corn, :soyoil, :soymeal]
        results_df[!, Symbol("x_$sector")] = Float64[]
    end

    # Add price columns
    results_df[!, :p_avi] = Float64[]
    results_df[!, :p_gas] = Float64[]
    results_df[!, :p_die] = Float64[]
    results_df[!, :r_land] = Float64[]

    # Add emission columns
    results_df[!, :emission_avi] = Float64[]
    results_df[!, :emission_road] = Float64[]
    results_df[!, :emission_food] = Float64[]
    results_df[!, :emission_total] = Float64[]

    # Fill DataFrame
    for (scenario_name, sol) in all_solutions
        if isnothing(sol)
            continue
        end

        scenario_str = String(scenario_name)
        config = EXTENDED_POLICY_MATRIX[scenario_name]

        # Determine policy type
        if occursin("carbontax", scenario_str)
            policy_type = "carbontax"
        elseif occursin("rfs", scenario_str)
            policy_type = "rfs"
        elseif occursin("lcfs", scenario_str)
            policy_type = "lcfs"
        elseif occursin("taxcredit", scenario_str)
            policy_type = "taxcredit"
        else
            policy_type = "statusquo"
        end

        row = [
            scenario_str,
            policy_type,
            config.t,
            config.θ_avi,
            config.σ,
            config.p
        ]

        # Aviation fuels
        for fuel in AVIATION_FUELS
            push!(row, sol.q[fuel])
        end

        # Road fuels
        for fuel in ROAD_FUELS
            push!(row, sol.q[fuel])
        end

        # Demand
        for sector in [:avi, :gas, :die, :corn, :soyoil, :soymeal]
            push!(row, sol.x[sector])
        end

        # Prices
        push!(row, sol.p_c[:avi])
        push!(row, sol.p_c[:gas])
        push!(row, sol.p_c[:die])
        push!(row, sol.duals.r_land)

        # Emissions
        avi_emission = sum(delta[g] * sol.q[g] for g in AVIATION_FUELS)
        road_emission = sum(delta[g] * sol.q[g] for g in ROAD_FUELS)
        food_emission = sum(delta[g] * sol.x[g] for g in FOOD_GOODS)

        push!(row, avi_emission * 1000)
        push!(row, road_emission * 1000)
        push!(row, food_emission * 1000)
        push!(row, (avi_emission + road_emission + food_emission) * 1000)

        push!(results_df, row)
    end

    return results_df
end

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
println("\nRunning extended policy analysis...")
all_results, all_solutions = run_all_scenarios(params, verbose=true)

results_df = solutions_to_dataframe(all_solutions, params)

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