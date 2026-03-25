# extended_grid.jl
cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "SAFModel.jl"))
include(joinpath(@__DIR__, "analysis.jl"))   # load analysis.jl from the same directory
import .SAFModel: params, build_unified_model, extract_solution,
    FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS
import .SAFAnalysis: calculate_emissions_detail, calculate_implicit_taxes,
    calculate_cs_changes, calculate_ps_land_changes,
    calculate_gr_changes, calculate_environmental_benefit,
    calculate_total_welfare
using JLD2
using DataFrames
using Printf
using Plots
using JuMP
const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"

# =================================================================================
# 1. Load Status Quo from Base Analysis
# =================================================================================

@load joinpath(OUTPUT_DIR, "results_base.jld2") results_base policy_configs_base welfare_base
status_quo = results_base[:statusquo]

# =================================================================================
# 2. Extended Policy Parameter Grid
# =================================================================================

# =====================
# 1) Define and solve extended policy grid scenarios
# =====================
# Define policy grid

const POLICY_RANGES = (
    t=0:0.1:500,
    θ_avi=0:0.001:0.9,
    σ=0.0:0.0003:0.5,
    p=0:0.05:100.0
)

function create_policy_scenarios()
    scenarios = Dict()

    # 1. Carbon tax scenarios (varying t)
    for t in POLICY_RANGES.t
        t_int = round(Int, t)
        scenarios[Symbol("carbontax_$t_int")] = (t=Float64(t), θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation)
    end

    # 2. RFS aviation scenarios (varying θ_avi)
    for θ in POLICY_RANGES.θ_avi
        θ_int = round(Int, θ * 1000)
        scenarios[Symbol("rfs_$θ_int")] = (t=0.0, θ_avi=Float64(θ), σ=0.0, p=0.0, carbon_tax_scope=:aviation)
    end

    # 3. LCFS scenarios (varying σ)
    for σ in POLICY_RANGES.σ
        σ_int = round(Int, σ * 1000)
        scenarios[Symbol("lcfs_$σ_int")] = (t=0.0, θ_avi=0.0, σ=Float64(σ), p=0.0, carbon_tax_scope=:aviation)
    end

    # 4. Tax credit scenarios (varying p)
    for p in POLICY_RANGES.p
        p_int = round(Int, p * 100)
        scenarios[Symbol("taxcredit_$p_int")] = (t=0.0, θ_avi=0.0, σ=0.0, p=Float64(p), carbon_tax_scope=:aviation)
    end

    # Add status quo
    scenarios[:statusquo] = (t=0.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation)
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

all_results, all_solutions = run_extended_analysis(params, EXTENDED_POLICY_MATRIX, verbose=true)

results_extended_analysis = calculate_extended_welfare_analysis(
    all_solutions,
    EXTENDED_POLICY_MATRIX,
    params,
    scc=190.0
)

@save joinpath(OUTPUT_DIR, "results_extended.jld2") all_solutions EXTENDED_POLICY_MATRIX results_extended_analysis
println("✓ Saved results_extended.jld2")

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

statusquo_emission = results_extended_analysis.solutions[:statusquo].emissions.total

println("=== Social MAC = 0 교차점 ===\n")

for policy_type in [:carbontax, :rfs, :lcfs, :taxcredit]
    mac_data = mac_extended[policy_type]
    isempty(mac_data) && continue

    data_iter = policy_type == :taxcredit ? mac_data[2:end] : mac_data

    abatement_values = Float64[]
    mac_social_values = Float64[]
    policy_values = Float64[]

    for d in data_iter
        abatement = statusquo_emission - d.emission
        push!(abatement_values, abatement)
        push!(mac_social_values, d.mac_social)

        # scenario 이름에서 policy stringency 추출
        config = EXTENDED_POLICY_MATRIX[d.scenario]
        pval = policy_type == :carbontax ? config.t :
               policy_type == :rfs ? config.θ_avi :
               policy_type == :lcfs ? config.σ :
               config.p
        push!(policy_values, pval)
    end

    idx = sortperm(abatement_values)
    abatement = abatement_values[idx]
    mac_soc = mac_social_values[idx]
    policy_val = policy_values[idx]

    zero_crossings = []
    for i in 1:length(mac_soc)-1
        if mac_soc[i] * mac_soc[i+1] <= 0
            x0, x1 = abatement[i], abatement[i+1]
            y0, y1 = mac_soc[i], mac_soc[i+1]
            p0, p1 = policy_val[i], policy_val[i+1]

            frac = -y0 / (y1 - y0)          # 선형 보간 비율
            x_zero = x0 + frac * (x1 - x0)    # abatement 보간
            p_zero = p0 + frac * (p1 - p0)    # policy stringency 보간

            push!(zero_crossings, (abatement=x_zero, policy=p_zero))
        end
    end

    policy_label = policy_type == :carbontax ? "t (\$/ton CO₂)" :
                   policy_type == :rfs ? "θ_avi" :
                   policy_type == :lcfs ? "σ" :
                   "p (\$/gallon)"

    if isempty(zero_crossings)
        println("$(policy_type): Social MAC = 0 교차점 없음")
        println("  MAC 범위: $(round(minimum(mac_soc),digits=1)) ~ $(round(maximum(mac_soc),digits=1))")
    else
        for zc in zero_crossings
            println("$(policy_type):")
            println("  abatement    = $(round(zc.abatement, digits=6)) Billion tons CO₂")
            println("  $(policy_label) = $(round(zc.policy, digits=4))")
        end
    end
    println()
end
# Plotting
using Plots

# ── ep_3B, ep_5B 먼저 로드 ──────────────────────────────────────────────
@load joinpath(OUTPUT_DIR, "results_equivalent_emissions.jld2") equivalent_emission_policies
ep_3B = equivalent_emission_policies

@load joinpath(OUTPUT_DIR, "results_equivalent_emissions_5.jld2") equivalent_emission_policies
ep_5B = equivalent_emission_policies

# ── vlines 계산 ──────────────────────────────────────────────────────────
statusquo_em = results_extended_analysis.solutions[:statusquo].emissions.total

abatement_3B = statusquo_em - ep_3B[:rfs].actual_emission
abatement_5B = statusquo_em - ep_5B[:rfs].actual_emission

vlines_abatement = [(abatement_3B, "3B"), (abatement_5B, "5B")]

vlines_data = Dict(
    :carbontax => [(ep_3B[:carbontax].policy_value, "3B"), (ep_5B[:carbontax].policy_value, "5B")],
    :rfs => [(ep_3B[:rfs].policy_value, "3B"), (ep_5B[:rfs].policy_value, "5B")],
    :lcfs => [(ep_3B[:lcfs].policy_value, "3B"), (ep_5B[:lcfs].policy_value, "5B")],
    :taxcredit => [(ep_3B[:taxcredit].policy_value, "3B"), (ep_5B[:taxcredit].policy_value, "5B")],
)

# ── MAC plot 함수 정의 (vlines_abatement 버전 하나만 유지) ────────────────
function plot_mac_comparison(results_extended_analysis, mac_extended; vlines_abatement=nothing)
    solutions = results_extended_analysis.solutions
    statusquo_emission = solutions[:statusquo].emissions.total
    max_abatement = 0.12

    p = plot(layout=(1, 2), size=(1800, 600),
        plot_title="Marginal Abatement Cost Curves Comparison",
        plot_titlefontsize=18, plot_titlefontweight=:bold,
        left_margin=15Plots.mm, bottom_margin=10Plots.mm)

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
            legend=:topleft, xlims=(0, max_abatement), ylims=(-100, 2400),
            legendfontsize=15,
            titlefontsize=20, titlefontweight=:bold,
            guidefontsize=15, tickfontsize=10)

        plot!(p[2], abatement_sorted, mac_social_sorted,
            label=info.label, linewidth=2.5, color=info.color,
            xlabel="Cumulative Abatement (Billion tons CO2)",
            ylabel="MAC (\$/ton CO2)", title="Social MAC",
            legend=:topleft, xlims=(0, max_abatement), ylims=(-500, 2400),
            legendfontsize=15,
            titlefontsize=20, titlefontweight=:bold,
            guidefontsize=15, tickfontsize=10)
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
p_mac_comparison = plot_mac_comparison(results_extended_analysis, mac_extended; vlines_abatement=vlines_abatement)


function plot_mac_comparison_simple(results_extended_analysis, mac_extended;
    vlines_abatement=nothing,
    y_max=2200.0, y_min=-250.0,
    fig_size=(1800, 1400))   # ← 세로 길게

    solutions = results_extended_analysis.solutions
    statusquo_emission = solutions[:statusquo].emissions.total
    max_abatement = 0.105
    bg_color = RGB(0.96, 0.96, 0.94)

    policy_info = [
        (:carbontax, "Carbon Tax", :blue),
        (:rfs, "RFS Aviation", :red),
        (:lcfs, "LCFS", :green),
        (:taxcredit, "Tax Credit", :purple),
    ]

    mac_plot_data = Dict()
    for (policy_type, _, _) in policy_info
        mac_data = mac_extended[policy_type]
        isempty(mac_data) && continue
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
        isempty(abatement_values) && continue
        idx = sortperm(abatement_values)
        mac_plot_data[policy_type] = (
            abatement=abatement_values[idx],
            private=mac_private_values[idx],
            social=mac_social_values[idx]
        )
    end

    vline_colors = [:darkred, :darkblue]

    function make_panel(mac_key, title, show_ylabel)
        p = plot(
            title=title,
            titlefontsize=18, titlefontweight=:bold,
            ylabel=show_ylabel ? "MAC (\$/ton CO₂)" : "",
            xlabel="Cumulative Abatement (Billion tons CO₂)",
            legend=false,
            grid=true,
            #xlims                   = (0.09, 0.10),
            xlims=(0, max_abatement),
            ylims=(y_min, y_max),
            yticks=collect(-200:200.0:y_max),  # 200 간격
            left_margin=show_ylabel ? 22Plots.mm : 5Plots.mm,
            right_margin=8Plots.mm,
            bottom_margin=12Plots.mm,
            top_margin=8Plots.mm,
            guidefontsize=20, tickfontsize=15,
            background_color_inside=bg_color,
            background_color=:white
        )

        for (policy_type, label, color) in policy_info
            haskey(mac_plot_data, policy_type) || continue
            d = mac_plot_data[policy_type]
            vals = mac_key == :private ? d.private : d.social
            plot!(p, d.abatement, vals,
                label=label, linewidth=2.5, color=color)
        end

        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=2.0, label="", alpha=0.8)

        if !isnothing(vlines_abatement)
            for (j, (xval, vlabel)) in enumerate(vlines_abatement)
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p, xval, y_max * 0.97, text(vlabel, c, :center, 9))
            end
        end

        return p
    end

    p_priv = make_panel(:private, "Private MAC", true)
    p_soc = make_panel(:social, "Social MAC", false)

    p_legend = plot(legend=:top, legendcolumns=4,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=12, background_color=:white)
    for (_, label, color) in policy_info
        plot!(p_legend, [NaN], [NaN], label=label, linewidth=3, color=color)
    end

    final_plot = plot(
        plot(p_priv, p_soc, layout=(1, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=fig_size,
        plot_title="Marginal Abatement Cost Curves Comparison",
        plot_titlefontsize=25,
        plot_titlefontweight=:bold,
        background_color=:white)

    return final_plot
end

p_mac_simple = plot_mac_comparison_simple(
    results_extended_analysis, mac_extended;
    vlines_abatement=vlines_abatement,
    y_max=2200.0,
    y_min=-250.0,
    fig_size=(1800, 1400))

display(p_mac_simple)


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

# Food

function plot_food_products_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    policies = [
        (:carbontax, "Carbon Tax (\$/ton CO2)", "Carbon Tax"),
        (:rfs, "RFS Aviation Mandate (θ_avi)", "RFS Aviation"),
        (:lcfs, "LCFS Standard (σ)", "LCFS"),
        (:taxcredit, "Tax Credit (\$/gal)", "Tax Credit")
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

    corn_plots = []
    soyoil_plots = []
    soymeal_plots = []

    for (policy_type, xlabel, title) in policies
        scenario_list = sort(scenario_groups[policy_type], by=s -> parse(Int, split(String(s), "_")[2]))

        x_vals = [get_policy_value(s, policy_type) for s in scenario_list]
        total_corn = [solutions[s].x[:corn] for s in scenario_list]
        ddgs = [0.092 * solutions[s].q[:ethanol] +
                0.159 * (solutions[s].q[:saf_atj_conv] + solutions[s].q[:saf_atj_cs])
                for s in scenario_list]
        corn_food = total_corn .- ddgs
        soy_oil = [solutions[s].x[:soyoil] for s in scenario_list]
        soy_meal = [solutions[s].x[:soymeal] for s in scenario_list]

        # vlines 추가 헬퍼
        function add_vlines!(p, ylim)
            if !isnothing(vlines) && haskey(vlines, policy_type)
                vline_colors = [:darkred, :darkblue]
                for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                    c = vline_colors[min(j, length(vline_colors))]
                    vline!(p, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                    annotate!(p, xval, ylim[2] * 0.97, text(vlabel, c, :center, 9))
                end
            end
        end

        # Corn
        p_corn = plot(xlabel=xlabel, ylabel="billion bushels",
            title=title, titlefontsize=25, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=(x_vals[1], x_vals[end]), ylims=(0, 15),
            left_margin=20Plots.mm, bottom_margin=10Plots.mm,
            guidefontsize=20, tickfontsize=18)
        plot!(p_corn, x_vals, corn_food, linewidth=4, color=:orange, label="Corn for food")
        plot!(p_corn, x_vals, total_corn, linewidth=4, color=:red, label="Total Corn (+ DDGS)")
        add_vlines!(p_corn, (0, 15))
        push!(corn_plots, p_corn)

        # Soybean Oil
        p_oil = plot(xlabel=xlabel, ylabel="billion lbs",
            title=title, titlefontsize=25, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=(x_vals[1], x_vals[end]), ylims=(0, 15),
            left_margin=20Plots.mm, bottom_margin=10Plots.mm,
            guidefontsize=20, tickfontsize=18)
        plot!(p_oil, x_vals, soy_oil, linewidth=4, color=:darkgreen, label="Soybean Oil")
        add_vlines!(p_oil, (10, 16))
        push!(soyoil_plots, p_oil)

        # Soybean Meal
        p_meal = plot(xlabel=xlabel, ylabel="MMT",
            title=title, titlefontsize=25, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=(x_vals[1], x_vals[end]), ylims=(0, 80),
            left_margin=20Plots.mm, bottom_margin=10Plots.mm,
            guidefontsize=20, tickfontsize=18)
        plot!(p_meal, x_vals, soy_meal, linewidth=4, color=:brown, label="Soybean Meal")
        add_vlines!(p_meal, (70, 79))
        push!(soymeal_plots, p_meal)
    end

    # 범례
    function make_legend(items)
        p_leg = plot(legend=:top, legendcolumns=length(items),
            grid=false, showaxis=false, ticks=false,
            xlims=(0, 1), ylims=(0, 1), framestyle=:none,
            legendfontsize=18, size=(2400, 60))
        for (lbl, color) in items
            plot!(p_leg, [NaN], [NaN], label=lbl, linewidth=2, color=color)
        end
        return p_leg
    end

    p_corn_final = plot(
        plot(corn_plots..., layout=(2, 2)),
        make_legend([("Corn for food", :orange), ("Total Corn (+ DDGS)", :red)]),
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2400, 1700),
        plot_title="Corn for Food by Policy Stringency",
        plot_titlefontsize=25, plot_titlefontweight=:bold, margin=8Plots.mm,
    )

    p_oil_final = plot(
        plot(soyoil_plots..., layout=(2, 2)),
        make_legend([("Soybean Oil", :darkgreen)]),
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2400, 1700),
        plot_title="Soybean Oil for Food by Policy Stringency",
        plot_titlefontsize=25, plot_titlefontweight=:bold, margin=8Plots.mm
    )

    p_meal_final = plot(
        plot(soymeal_plots..., layout=(2, 2)),
        make_legend([("Soybean Meal", :brown)]),
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2400, 1700),
        plot_title="Soybean Meal for Food by Policy Stringency",
        plot_titlefontsize=25, plot_titlefontweight=:bold, margin=8Plots.mm
    )

    return p_corn_final, p_oil_final, p_meal_final
end

# 실행
p_corn, p_oil, p_meal = plot_food_products_by_policy(results_extended_analysis; vlines=vlines_data)
display(p_corn)
display(p_oil)
display(p_meal)
#savefig(p_corn, joinpath(OUTPUT_DIR, "corn_by_policy.png"))
#savefig(p_oil, joinpath(OUTPUT_DIR, "soyoil_by_policy.png"))
#savefig(p_meal, joinpath(OUTPUT_DIR, "soymeal_by_policy.png"))



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
        titlefontsize=18, titlefontweight=:bold,
        left_margin=10Plots.mm, bottom_margin=5Plots.mm)

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
            labelfontsize=15, titlefontsize=16, titlefontweight=:bold,
            title=info.label, legend=:best, legendfontsize=10)

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
        left_margin=10Plots.mm, bottom_margin=5Plots.mm)

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
            ylabel="Emissions (Billion ton CO2e)", labelfontsize=15, titlefontsize=16, titlefontweight=:bold, tickfontsize=10,
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

# Emissions stacked by sector plot
function plot_emissions_stacked_broken_axis(results_extended_analysis; vlines=nothing,
    break_point=1.2, y_max=2.6)

    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    GAS_FUELS = [:gasoline, :ethanol]
    DIESEL_FUELS = [:diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]

    function get_sector_emissions(sol)
        gas = sum(sol.emissions.by_fuel[g] for g in GAS_FUELS)
        die = sum(sol.emissions.by_fuel[g] for g in DIESEL_FUELS)
        return (food=sol.emissions.food, gas=gas, die=die, avi=sol.emissions.aviation)
    end

    sector_info = [
        (:gas, "Road (Gasoline)", :steelblue),
        (:die, "Road (Diesel)", :orange),
        (:food, "Food", :green),
        (:avi, "Aviation", :red),
    ]

    policy_info = Dict(
        :carbontax => (label="Carbon Tax", xlabel="Tax (\$/ton CO₂)"),
        :rfs => (label="RFS Aviation", xlabel="Mandate Share (θ_avi)"),
        :lcfs => (label="LCFS", xlabel="Carbon Intensity Limit (σ)"),
        :taxcredit => (label="Tax Credit", xlabel="Credit (\$/gallon)")
    )

    function get_x(s, policy_type)
        config = EXTENDED_POLICY_MATRIX[s]
        policy_type == :carbontax ? config.t :
        policy_type == :rfs ? config.θ_avi :
        policy_type == :lcfs ? config.σ :
        config.p
    end

    function draw_stacked!(p, xs, sector_vals)
        cumsum_vals = zeros(length(xs))
        for (key, label, color) in sector_info
            new_cum = cumsum_vals .+ sector_vals[key]
            plot!(p, xs, new_cum,
                fillrange=cumsum_vals,
                fillalpha=0.75, fillcolor=color,
                linewidth=1.2, color=color, label=label)
            cumsum_vals = new_cum
        end
        return cumsum_vals
    end

    top_plots = []
    bot_plots = []

    for policy_type in [:carbontax, :rfs, :lcfs, :taxcredit]
        scenario_list = scenario_groups[policy_type]
        isempty(scenario_list) && continue

        sorted_scenarios = sort(scenario_list,
            by=s -> parse(Int, split(String(s), "_")[2]))

        xs = Float64[]
        sector_vals = Dict(k => Float64[] for (k, _, _) in sector_info)

        for s in sorted_scenarios
            sol = solutions[s]
            isnothing(sol) && continue
            push!(xs, get_x(s, policy_type))
            em = get_sector_emissions(sol)
            for (key, _, _) in sector_info
                push!(sector_vals[key], getfield(em, key))
            end
        end

        info = policy_info[policy_type]
        xl = (xs[1], xs[end])

        # ── TOP 패널 (break_point ~ y_max) ─────────────────────────────
        p_top = plot(
            title=info.label,
            titlefontsize=16, titlefontweight=:bold,
            ylabel="Billion ton CO₂e",
            legend=false, grid=true,
            xlims=xl,
            ylims=(break_point, y_max),
            xticks=:none,
            bottom_margin=-4Plots.mm,
            top_margin=8Plots.mm,
            left_margin=18Plots.mm,
            guidefontsize=12, tickfontsize=10
        )
        draw_stacked!(p_top, xs, sector_vals)

        # 흰 띠 — break 경계를 두껍게 표시
        hline!(p_top, [break_point],
            color=:white, linewidth=6, label="")

        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p_top, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p_top, xval, y_max * 0.97, text(vlabel, c, :center, 9))
            end
        end

        push!(top_plots, p_top)

        # ── BOTTOM 패널 (0 ~ break_point) ──────────────────────────────
        p_bot = plot(
            xlabel=info.xlabel,
            ylabel="",
            legend=false, grid=true,
            xlims=xl,
            ylims=(0.0, break_point),
            yticks=[0.0],          # 0만 표시
            top_margin=-4Plots.mm,
            bottom_margin=12Plots.mm,
            left_margin=18Plots.mm,
            guidefontsize=12, tickfontsize=10
        )
        draw_stacked!(p_bot, xs, sector_vals)

        # 흰 띠 — break 경계를 두껍게 표시
        hline!(p_bot, [break_point],
            color=:white, linewidth=6, label="")

        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p_bot, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
            end
        end

        push!(bot_plots, p_bot)
    end

    # 공통 범례
    p_legend = plot(
        legend=:top, legendcolumns=4,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=13, size=(2200, 60)
    )
    for (key, label, color) in sector_info
        plot!(p_legend, [NaN], [NaN],
            label=label, linewidth=4, color=color,
            fillalpha=0.75, fillcolor=color)
    end

    all_panels = vcat(top_plots, bot_plots)

    final_plot = plot(
        plot(all_panels..., layout=grid(2, 4, heights=[0.75, 0.25])),
        p_legend,
        layout=grid(2, 1, heights=[0.93, 0.07]),
        size=(2200, 1300),
        plot_title="Emissions by Sector and Policy Type (Stacked, Broken Axis)",
        plot_titlefontsize=20,
        plot_titlefontweight=:bold,
        margin=8Plots.mm
    )

    return final_plot
end

p_emissions_broken = plot_emissions_stacked_broken_axis(
    results_extended_analysis;
    vlines=vlines_data,
    break_point=1.2,
    y_max=2.6)
display(p_emissions_broken)




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
            titlefontsize=25, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=(x_min, x_max),
            ylims=fuel_config.ylims,
            margin=10Plots.mm, guidefontsize=25,
            left_margin=15Plots.mm, bottom_margin=15Plots.mm,
            tickfontsize=20,
            labelfontsize=23
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
        legendfontsize=20, size=(2200, 80)
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
        plot_titlefontsize=30, plot_titlefontweight=:bold,
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
    legendcolumns=3
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
    ylims=(0, 200),
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
    ylims=(30, 55),
    biofuel_types=[
        (:q_rd_soy, "Soy Renewable Diesel", :red),
        (:q_rd_nonsoy, "Non-soy Renewable Diesel", :green),
        (:q_biodiesel_soy, "Soy Biodiesel", :blue),
        (:q_biodiesel_nonsoy, "Non-soy Biodiesel", :purple)
    ],
    plot_title="Diesel Fuel Production by Policy Stringency",
    filename="diesel_fuel_stacked.png",
    legendcolumns=3
)

diesel_plot = plot_fuel_production_stacked(results_df, diesel_config; vlines=vlines_data)
#savefig(diesel_plot, diesel_config.filename)
#display(diesel_plot)


# =================================================================================
# Sensitivity Analysis: c0 ATJ/HEFA across full policy grid
# =================================================================================

c0_scenarios = [
    (atj=0.32, hefa=0.44, label="Low"), # lowest production cost - feedstock cost
    (atj=2.3, hefa=1.145, label="Kumar TEA"), # Kumar TEA
    (atj=4.91, hefa=1.56, label="High") # highest production cost - feedstock cost
]

sensitivity_extended = Dict()

for c0_case in c0_scenarios
    println("\n" * "="^80)
    println("Running extended grid: c0_atj=$(c0_case.atj), c0_hefa=$(c0_case.hefa)")
    println("="^80)

    # params 수정
    new_fuel_cost = deepcopy(params.supply.fuel)
    new_fuel_cost[:saf_atj_shared] = (
        c0=c0_case.atj,
        c1=params.supply.fuel[:saf_atj_shared].c1,
        c2=params.supply.fuel[:saf_atj_shared].c2,
        v=params.supply.fuel[:saf_atj_shared].v
    )
    new_fuel_cost[:saf_hefa_shared] = (
        c0=c0_case.hefa,
        c1=params.supply.fuel[:saf_hefa_shared].c1,
        c2=params.supply.fuel[:saf_hefa_shared].c2,
        v=params.supply.fuel[:saf_hefa_shared].v
    )
    new_params = (
        sets=params.sets,
        meta=params.meta,
        coeff=params.coeff,
        demand=params.demand,
        supply=(fuel=new_fuel_cost, land=params.supply.land)
    )

    # 전체 그리드 실행
    all_results_s, all_solutions_s = run_extended_analysis(new_params, EXTENDED_POLICY_MATRIX, verbose=false)

    # welfare 계산
    results_extended_s = calculate_extended_welfare_analysis(
        all_solutions_s, EXTENDED_POLICY_MATRIX, new_params, scc=190.0
    )

    sensitivity_extended[c0_case.label] = results_extended_s
end

# 저장
@save joinpath(OUTPUT_DIR, "sensitivity_extended_c0.jld2") sensitivity_extended c0_scenarios
println("✓ Saved sensitivity_extended_c0.jld2")

function plot_saf_sensitivity(sensitivity_extended, c0_scenarios)
    policies = [
        ("carbontax", :t, "Carbon Tax (\$/ton CO2e)", "Carbon Tax Policy"),
        ("rfs", :θ_avi, "RFS Aviation Mandate (θ_avi)", "RFS Aviation Policy"),
        ("lcfs", :σ, "LCFS Standard (σ)", "LCFS Policy"),
        ("taxcredit", :p, "Tax Credit Rate (\$/gallon)", "Tax Credit Policy")
    ]

    saf_fuels = [
        (:q_saf_atj_conv, "ATJ-conv"),
        (:q_saf_atj_cs, "ATJ-cs"),
        (:q_saf_hefa_conv, "HEFA-conv"),
        (:q_saf_hefa_cs, "HEFA-cs"),
        (:q_saf_hefa_nonsoy, "HEFA-nonsoy")
    ]

    case_colors = Dict("Low" => :blue, "Kumar TEA" => :black, "High" => :red)
    case_styles = Dict("Low" => :dash, "Kumar TEA" => :solid, "High" => :dot)

    plots = []

    for (policy_type, xcol, xlabel, title) in policies
        p = plot(
            xlabel=xlabel,
            ylabel="SAF Production (billion gallons)",
            title=title,
            titlefontsize=14, titlefontweight=:bold,
            legend=:topleft, grid=true,
            margin=10Plots.mm, guidefontsize=11
        )

        for c0_case in c0_scenarios
            label = c0_case.label
            results_s = sensitivity_extended[label]

            # DataFrame 생성
            df_s = results_to_dataframe(results_s, EXTENDED_POLICY_MATRIX)
            df = filter(row -> row.policy_type == policy_type, df_s)
            sort!(df, xcol)

            # total SAF
            total_saf = sum(df[!, col] for (col, _) in saf_fuels)

            plot!(p, df[!, xcol], total_saf,
                label="Total SAF ($(label))",
                linewidth=2.5,
                color=case_colors[label],
                linestyle=case_styles[label]
            )
        end

        push!(plots, p)
    end

    final_plot = plot(
        plots...,
        layout=(2, 2),
        size=(2000, 1400),
        plot_title="SAF Production by Policy Stringency (c0 Sensitivity)",
        plot_titlefontsize=20, plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end

p_saf_sensitivity = plot_saf_sensitivity(sensitivity_extended, c0_scenarios)
#display(p_saf_sensitivity)
#savefig(p_saf_sensitivity, joinpath(OUTPUT_DIR, "saf_sensitivity_c0.png"))


# Stacked version of SAF sensitivity plot
function plot_saf_stacked_sensitivity(sensitivity_extended, c0_scenarios, fuel_config)
    policies = [
        ("carbontax", :t, "Carbon Tax (\$/ton CO2e)", "Carbon Tax"),
        ("rfs", :θ_avi, "RFS Aviation Mandate (θ_avi)", "RFS Aviation"),
        ("lcfs", :σ, "LCFS Standard (σ)", "LCFS"),
        ("taxcredit", :p, "Tax Credit (\$/gallon)", "Tax Credit")
    ]

    all_plots = []

    for c0_case in c0_scenarios
        label = c0_case.label
        results_s = sensitivity_extended[label]
        df_s = results_to_dataframe(results_s, EXTENDED_POLICY_MATRIX)

        for (policy_type, xcol, xlabel, pol_title) in policies
            df = filter(row -> row.policy_type == policy_type, df_s)
            sort!(df, xcol)

            x_min = minimum(df[!, xcol])
            x_max = maximum(df[!, xcol])
            x_vals = df[!, xcol]

            biofuel_sorted = sort(fuel_config.biofuel_types,
                by=x -> mean(df[!, x[1]]),
                rev=true
            )

            row_idx = findfirst(c -> c.label == label, c0_scenarios)
            col_idx = findfirst(p -> p[1] == policy_type, policies)

            top_title = row_idx == 1 ? pol_title : ""
            y_label = col_idx == 1 ? "(billion gal)" : ""
            x_label = row_idx == length(c0_scenarios) ? xlabel : ""

            p = plot(
                xlabel=x_label,
                ylabel=y_label,
                title=top_title,
                titlefontsize=20, titlefontweight=:bold,
                legend=false, grid=true,
                xlims=(x_min, x_max),
                ylims=fuel_config.ylims,
                margin=5Plots.mm, guidefontsize=10
            )

            main_fuel_vals = df[!, fuel_config.main_fuel]
            plot!(p, x_vals, main_fuel_vals,
                fillrange=fuel_config.ylims[1],
                fillalpha=0.7, fillcolor=:lightgray,
                linewidth=1.5, color=:black, label=""
            )

            cumsum_vals = copy(main_fuel_vals)
            for (col, lbl, color) in biofuel_sorted
                biofuel_vals = df[!, col]
                new_cumsum = cumsum_vals .+ biofuel_vals
                plot!(p, x_vals, new_cumsum,
                    fillrange=cumsum_vals,
                    fillalpha=0.7, fillcolor=color,
                    linewidth=1.5, color=color, label=""
                )
                cumsum_vals = new_cumsum
            end

            push!(all_plots, p)
        end
    end

    # ── 행 레이블 플롯 ──────────────────────────────────────────────────
    label_plots = []
    for c0_case in c0_scenarios
        p_lbl = plot(
            grid=false, showaxis=false, ticks=false,
            xlims=(0, 1), ylims=(0, 1), framestyle=:none
        )
        annotate!(p_lbl, 0.5, 0.5, text(c0_case.label, :black, :center, 20, rotation=90))
        push!(label_plots, p_lbl)
    end

    # ── 레이블 열 + 데이터 열 결합 (5열) ────────────────────────────────
    combined_plots = []
    for (i, lbl_p) in enumerate(label_plots)
        push!(combined_plots, lbl_p)
        for j in 1:4
            push!(combined_plots, all_plots[(i-1)*4+j])
        end
    end

    # ── 범례 ────────────────────────────────────────────────────────────
    p_legend = plot(
        legend=:top, legendcolumns=fuel_config.legendcolumns,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=15, size=(2500, 80)
    )
    plot!(p_legend, [NaN], [NaN],
        label=fuel_config.main_fuel_label,
        linewidth=3, color=:black, fillalpha=0.7, fillcolor=:lightgray
    )
    for (col, lbl, color) in fuel_config.biofuel_types
        plot!(p_legend, [NaN], [NaN],
            label=lbl, linewidth=3, color=color,
            fillalpha=0.7, fillcolor=color
        )
    end

    final_plot = plot(
        plot(combined_plots..., layout=grid(length(c0_scenarios), 5,
            widths=[0.05, 0.2375, 0.2375, 0.2375, 0.2375])),
        p_legend,
        layout=grid(2, 1, heights=[0.97, 0.03]),
        size=(3000, 2000),
        plot_title="SAF Production by Policy Stringency (c0 Sensitivity)",
        plot_titlefontsize=20, plot_titlefontweight=:bold,
        margin=8Plots.mm
    )

    return final_plot
end

# 실행
p_saf_stacked_sens = plot_saf_stacked_sensitivity(sensitivity_extended, c0_scenarios, aviation_config)
display(p_saf_stacked_sens)
savefig(p_saf_stacked_sens, joinpath(OUTPUT_DIR, "saf_stacked_sensitivity_c0.png"))




# price plot by policy type
using Plots
function plot_aviation_prices_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    r_energy = params.coeff.r
    β = params.coeff.beta

    policies = [
        (:carbontax, "Carbon Tax (\$/ton CO2e)", "Carbon Tax", false),
        (:rfs, "RFS Aviation Mandate (θ_avi)", "RFS Aviation", false),
        (:lcfs, "LCFS Standard (σ)", "LCFS", true),
        (:taxcredit, "Tax Credit (\$/gallon)", "Tax Credit", false)
    ]

    fuel_info = [
        (:jet_fuel, "Jet Fuel", :black, true),
        (:saf_atj_conv, "ATJ SAF (Conv)", :blue, false),
        (:saf_atj_cs, "ATJ SAF (CS)", :dodgerblue, false),
        (:saf_hefa_conv, "Soy HEFA SAF (Conv)", :green, false),
        (:saf_hefa_cs, "Soy HEFA SAF (CS)", :limegreen, false),
        (:saf_hefa_nonsoy, "Non-Soy HEFA SAF", :purple, false),
    ]

    function ppu(sol, g)
        p_avi = sol.p_c[:avi]
        if g == :jet_fuel
            return r_energy[:jet_fuel] * p_avi
        else
            return r_energy[:jet_fuel] * β[(:saf, :jet_fuel)] * p_avi
        end
    end

    function get_x(s, policy_type)
        config = EXTENDED_POLICY_MATRIX[s]
        policy_type == :carbontax ? config.t :
        policy_type == :rfs ? config.θ_avi :
        policy_type == :lcfs ? config.σ :
        config.p
    end

    AVY_YLIMS = (0.0, 0.3)  # 오른쪽 축 공통 범위

    plots_list = []

    for (policy_type, xlabel, title, reverse_sort) in policies
        scenario_list = scenario_groups[policy_type]
        sorted_scenarios = sort(scenario_list,
            by=s -> parse(Int, split(String(s), "_")[2]),
            rev=reverse_sort)

        p = plot(
            xlabel=xlabel,
            ylabel="\$/gallon",
            title=title,
            titlefontsize=22, titlefontweight=:bold,
            legend=false,                          # 개별 범례 제거
            grid=true,
            ylims=(0, 15),
            left_margin=18Plots.mm, bottom_margin=12Plots.mm,
            right_margin=22Plots.mm, top_margin=8Plots.mm,
            guidefontsize=16, tickfontsize=14
        )

        # ── 왼쪽 축: 연료 가격들 ──────────────────────────────────────────
        for (g, label, color, always_show) in fuel_info
            xs = Float64[]
            prices = Float64[]

            for s in sorted_scenarios
                sol = solutions[s]
                isnothing(sol) && continue
                qty = sol.q[g]
                if always_show || qty > 1e-6
                    push!(xs, get_x(s, policy_type))
                    push!(prices, ppu(sol, g))
                end
            end

            isempty(xs) && continue

            plot!(p, xs, prices,
                label=label, linewidth=3, color=color)
        end

        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1, label="")

        # ── 오른쪽 축: Aviation Mile Price (범위 고정) ────────────────────
        xs_avi = Float64[]
        p_avi_vals = Float64[]
        for s in sorted_scenarios
            sol = solutions[s]
            isnothing(sol) && continue
            push!(xs_avi, get_x(s, policy_type))
            push!(p_avi_vals, sol.p_c[:avi])
        end

        if !isempty(xs_avi)
            p_right = Plots.twinx(p)
            plot!(p_right, xs_avi, p_avi_vals,
                label="Aviation Mile Price",
                linewidth=3, color=:red, linestyle=:dash,
                ylabel="\$/aviation mile",
                ylims=AVY_YLIMS,                   # 오른쪽 축 범위 고정
                guidefontsize=16, tickfontsize=14,
                legend=false                       # 개별 범례 제거
            )
        end

        # ── vlines ───────────────────────────────────────────────────────
        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=2, label="")
                cur_ylims = ylims(p)
                annotate!(p, xval, cur_ylims[2] * 0.96,
                    text(vlabel, c, :center, 11))
            end
        end

        push!(plots_list, p)
    end

    # ── 공통 범례 패널 ────────────────────────────────────────────────────
    p_legend = plot(
        legend=:top, legendcolumns=4,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=14, size=(2200, 60)
    )
    for (g, label, color, _) in fuel_info
        plot!(p_legend, [NaN], [NaN], label=label, linewidth=3, color=color)
    end
    plot!(p_legend, [NaN], [NaN],
        label="Aviation Mile Price",
        linewidth=3, color=:red, linestyle=:dash)

    # ── 최종 조합 ─────────────────────────────────────────────────────────
    final_plot = plot(
        plot(plots_list..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.93, 0.07]),
        size=(2200, 1600),
        plot_title="Aviation Fuel Prices by Policy Stringency(Left: Fuel Price, Right: Aviation Mile Price)",
        plot_titlefontsize=26, plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end

p_avi_prices = plot_aviation_prices_by_policy(
    results_extended_analysis; vlines=vlines_data)
display(p_avi_prices)

# ── Gasoline Sector ──────────────────────────────────────────────────────────
function plot_gasoline_prices_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    r_energy = params.coeff.r
    β = params.coeff.beta

    policies = [
        (:carbontax, "Carbon Tax (\$/ton CO2e)", "Carbon Tax", false),
        (:rfs, "RFS Aviation Mandate (θ_avi)", "RFS Aviation", false),
        (:lcfs, "LCFS Standard (σ)", "LCFS", true),
        (:taxcredit, "Tax Credit (\$/gallon)", "Tax Credit", false)
    ]

    fuel_info = [
        (:gasoline, "Gasoline", :black, true),
        (:ethanol, "Ethanol", :red, false),
    ]

    function ppu(sol, g)
        p_gas = sol.p_c[:gas]
        if g == :gasoline
            return r_energy[:gasoline] * p_gas
        else  # ethanol
            return r_energy[:gasoline] * β[(:ethanol, :gasoline)] * p_gas
        end
    end

    function get_x(s, policy_type)
        config = EXTENDED_POLICY_MATRIX[s]
        policy_type == :carbontax ? config.t :
        policy_type == :rfs ? config.θ_avi :
        policy_type == :lcfs ? config.σ :
        config.p
    end

    GAS_YLIMS = (0.0, 4.0)   # 왼쪽 축
    GAS_RYLIMS = (0.0, 1.0)   # 오른쪽 축 (p_c[:gas])

    plots_list = []

    for (policy_type, xlabel, title, reverse_sort) in policies
        scenario_list = scenario_groups[policy_type]
        sorted_scenarios = sort(scenario_list,
            by=s -> parse(Int, split(String(s), "_")[2]),
            rev=reverse_sort)

        p = plot(
            xlabel=xlabel,
            ylabel="\$/gallon",
            title=title,
            titlefontsize=22, titlefontweight=:bold,
            legend=false,
            grid=true,
            ylims=GAS_YLIMS,
            left_margin=18Plots.mm, bottom_margin=12Plots.mm,
            right_margin=22Plots.mm, top_margin=8Plots.mm,
            guidefontsize=16, tickfontsize=14
        )

        for (g, label, color, always_show) in fuel_info
            xs = Float64[]
            prices = Float64[]
            for s in sorted_scenarios
                sol = solutions[s]
                isnothing(sol) && continue
                qty = sol.q[g]
                if always_show || qty > 1e-6
                    push!(xs, get_x(s, policy_type))
                    push!(prices, ppu(sol, g))
                end
            end
            isempty(xs) && continue
            plot!(p, xs, prices, label=label, linewidth=3, color=color)
        end

        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1, label="")

        # 오른쪽 축: p_c[:gas]
        xs_gas = Float64[]
        p_gas_vals = Float64[]
        for s in sorted_scenarios
            sol = solutions[s]
            isnothing(sol) && continue
            push!(xs_gas, get_x(s, policy_type))
            push!(p_gas_vals, sol.p_c[:gas])
        end

        if !isempty(xs_gas)
            p_right = Plots.twinx(p)
            plot!(p_right, xs_gas, p_gas_vals,
                label="Gasoline Consumer Price",
                linewidth=3, color=:orange, linestyle=:dash,
                ylabel="\$/gasoline mile",
                ylims=GAS_RYLIMS,
                guidefontsize=16, tickfontsize=14,
                legend=false
            )
        end

        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=2, label="")
                cur_ylims = ylims(p)
                annotate!(p, xval, cur_ylims[2] * 0.96, text(vlabel, c, :center, 11))
            end
        end

        push!(plots_list, p)
    end

    p_legend = plot(
        legend=:top, legendcolumns=3,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=14, size=(2200, 60)
    )
    for (g, label, color, _) in fuel_info
        plot!(p_legend, [NaN], [NaN], label=label, linewidth=3, color=color)
    end
    plot!(p_legend, [NaN], [NaN],
        label="Gasoline Consumer Price",
        linewidth=3, color=:orange, linestyle=:dash)

    final_plot = plot(
        plot(plots_list..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.93, 0.07]),
        size=(2200, 1600),
        plot_title="Gasoline Sector Prices by Policy Stringency (Left: Fuel Price, Right: Consumer Price)",
        plot_titlefontsize=22, plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end


# ── Diesel Sector ────────────────────────────────────────────────────────────
function plot_diesel_prices_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    r_energy = params.coeff.r
    β = params.coeff.beta

    policies = [
        (:carbontax, "Carbon Tax (\$/ton CO2e)", "Carbon Tax", false),
        (:rfs, "RFS Aviation Mandate (θ_avi)", "RFS Aviation", false),
        (:lcfs, "LCFS Standard (σ)", "LCFS", true),
        (:taxcredit, "Tax Credit (\$/gallon)", "Tax Credit", false)
    ]

    fuel_info = [
        (:diesel, "Diesel", :black, true),
        (:rd_soy, "Soy RD", :red, false),
        (:rd_nonsoy, "Non-Soy RD", :orange, false),
        (:biodiesel_soy, "Soy Biodiesel", :blue, false),
        (:biodiesel_nonsoy, "Non-Soy Biodiesel", :purple, false),
    ]

    function ppu(sol, g)
        p_die = sol.p_c[:die]
        if g == :diesel
            return r_energy[:diesel] * p_die
        elseif g in (:biodiesel_soy, :biodiesel_nonsoy)
            return r_energy[:diesel] * β[(:biodiesel, :diesel)] * p_die
        else  # rd_soy, rd_nonsoy
            return r_energy[:diesel] * β[(:rd, :diesel)] * p_die
        end
    end

    function get_x(s, policy_type)
        config = EXTENDED_POLICY_MATRIX[s]
        policy_type == :carbontax ? config.t :
        policy_type == :rfs ? config.θ_avi :
        policy_type == :lcfs ? config.σ :
        config.p
    end

    DIE_YLIMS = (0.0, 6.5)   # 왼쪽 축
    DIE_RYLIMS = (0.0, 2.0)   # 오른쪽 축 (p_c[:die])

    plots_list = []

    for (policy_type, xlabel, title, reverse_sort) in policies
        scenario_list = scenario_groups[policy_type]
        sorted_scenarios = sort(scenario_list,
            by=s -> parse(Int, split(String(s), "_")[2]),
            rev=reverse_sort)

        p = plot(
            xlabel=xlabel,
            ylabel="\$/gallon",
            title=title,
            titlefontsize=22, titlefontweight=:bold,
            legend=false,
            grid=true,
            ylims=DIE_YLIMS,
            left_margin=18Plots.mm, bottom_margin=12Plots.mm,
            right_margin=22Plots.mm, top_margin=8Plots.mm,
            guidefontsize=16, tickfontsize=14
        )

        for (g, label, color, always_show) in fuel_info
            xs = Float64[]
            prices = Float64[]
            for s in sorted_scenarios
                sol = solutions[s]
                isnothing(sol) && continue
                qty = sol.q[g]
                if always_show || qty > 1e-6
                    push!(xs, get_x(s, policy_type))
                    push!(prices, ppu(sol, g))
                end
            end
            isempty(xs) && continue
            plot!(p, xs, prices, label=label, linewidth=3, color=color)
        end

        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1, label="")

        # 오른쪽 축: p_c[:die]
        xs_die = Float64[]
        p_die_vals = Float64[]
        for s in sorted_scenarios
            sol = solutions[s]
            isnothing(sol) && continue
            push!(xs_die, get_x(s, policy_type))
            push!(p_die_vals, sol.p_c[:die])
        end

        if !isempty(xs_die)
            p_right = Plots.twinx(p)
            plot!(p_right, xs_die, p_die_vals,
                label="Diesel Consumer Price",
                linewidth=3, color=:brown, linestyle=:dash,
                ylabel="\$/diesel mile",
                ylims=DIE_RYLIMS,
                guidefontsize=16, tickfontsize=14,
                legend=false
            )
        end

        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=2, label="")
                cur_ylims = ylims(p)
                annotate!(p, xval, cur_ylims[2] * 0.96, text(vlabel, c, :center, 11))
            end
        end

        push!(plots_list, p)
    end

    p_legend = plot(
        legend=:top, legendcolumns=3,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=14, size=(2200, 60)
    )
    for (g, label, color, _) in fuel_info
        plot!(p_legend, [NaN], [NaN], label=label, linewidth=3, color=color)
    end
    plot!(p_legend, [NaN], [NaN],
        label="Diesel Consumer Price",
        linewidth=3, color=:brown, linestyle=:dash)

    final_plot = plot(
        plot(plots_list..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.93, 0.07]),
        size=(2200, 1600),
        plot_title="Diesel Sector Prices by Policy Stringency (Left: Fuel Price, Right: Consumer Price)",
        plot_titlefontsize=22, plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end


# ── 실행 ─────────────────────────────────────────────────────────────────────
p_gas_prices = plot_gasoline_prices_by_policy(
    results_extended_analysis; vlines=vlines_data)
display(p_gas_prices)

p_die_prices = plot_diesel_prices_by_policy(
    results_extended_analysis; vlines=vlines_data)
display(p_die_prices)


# 
function plot_feedstock_prices_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    policies = [
        (:carbontax, "Carbon Tax (\$/ton CO2e)", "Carbon Tax", false),
        (:rfs, "RFS Aviation Mandate (θ_avi)", "RFS Aviation", false),
        (:lcfs, "LCFS Standard (σ)", "LCFS", true),
        (:taxcredit, "Tax Credit (\$/gallon)", "Tax Credit", false)
    ]

    corn_info = [
        (:feedstock_corn_n, "Corn (Conv)", :darkorange, :solid),
        (:feedstock_corn_cs, "Corn (CS)", :darkorange, :dash),
    ]

    soyoil_info = [
        (:feedstock_soy_n, "Soyoil (Conv)", :darkgreen, :solid),
        (:feedstock_soy_cs, "Soyoil (CS)", :darkgreen, :dash),
    ]

    # ── conv + CS 모두 포함해서 범위 설정 ────────────────────────────────
    all_corn = [solutions[k].p_f[c] for k in keys(solutions)
                                        if !isnothing(solutions[k])
                for (c, _, _, _) in corn_info]
    all_soy = [solutions[k].p_f[s] for k in keys(solutions)
                                       if !isnothing(solutions[k])
               for (s, _, _, _) in soyoil_info]
    CORN_YLIMS = (floor(minimum(all_corn)), ceil(maximum(all_corn)))
    SOYOIL_YLIMS = (floor(minimum(all_soy)), ceil(maximum(all_soy)))

    function get_x(s, policy_type)
        config = EXTENDED_POLICY_MATRIX[s]
        policy_type == :carbontax ? config.t :
        policy_type == :rfs ? config.θ_avi :
        policy_type == :lcfs ? config.σ :
        config.p
    end

    plots_list = []

    for (policy_type, xlabel, title, reverse_sort) in policies
        scenario_list = scenario_groups[policy_type]
        sorted_scenarios = sort(scenario_list,
            by=s -> parse(Int, split(String(s), "_")[2]),
            rev=reverse_sort)

        p = plot(
            xlabel=xlabel,
            ylabel="\$/bushel (Corn)",
            xguidefontcolor=:black,              # ← xlabel 검정
            yguidefontcolor=:darkorange,         # ← ylabel 다크오렌지
            title=title,
            titlefontsize=22, titlefontweight=:bold,
            legend=false,
            grid=true,
            ylims=CORN_YLIMS,
            left_margin=18Plots.mm, bottom_margin=12Plots.mm,
            right_margin=22Plots.mm, top_margin=8Plots.mm,
            guidefontsize=20, tickfontsize=14
        )

        # ── 왼쪽 축: Corn feedstock prices ───────────────────────────────
        for (key, label, color, lstyle) in corn_info
            xs = Float64[]
            prices = Float64[]
            for s in sorted_scenarios
                sol = solutions[s]
                isnothing(sol) && continue
                if key == :feedstock_corn_cs
                    sol.q_feedstock[:corn_cs] > 1e-6 || continue
                end
                push!(xs, get_x(s, policy_type))
                push!(prices, sol.p_f[key])
            end
            isempty(xs) && continue
            plot!(p, xs, prices,
                label=label, linewidth=3, color=color, linestyle=lstyle)
        end

        hline!(p, [0], color=:black, linestyle=:solid, linewidth=1.5, label="", alpha=0.6)

        # ── vlines — 메인 패널에만 ────────────────────────────────────────
        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=2, label="")
                annotate!(p, xval, CORN_YLIMS[2] * 0.96,
                    text(vlabel, c, :center, 11))
            end
        end

        # ── 오른쪽 축: Soyoil feedstock prices ───────────────────────────
        p_right = Plots.twinx(p)
        plot!(p_right,
            ylabel="\$/lb (Soyoil)",
            xguidefontcolor=:black,              # ← xlabel 검정
            yguidefontcolor=:darkgreen,          # ← ylabel 다크그린
            ylims=SOYOIL_YLIMS,
            guidefontsize=20, tickfontsize=14,
            legend=false,
            grid=false   # 격자 중복 방지
        )

        for (key, label, color, lstyle) in soyoil_info
            xs = Float64[]
            prices = Float64[]
            for s in sorted_scenarios
                sol = solutions[s]
                isnothing(sol) && continue
                if key == :feedstock_soy_cs
                    sol.q_feedstock[:soy_cs] > 1e-6 || continue
                end
                push!(xs, get_x(s, policy_type))
                push!(prices, sol.p_f[key])
            end
            isempty(xs) && continue
            plot!(p_right, xs, prices,
                label=label, linewidth=3, color=color, linestyle=lstyle)
        end

        push!(plots_list, p)
    end

    # ── 공통 범례 ─────────────────────────────────────────────────────────
    p_legend = plot(
        legend=:top, legendcolumns=4,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=14, background_color=:white
    )
    for (_, label, color, lstyle) in corn_info
        plot!(p_legend, [NaN], [NaN],
            label=label, linewidth=3, color=color, linestyle=lstyle)
    end
    for (_, label, color, lstyle) in soyoil_info
        plot!(p_legend, [NaN], [NaN],
            label=label, linewidth=3, color=color, linestyle=lstyle)
    end

    final_plot = plot(
        plot(plots_list..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.93, 0.07]),
        size=(2200, 1600),
        plot_title="Feedstock Prices by Policy Stringency (Left: Corn, Right: Soyoil)",
        plot_titlefontsize=24, plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end

p_feedstock = plot_feedstock_prices_by_policy(
    results_extended_analysis; vlines=vlines_data)
display(p_feedstock)

# =================================================================================
# Tax Credit: x[:avi] for all p values
# =================================================================================


tax_credit_scenarios = filter(k -> startswith(String(k), "taxcredit_"), keys(results_extended_analysis.solutions))
for scen in sort(collect(tax_credit_scenarios), by=s -> begin
    p_str = split(String(s), "_")[2]
    parse(Float64, p_str) / 100
end)
    sol = results_extended_analysis.solutions[scen]
    p_val = parse(Float64, split(String(scen), "_")[2]) / 100
    x_avi = sol.x[:avi]
    println(@sprintf("%-15.2f %15.6f", p_val, x_avi))
end
# =================================================================================
# Implicit Tax by Policy Type
# =================================================================================

function plot_implicit_tax_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    scenario_groups = results_extended_analysis.scenario_groups

    fuel_info = [
        (:jet_fuel, "Jet Fuel", :black, :solid),
        (:saf_atj_conv, "Conv ATJ-SAF", :blue, :solid),
        (:saf_atj_cs, "CS ATJ-SAF", :red, :solid),
        (:saf_hefa_conv, "Conv HEFA-SAF", :green, :solid),
        (:saf_hefa_cs, "CS HEFA-SAF", :orange, :solid),
        (:saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple, :solid),
    ]

    RFS_GROUP_SAF = [:saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    x_ranges = Dict(
        :carbontax => (0.0, 500.0),
        :rfs => (0.0, 0.9),
        :lcfs => (0.0, 0.3),
        :taxcredit => (0.0, 75.0),
    )

    policy_key = Dict(
        :carbontax => :carbon_tax,
        :rfs => :rfs_avi,
        :lcfs => :lcfs,
        :taxcredit => :tax_credit,
    )

    policies = [
        (:carbontax, "Carbon Tax (\$/ton CO₂e)", "Carbon Tax", false),
        (:rfs, "RFS Aviation Mandate (θ_avi)", "RFS Aviation", false),
        (:lcfs, "LCFS Standard (σ)", "LCFS", false),
        (:taxcredit, "Tax Credit (\$/gallon)", "Tax Credit", false),
    ]

    YLIMS_CARBONTAX = (0.0, 10.0)
    YLIMS_RFS = (-5.0, 2.0)
    YLIMS_LCFS = (-10.0, 5.0)
    YLIMS_TAXCREDIT = (-55.0, 1.0)

    function get_ylims(policy_type)
        policy_type == :carbontax ? YLIMS_CARBONTAX :
        policy_type == :rfs ? YLIMS_RFS :
        policy_type == :lcfs ? YLIMS_LCFS :
        YLIMS_TAXCREDIT
    end

    function get_x(s, policy_type)
        config = EXTENDED_POLICY_MATRIX[s]
        policy_type == :carbontax ? config.t :
        policy_type == :rfs ? config.θ_avi :
        policy_type == :lcfs ? config.σ :
        config.p
    end

    label_map = Dict(
        :jet_fuel => ("Jet Fuel", :black),
        :saf_atj_conv => ("Conv ATJ-SAF", :blue),
        :saf_atj_cs => ("CS ATJ-SAF", :red),
        :saf_hefa_conv => ("Conv HEFA-SAF", :green),
        :saf_hefa_cs => ("CS HEFA-SAF", :orange),
        :saf_hefa_nonsoy => ("Non-soy HEFA-SAF", :purple),
        :all_other_saf => ("All Other SAF", RGB(0.2, 0.2, 0.2)),
    )

    plots_list = []

    for (policy_type, xlabel, title, reverse_sort) in policies
        scenario_list = scenario_groups[policy_type]
        sorted_scenarios = sort(scenario_list,
            by=s -> parse(Int, split(String(s), "_")[2]),
            rev=reverse_sort)

        ylims_cur = get_ylims(policy_type)
        xlims_cur = x_ranges[policy_type]
        it_key = policy_key[policy_type]

        p = plot(
            xlabel=xlabel,
            ylabel="Implicit Tax / Subsidy (\$/gallon)",
            title=title,
            titlefontsize=28, titlefontweight=:bold,
            legend=false,
            grid=true,
            xlims=xlims_cur,
            ylims=ylims_cur,
            left_margin=22Plots.mm,
            bottom_margin=15Plots.mm,
            right_margin=55Plots.mm,
            top_margin=10Plots.mm,
            guidefontsize=22,
            tickfontsize=18
        )

        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1.5, label="")

        series_data = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()

        if policy_type == :rfs
            individual_fuels = [:jet_fuel, :saf_atj_conv]
            individual_info = filter(x -> x[1] in individual_fuels, fuel_info)

            for (g, label, color, lstyle) in individual_info
                xs = Float64[]
                itvals = Float64[]
                for s in sorted_scenarios
                    sol = solutions[s]
                    isnothing(sol) && continue
                    !hasproperty(sol, :implicit_taxes) && continue
                    isnothing(sol.implicit_taxes) && continue
                    !haskey(sol.implicit_taxes, g) && continue
                    push!(xs, get_x(s, policy_type))
                    push!(itvals, sol.implicit_taxes[g][it_key])
                end
                isempty(xs) && continue
                series_data[g] = (xs, itvals)
                plot!(p, xs, itvals, linewidth=3.0, color=color,
                    linestyle=lstyle, label=label)
            end

            g_rep = RFS_GROUP_SAF[1]
            xs = Float64[]
            itvals = Float64[]
            for s in sorted_scenarios
                sol = solutions[s]
                isnothing(sol) && continue
                !hasproperty(sol, :implicit_taxes) && continue
                isnothing(sol.implicit_taxes) && continue
                !haskey(sol.implicit_taxes, g_rep) && continue
                push!(xs, get_x(s, policy_type))
                push!(itvals, sol.implicit_taxes[g_rep][it_key])
            end
            if !isempty(xs)
                series_data[:all_other_saf] = (xs, itvals)
                plot!(p, xs, itvals, linewidth=3.0, color=:darkgray,
                    linestyle=:solid, label="All Other SAF")
            end

        else
            for (g, label, color, lstyle) in fuel_info
                xs = Float64[]
                itvals = Float64[]
                for s in sorted_scenarios
                    sol = solutions[s]
                    isnothing(sol) && continue
                    !hasproperty(sol, :implicit_taxes) && continue
                    isnothing(sol.implicit_taxes) && continue
                    !haskey(sol.implicit_taxes, g) && continue
                    # x범위 필터링
                    xval = get_x(s, policy_type)
                    xlims_cur[1] <= xval <= xlims_cur[2] || continue
                    push!(xs, xval)
                    push!(itvals, sol.implicit_taxes[g][it_key])
                end
                isempty(xs) && continue
                series_data[g] = (xs, itvals)
                plot!(p, xs, itvals, linewidth=3.0, color=color,
                    linestyle=lstyle, label=label)
            end
        end

        # 끝점 라벨
        OVERLAP_THRESH = (ylims_cur[2] - ylims_cur[1]) * 0.05

        endpoint_vals = Dict{Symbol,Float64}()
        for (g, (xs, itvals)) in series_data
            isempty(xs) && continue
            idx = sortperm(xs)
            endpoint_vals[g] = itvals[idx[end]]
        end

        isempty(endpoint_vals) && (push!(plots_list, p); continue)

        sorted_by_y = sort(collect(endpoint_vals), by=x -> x[2])
        label_pos = Dict(g => y for (g, y) in sorted_by_y)

        for i in 2:length(sorted_by_y)
            g_prev, _ = sorted_by_y[i-1]
            g_cur, _ = sorted_by_y[i]
            if abs(label_pos[g_cur] - label_pos[g_prev]) < OVERLAP_THRESH
                label_pos[g_cur] = label_pos[g_prev] + OVERLAP_THRESH
            end
        end

        x_width = xlims_cur[2] - xlims_cur[1]

        for (g, (lbl, color)) in label_map
            haskey(label_pos, g) || continue
            ypos = label_pos[g]
            ylims_cur[1] - OVERLAP_THRESH <= ypos <= ylims_cur[2] + OVERLAP_THRESH || continue
            annotate!(p,
                xlims_cur[2] + x_width * 0.02,
                ypos,
                text(lbl, color, :left, 16))
        end

        # vlines
        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p, xval, ylims_cur[2] * 0.93,
                    text(vlabel, c, :center, 14))
            end
        end

        push!(plots_list, p)
    end
    if policy_type == :taxcredit
        x_width = xlims_cur[2] - xlims_cur[1]
        annotate!(p,
            xlims_cur[2] + x_width * 0.02,
            0.0 + (ylims_cur[2] - ylims_cur[1]) * 0.03,  # 살짝 위로
            text("Jet Fuel", :black, :left, 16))
    end

    final_plot = plot(
        plots_list...,
        layout=(2, 2),
        size=(2600, 1800),
        plot_title="Implicit Tax / Subsidy on Aviation Fuels by Policy Stringency",
        plot_titlefontsize=28,
        plot_titlefontweight=:bold,
        margin=12Plots.mm
    )

    return final_plot
end

p_implicit_tax = plot_implicit_tax_by_policy(
    results_extended_analysis; vlines=vlines_data)
display(p_implicit_tax)
# savefig(p_implicit_tax, joinpath(OUTPUT_DIR, "implicit_tax_aviation.png"))



