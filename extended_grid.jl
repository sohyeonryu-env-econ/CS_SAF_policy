# extended_grid.jl
cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "SAFModel.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
import .SAFModel: params, build_unified_model, extract_solution,
    FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS
import .SAFAnalysis: calculate_emissions_detail, calculate_implicit_taxes,
    calculate_cs_changes, calculate_ps_land_changes,
    calculate_gr_changes, calculate_environmental_benefit,
    calculate_total_welfare
using JLD2, DataFrames, Printf, Plots, JuMP, Statistics

const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"

# =================================================================================
# 0. Shared Constants & Helpers
# =================================================================================

# 4개 정책 공통 메타데이터 (policy_type, xcol, xlabel, title)
const POLICIES = [
    (:carbontax, :t, "Carbon Tax (\$/ton CO₂e)", "Carbon Tax"),
    (:rfs, :θ_avi, "RFS Aviation Mandate (θ_avi)", "RFS Aviation"),
    (:lcfs, :σ, "LCFS Standard (σ)", "LCFS"),
    (:taxcredit, :p, "Tax Credit (\$/gallon)", "Tax Credit"),
]

get_x(s, policy_type) = begin
    haskey(EXTENDED_POLICY_MATRIX, s) || return NaN
    c = EXTENDED_POLICY_MATRIX[s]
    policy_type == :carbontax ? c.t :
    policy_type == :rfs ? c.θ_avi :
    policy_type == :lcfs ? c.σ : c.p
end

sort_scenarios(list) = sort(list, by=s -> parse(Int, split(String(s), "_")[2]))

function add_vlines!(p, policy_type, vlines; y_top=nothing, annotate_y=nothing)
    isnothing(vlines) && return
    haskey(vlines, policy_type) || return
    vline_colors = [:darkred, :darkblue]
    for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
        c = vline_colors[min(j, length(vline_colors))]
        vline!(p, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
        if !isnothing(annotate_y)
            annotate!(p, xval, annotate_y, text(vlabel, c, :center, 9))
        end
    end
end

# =================================================================================
# 1. Policy Grid Setup
# =================================================================================

const POLICY_RANGES = (
    t=0:0.1:700,
    θ_avi=0:0.001:0.9,
    σ=0.0:0.0003:0.5,
    p=0:0.05:100.0
)

function create_policy_scenarios()
    scenarios = Dict()
    for t in POLICY_RANGES.t
        scenarios[Symbol("carbontax_$(round(Int,t))")] =
            (t=Float64(t), θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation)
    end
    for θ in POLICY_RANGES.θ_avi
        scenarios[Symbol("rfs_$(round(Int, θ*1000))")] =
            (t=0.0, θ_avi=Float64(θ), σ=0.0, p=0.0, carbon_tax_scope=:aviation)
    end
    for σ in POLICY_RANGES.σ
        scenarios[Symbol("lcfs_$(round(Int, σ*1000))")] =
            (t=0.0, θ_avi=0.0, σ=Float64(σ), p=0.0, carbon_tax_scope=:aviation)
    end
    for p in POLICY_RANGES.p
        scenarios[Symbol("taxcredit_$(round(Int, p*100))")] =
            (t=0.0, θ_avi=0.0, σ=0.0, p=Float64(p), carbon_tax_scope=:aviation)
    end
    scenarios[:statusquo] = (t=0.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation)
    return scenarios
end

const EXTENDED_POLICY_MATRIX = create_policy_scenarios()

# =================================================================================
# 2. Run & Welfare Analysis
# =================================================================================

function run_extended_analysis(params, policy_configs; verbose=true)
    results, solutions = Dict(), Dict()
    solved = failed = 0
    for (name, config) in policy_configs
        try
            model = build_unified_model(params, config)
            optimize!(model)
            if is_solved_and_feasible(model)
                sol = extract_solution(model, name)
                sol = merge(sol, (emissions=calculate_emissions_detail(sol, params),))
                if name != :statusquo
                    sol = merge(sol, (implicit_taxes=calculate_implicit_taxes(sol, params, config),))
                end
                results[name], solutions[name] = model, sol
                solved += 1
            else
                verbose && println("  ✗ $name: Failed")
                results[name] = solutions[name] = nothing
                failed += 1
            end
        catch e
            verbose && println("  ✗ $name: Error - $e")
            results[name] = solutions[name] = nothing
            failed += 1
        end
    end
    verbose && @printf("\nSolved: %d / %d  |  Failed: %d\n", solved, solved + failed, failed)
    return results, solutions
end

function calculate_extended_welfare_analysis(solutions, policy_configs, params; scc=190.0)
    valid = filter(p -> !isnothing(p.second), solutions)
    isempty(valid) && (println("No valid solutions!"); return nothing)
    sq = valid[:statusquo]

    println("Calculating welfare components...")
    cs = calculate_cs_changes(valid, sq, params)
    ps = calculate_ps_land_changes(valid, sq, params)
    gr = calculate_gr_changes(valid)
    env = calculate_environmental_benefit(valid, sq, scc)
    welf = calculate_total_welfare(cs, ps, gr, env)

    return (
        solutions=valid,
        cs_changes=cs,
        ps_land_changes=ps,
        gr_changes=gr,
        env_benefits=env,
        welfare_summary=welf,
        scenario_groups=(
            carbontax=[k for k in keys(valid) if startswith(String(k), "carbontax_")],
            rfs=[k for k in keys(valid) if startswith(String(k), "rfs_")],
            lcfs=[k for k in keys(valid) if startswith(String(k), "lcfs_")],
            taxcredit=[k for k in keys(valid) if startswith(String(k), "taxcredit_")],
        )
    )
end

# ── Run ──────────────────────────────────────────────────────────────────────
all_results, all_solutions = run_extended_analysis(params, EXTENDED_POLICY_MATRIX)
results_extended_analysis = calculate_extended_welfare_analysis(
    all_solutions, EXTENDED_POLICY_MATRIX, params, scc=190.0)

@save joinpath(OUTPUT_DIR, "results_extended.jld2") results_extended_analysis
println("✓ Saved results_extended.jld2")

# =================================================================================
# 3. MAC Calculation
# =================================================================================

function calculate_mac_extended(results_extended_analysis)
    welfare_summary = results_extended_analysis.welfare_summary
    solutions = results_extended_analysis.solutions
    mac_results = Dict()

    for (policy_type, scenario_list) in pairs(results_extended_analysis.scenario_groups)
        sorted = sort_scenarios(scenario_list)
        mac_data = []
        for i in 2:length(sorted)
            s_i, s_p = sorted[i], sorted[i-1]
            Δem = solutions[s_i].emissions.total - solutions[s_p].emissions.total
            abs(Δem) < 1e-6 && continue
            Δps = welfare_summary[s_i].private_surplus - welfare_summary[s_p].private_surplus
            Δsw = welfare_summary[s_i].social_welfare - welfare_summary[s_p].social_welfare
            push!(mac_data, (
                scenario=s_i, scenario_prev=s_p,
                emission=solutions[s_i].emissions.total,
                Δemission=Δem,
                mac_private=Δps / Δem, mac_social=Δsw / Δem
            ))
        end
        mac_results[policy_type] = mac_data
    end
    return mac_results
end

mac_extended = calculate_mac_extended(results_extended_analysis)
statusquo_emission = results_extended_analysis.solutions[:statusquo].emissions.total

# ── Social MAC = 0 교차점 출력 ───────────────────────────────────────────────
println("=== Social MAC = 0 교차점 ===\n")
for (policy_type, _, _, _) in POLICIES
    mac_data = mac_extended[policy_type]
    isempty(mac_data) && continue
    data_iter = policy_type in [:taxcredit, :lcfs] ? mac_data[2:end] : mac_data

    ab_vals = [statusquo_emission - d.emission for d in data_iter]
    ms_vals = [d.mac_social for d in data_iter]
    pv_vals = [EXTENDED_POLICY_MATRIX[d.scenario][
        policy_type == :carbontax ? :t :
        policy_type == :rfs ? :θ_avi :
        policy_type == :lcfs ? :σ : :p] for d in data_iter]

    idx = sortperm(ab_vals)
    ab, ms, pv = ab_vals[idx], ms_vals[idx], pv_vals[idx]

    crossings = [(ab[i] + (-ms[i] / (ms[i+1] - ms[i])) * (ab[i+1] - ab[i]),
        pv[i] + (-ms[i] / (ms[i+1] - ms[i])) * (pv[i+1] - pv[i]))
                 for i in 1:length(ms)-1 if ms[i] * ms[i+1] <= 0]

    println("$policy_type:")
    if isempty(crossings)
        println("  교차점 없음 (MAC 범위: $(round(minimum(ms),digits=1)) ~ $(round(maximum(ms),digits=1)))")
    else
        for (xa, xp) in crossings
            @printf("  abatement = %.6f B ton CO₂  |  stringency = %.4f\n", xa, xp)
        end
    end
    println()
end

# =================================================================================
# 4. Load vlines data
# =================================================================================

@load joinpath(OUTPUT_DIR, "results_equivalent_emissions.jld2") equivalent_emission_policies
ep_3B = equivalent_emission_policies
@load joinpath(OUTPUT_DIR, "results_equivalent_emissions_5.jld2") equivalent_emission_policies
ep_5B = equivalent_emission_policies

statusquo_em = results_extended_analysis.solutions[:statusquo].emissions.total
abatement_3B = statusquo_em - ep_3B[:rfs].actual_emission
abatement_5B = statusquo_em - ep_5B[:rfs].actual_emission
vlines_abatement = [(abatement_3B, "3B"), (abatement_5B, "5B")]

vlines_data = Dict(pt => [(ep_3B[pt].policy_value, "3B"), (ep_5B[pt].policy_value, "5B")]
                   for pt in [:carbontax, :rfs, :lcfs, :taxcredit])

# =================================================================================
# 5. MAC Plot
# =================================================================================

function plot_mac_comparison_simple(results_extended_analysis, mac_extended;
    vlines_abatement=nothing, y_max=2200.0, y_min=-250.0, fig_size=(1800, 1400))

    sq_em = results_extended_analysis.solutions[:statusquo].emissions.total
    max_ab = 0.105
    bg = RGB(0.96, 0.96, 0.94)
    policy_colors = [(:carbontax, :blue), (:rfs, :red), (:lcfs, :green), (:taxcredit, :purple)]
    policy_labels = Dict(:carbontax => "Carbon Tax", :rfs => "RFS Aviation",
        :lcfs => "LCFS", :taxcredit => "Tax Credit")

    # collect plot data
    plot_data = Dict()
    for (pt, _) in policy_colors
        data = pt in [:taxcredit, :lcfs] ? mac_extended[pt][2:end] : mac_extended[pt]
        ab = [sq_em - d.emission for d in data if sq_em - d.emission <= max_ab]
        prv = [d.mac_private for d in data if sq_em - d.emission <= max_ab]
        soc = [d.mac_social for d in data if sq_em - d.emission <= max_ab]
        isempty(ab) && continue
        idx = sortperm(ab)
        plot_data[pt] = (ab=ab[idx], private=prv[idx], social=soc[idx])
    end

    function make_panel(key, title, show_y)
        p = plot(title=title, titlefontsize=18, titlefontweight=:bold,
            ylabel=show_y ? "MAC (\$/ton CO₂)" : "",
            xlabel="Cumulative Abatement (Billion tons CO₂)",
            legend=false, grid=true,
            xlims=(0, max_ab), ylims=(y_min, y_max),
            yticks=collect(-200:200.0:y_max),
            left_margin=show_y ? 22Plots.mm : 5Plots.mm,
            right_margin=8Plots.mm, bottom_margin=12Plots.mm, top_margin=8Plots.mm,
            guidefontsize=20, tickfontsize=15,
            background_color_inside=bg, background_color=:white)
        for (pt, color) in policy_colors
            haskey(plot_data, pt) || continue
            d = plot_data[pt]
            plot!(p, d.ab, key == :private ? d.private : d.social, linewidth=2.5, color=color)
        end
        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=2.0, label="", alpha=0.8)
        if !isnothing(vlines_abatement)
            for (j, (xval, vlabel)) in enumerate(vlines_abatement)
                c = [:darkred, :darkblue][min(j, 2)]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p, xval, y_max * 0.97, text(vlabel, c, :center, 9))
            end
        end
        return p
    end

    p_leg = plot(legend=:top, legendcolumns=4, grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none, legendfontsize=12, background_color=:white)
    for (pt, color) in policy_colors
        plot!(p_leg, [NaN], [NaN], label=policy_labels[pt], linewidth=3, color=color)
    end

    return plot(plot(make_panel(:private, "Private MAC", true), make_panel(:social, "Social MAC", false), layout=(1, 2)),
        p_leg, layout=grid(2, 1, heights=[0.95, 0.05]),
        size=fig_size, plot_title="",
        plot_titlefontsize=25, plot_titlefontweight=:bold, background_color=:white)
end

display(plot_mac_comparison_simple(results_extended_analysis, mac_extended;
    vlines_abatement=vlines_abatement, y_max=2200.0, y_min=-250.0))

# =================================================================================
# 6. Results DataFrame
# =================================================================================

function results_to_dataframe(extended_analysis, policy_configs)
    solutions = extended_analysis.solutions
    welfare_summary = extended_analysis.welfare_summary

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    SECTORS = [:avi, :gas, :die, :corn, :soyoil, :soymeal]

    df = DataFrame(scenario=String[], policy_type=String[], t=Float64[], θ_avi=Float64[], σ=Float64[], p=Float64[])
    for f in vcat(AVIATION_FUELS, ROAD_FUELS)
        df[!, Symbol("q_$f")] = Float64[]
    end
    for s in SECTORS
        df[!, Symbol("x_$s")] = Float64[]
    end
    for col in [:p_avi, :p_gas, :p_die, :r_land,
        :emission_avi, :emission_road, :emission_food, :emission_total,
        :cs_change, :ps_land_change, :gr_change, :env_benefit, :private_surplus, :social_welfare]
        df[!, col] = Float64[]
    end

    for (name, sol) in solutions
        isnothing(sol) && continue
        config = policy_configs[name]
        str = String(name)
        ptype = occursin("carbontax", str) ? "carbontax" :
                occursin("rfs", str) ? "rfs" :
                occursin("lcfs", str) ? "lcfs" :
                occursin("taxcredit", str) ? "taxcredit" : "statusquo"

        row = Any[str, ptype, config.t, config.θ_avi, config.σ, config.p]
        append!(row, [sol.q[f] for f in vcat(AVIATION_FUELS, ROAD_FUELS)])
        append!(row, [sol.x[s] for s in SECTORS])
        append!(row, [sol.p_c[:avi], sol.p_c[:gas], sol.p_c[:die], sol.duals.r_land])
        append!(row, [sol.emissions.aviation * 1000, sol.emissions.road * 1000,
            sol.emissions.food * 1000, sol.emissions.total * 1000])
        if name == :statusquo
            append!(row, zeros(6))
        else
            w = welfare_summary[name]
            append!(row, [w.cs_change, w.ps_land_change, w.gr_change,
                w.env_benefit, w.private_surplus, w.social_welfare])
        end
        push!(df, row)
    end
    return df
end

results_df = results_to_dataframe(results_extended_analysis, EXTENDED_POLICY_MATRIX)
#@save joinpath(OUTPUT_DIR, "extended_policy_results.jld2") results_df

# =================================================================================
# 7. Plot Functions
# =================================================================================

# ── 공통 legend 패널 생성 ────────────────────────────────────────────────────
function make_legend_panel(items; ncols=length(items), fontsize=14)
    p = plot(legend=:top, legendcolumns=ncols, grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none, legendfontsize=fontsize)
    for (label, color, args...) in items
        kw = isempty(args) ? () : args[1]
        plot!(p, [NaN], [NaN], label=label, linewidth=3, color=color)
    end
    return p
end

# ── Stacked fuel production ──────────────────────────────────────────────────
function plot_fuel_production_stacked(results_df, fuel_config; vlines=nothing)
    plots = []
    for (policy_type, xcol, xlabel, title) in POLICIES
        df = sort(filter(r -> r.policy_type == String(policy_type), results_df), xcol)
        x_vals = df[!, xcol]
        x_min, x_max = extrema(x_vals)

        #biofuel_sorted = sort(fuel_config.biofuel_types, by=x -> mean(df[!, x[1]]), rev=true)
        biofuel_sorted = fuel_config.biofuel_types  # 이렇게 하면 정의된 순서대로 깔림

        p = plot(xlabel=xlabel, ylabel="Quantity (billion gallons)", title=title,
            titlefontsize=25, titlefontweight=:bold, legend=false, grid=true,
            xlims=(policy_type == :lcfs ? (0.0, 0.2) : (x_min, x_max)), ylims=fuel_config.ylims,
            margin=10Plots.mm, guidefontsize=25, left_margin=15Plots.mm,
            bottom_margin=15Plots.mm, tickfontsize=20, labelfontsize=23)

        main_vals = df[!, fuel_config.main_fuel]
        plot!(p, x_vals, main_vals, fillrange=fuel_config.ylims[1],
            fillalpha=0.7, fillcolor=:lightgray, linewidth=1.5, color=:black, label="")

        cumsum_vals = copy(main_vals)
        for (col, _, color) in biofuel_sorted
            col_vals = df[!, col]
            new_cum = similar(cumsum_vals)
            for i in 1:length(cumsum_vals)
                # 0인 값은 건너뛰기 (NaN으로 처리)
                if col_vals[i] < 1e-10
                    new_cum[i] = NaN
                else
                    new_cum[i] = cumsum_vals[i] + col_vals[i]
                end
            end
            # NaN이 아닌 부분만 플롯
            mask = .!isnan.(new_cum)
            if any(mask)
                plot!(p, x_vals[mask], new_cum[mask], fillrange=cumsum_vals[mask],
                    fillalpha=0.7, fillcolor=color, linewidth=1.5, color=color, label="")
            end
            # cumsum 업데이트 (0인 경우는 이전값 유지)
            cumsum_vals = [col_vals[i] < 1e-10 ? cumsum_vals[i] : new_cum[i] for i in 1:length(new_cum)]
        end
        add_vlines!(p, policy_type, vlines; annotate_y=fuel_config.ylims[2] * 0.97)
        push!(plots, p)
    end

    leg_items = vcat([(fuel_config.main_fuel_label, :black)],
        [(lbl, color) for (_, lbl, color) in fuel_config.biofuel_types])
    return plot(plot(plots..., layout=(2, 2)),
        make_legend_panel(leg_items, ncols=fuel_config.legendcolumns, fontsize=20),
        layout=grid(2, 1, heights=[0.95, 0.05]), size=(2500, 1800),
        plot_title="", plot_titlefontsize=30,
        plot_titlefontweight=:bold, margin=10Plots.mm)
end

# ── Food products ────────────────────────────────────────────────────────────
function plot_food_products_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions

    corn_plots = []
    oil_plots = []
    meal_plots = []

    for (policy_type, _, xlabel, title) in POLICIES
        scenario_list = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xs = [get_x(s, policy_type) for s in scenario_list]
        ddgs = [0.092 * solutions[s].q[:ethanol] +
                0.159 * (solutions[s].q[:saf_atj_conv] + solutions[s].q[:saf_atj_cs])
                for s in scenario_list]
        corn_t = [solutions[s].x[:corn] for s in scenario_list]
        soy_oil = [solutions[s].x[:soyoil] for s in scenario_list]
        soy_meal = [solutions[s].x[:soymeal] for s in scenario_list]

        common_kw = (xlabel=xlabel, titlefontsize=25, titlefontweight=:bold,
            legend=false, grid=true, xlims=extrema(xs),
            left_margin=20Plots.mm, bottom_margin=10Plots.mm,
            guidefontsize=20, tickfontsize=18)

        p_c = plot(; title=title, ylabel="billion bushels", ylims=(0, 15), common_kw...)
        plot!(p_c, xs, corn_t .- ddgs, linewidth=4, color=:orange)
        plot!(p_c, xs, corn_t, linewidth=4, color=:red)
        add_vlines!(p_c, policy_type, vlines; annotate_y=14.5)
        push!(corn_plots, p_c)

        p_o = plot(; title=title, ylabel="billion lbs", ylims=(0, 15), common_kw...)
        plot!(p_o, xs, soy_oil, linewidth=4, color=:darkgreen)
        add_vlines!(p_o, policy_type, vlines; annotate_y=14.5)
        push!(oil_plots, p_o)

        p_m = plot(; title=title, ylabel="MMT", ylims=(0, 80), common_kw...)
        plot!(p_m, xs, soy_meal, linewidth=4, color=:brown)
        add_vlines!(p_m, policy_type, vlines; annotate_y=77.0)
        push!(meal_plots, p_m)
    end

    wrap(plots, title, leg_items) = plot(
        plot(plots..., layout=(2, 2)),
        make_legend_panel(leg_items, fontsize=18),
        layout=grid(2, 1, heights=[0.95, 0.05]), size=(2400, 1700),
        plot_title=title, plot_titlefontsize=25, plot_titlefontweight=:bold, margin=8Plots.mm)

    return (wrap(corn_plots, "",
            [("Corn for food", :orange), ("Total Corn (+ DDGS)", :red)]),
        wrap(oil_plots, "",
            [("Soybean Oil", :darkgreen)]),
        wrap(meal_plots, "",
            [("Soybean Meal", :brown)]))
end

# ── Welfare summary ──────────────────────────────────────────────────────────
function plot_welfare_summary_by_policy(results_extended_analysis; vlines=nothing)
    welfare_summary = results_extended_analysis.welfare_summary
    p = plot(layout=(2, 2), size=(1400, 1000),
        plot_title="",
        left_margin=10Plots.mm, bottom_margin=5Plots.mm)

    for (idx, (policy_type, _, xlabel, title)) in enumerate(POLICIES)
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xs = [get_x(s, policy_type) for s in sorted if haskey(welfare_summary, s)]
        ws = [welfare_summary[s] for s in sorted if haskey(welfare_summary, s)]

        plot!(p[idx], xs, [w.cs_change for w in ws], label="CS", linewidth=2, color=:steelblue,
            xlabel=xlabel, ylabel="Welfare Change (B\$)", title=title,
            titlefontsize=16, titlefontweight=:bold, legend=:best, legendfontsize=10)
        plot!(p[idx], xs, [w.ps_land_change for w in ws], label="PS", linewidth=2, color=:orange)
        policy_type in [:carbontax, :taxcredit] &&
            plot!(p[idx], xs, [w.gr_change for w in ws], label="Govt Revenue", linewidth=2, color=:green)
        plot!(p[idx], xs, [w.private_surplus for w in ws], label="Private", linewidth=3, linestyle=:dot, color=:purple)
        plot!(p[idx], xs, [w.social_welfare for w in ws], label="Social", linewidth=3, linestyle=:dot, color=:red)
        hline!(p[idx], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)

        y_top = maximum(vcat([w.cs_change, w.ps_land_change, w.private_surplus, w.social_welfare] for w in ws)...)
        add_vlines!(p[idx], policy_type, vlines; annotate_y=y_top * 0.95)
    end
    return p
end

# ── Emissions stacked (broken axis) ─────────────────────────────────────────
function plot_emissions_stacked_broken_axis(results_extended_analysis;
    vlines=nothing, break_point=1.2, y_max=2.6)

    solutions = results_extended_analysis.solutions
    GAS_FUELS = [:gasoline, :ethanol]
    DIESEL_FUELS = [:diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    sector_info = [(:gas, "Road (Gasoline)", :steelblue), (:die, "Road (Diesel)", :orange),
        (:food, "Food", :green), (:avi, "Aviation", :red)]

    function get_sector_em(sol)
        (food=sol.emissions.food,
            gas=sum(sol.emissions.by_fuel[g] for g in GAS_FUELS),
            die=sum(sol.emissions.by_fuel[g] for g in DIESEL_FUELS),
            avi=sol.emissions.aviation)
    end

    function draw_stacked!(p, xs, vals)
        cum = zeros(length(xs))
        for (key, _, color) in sector_info
            new_cum = cum .+ vals[key]
            plot!(p, xs, new_cum, fillrange=cum, fillalpha=0.75, fillcolor=color,
                linewidth=1.2, color=color, label=string(key))
            cum = new_cum
        end
    end

    top_plots, bot_plots = [], []

    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xs = [get_x(s, policy_type) for s in sorted if !isnothing(solutions[s])]
        vals = Dict(k => [getfield(get_sector_em(solutions[s]), k)
                          for s in sorted if !isnothing(solutions[s])]
                    for (k, _, _) in sector_info)
        xl = extrema(xs)

        p_top = plot(title=title, titlefontsize=16, titlefontweight=:bold,
            ylabel="Billion ton CO₂e", legend=false, grid=true,
            xlims=xl, ylims=(break_point, y_max), xticks=:none,
            bottom_margin=-4Plots.mm, top_margin=8Plots.mm, left_margin=18Plots.mm,
            guidefontsize=12, tickfontsize=10)
        draw_stacked!(p_top, xs, vals)
        hline!(p_top, [break_point], color=:white, linewidth=6, label="")
        add_vlines!(p_top, policy_type, vlines; annotate_y=y_max * 0.97)
        push!(top_plots, p_top)

        p_bot = plot(xlabel=xlabel, legend=false, grid=true,
            xlims=xl, ylims=(0.0, break_point), yticks=[0.0],
            top_margin=-4Plots.mm, bottom_margin=12Plots.mm, left_margin=18Plots.mm,
            guidefontsize=12, tickfontsize=10)
        draw_stacked!(p_bot, xs, vals)
        hline!(p_bot, [break_point], color=:white, linewidth=6, label="")
        add_vlines!(p_bot, policy_type, vlines)
        push!(bot_plots, p_bot)
    end

    p_leg = make_legend_panel([(lbl, color) for (_, lbl, color) in sector_info], ncols=4, fontsize=13)
    return plot(plot(vcat(top_plots, bot_plots)..., layout=grid(2, 4, heights=[0.75, 0.25])),
        p_leg, layout=grid(2, 1, heights=[0.93, 0.07]), size=(2200, 1300),
        plot_title="",
        plot_titlefontsize=20, plot_titlefontweight=:bold, margin=8Plots.mm)
end

# ── Generic fuel price plotter ───────────────────────────────────────────────
function plot_prices_by_policy(results_extended_analysis, fuel_info, ppu_fn,
    left_ylims, right_fn, right_ylims, right_ylabel, right_color, right_label,
    plot_title; vlines=nothing)

    solutions = results_extended_analysis.solutions
    plots_list = []

    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])

        p = plot(xlabel=xlabel, ylabel="\$/gallon", title="Title",
            titlefontsize=22, titlefontweight=:bold, legend=false, grid=true,
            ylims=left_ylims, left_margin=18Plots.mm, bottom_margin=12Plots.mm,
            right_margin=22Plots.mm, top_margin=8Plots.mm,
            guidefontsize=16, tickfontsize=14)

        for (g, label, color, always_show) in fuel_info
            xs, ps = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                (always_show || sol.q[g] > 1e-6) || continue
                push!(xs, get_x(s, policy_type))
                push!(ps, ppu_fn(sol, g))
            end
            isempty(xs) && continue
            plot!(p, xs, ps, label=label, linewidth=3, color=color)
        end
        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1, label="")

        xs_r, ps_r = Float64[], Float64[]
        for s in sorted
            sol = solutions[s]
            isnothing(sol) && continue
            push!(xs_r, get_x(s, policy_type))
            push!(ps_r, right_fn(sol))
        end
        if !isempty(xs_r)
            pr = Plots.twinx(p)
            plot!(pr, xs_r, ps_r, label=right_label, linewidth=3,
                color=right_color, linestyle=:dash, ylabel=right_ylabel,
                ylims=right_ylims, guidefontsize=16, tickfontsize=14, legend=false)
        end
        add_vlines!(p, policy_type, vlines; annotate_y=left_ylims[2] * 0.96)
        push!(plots_list, p)
    end

    leg_items = [(lbl, color) for (_, lbl, color, _) in fuel_info]
    push!(leg_items, (right_label, right_color))
    return plot(plot(plots_list..., layout=(2, 2)),
        make_legend_panel(leg_items, ncols=4, fontsize=14),
        layout=grid(2, 1, heights=[0.93, 0.07]), size=(2200, 1600),
        plot_title=plot_title, plot_titlefontsize=22,
        plot_titlefontweight=:bold, margin=10Plots.mm)
end

# ── Feedstock prices ─────────────────────────────────────────────────────────
function plot_feedstock_prices_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions

    corn_info = [(:feedstock_corn_n, "Corn (Conv)", :darkorange, :solid),
        (:feedstock_corn_cs, "Corn (CS)", :darkorange, :dash)]
    soyoil_info = [(:feedstock_soy_n, "Soyoil (Conv)", :darkgreen, :solid),
        (:feedstock_soy_cs, "Soyoil (CS)", :darkgreen, :dash)]

    valid_sols = [sol for sol in values(solutions) if !isnothing(sol)]
    CORN_YLIMS = (floor(minimum(s.p_f[k] for s in valid_sols for (k, _, _, _) in corn_info)),
        ceil(maximum(s.p_f[k] for s in valid_sols for (k, _, _, _) in corn_info)))
    SOYOIL_YLIMS = (floor(minimum(s.p_f[k] for s in valid_sols for (k, _, _, _) in soyoil_info)),
        ceil(maximum(s.p_f[k] for s in valid_sols for (k, _, _, _) in soyoil_info)))

    plots_list = []
    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])

        p = plot(xlabel=xlabel, ylabel="\$/bushel (Corn)", title=title,
            yguidefontcolor=:darkorange,
            titlefontsize=22, titlefontweight=:bold, legend=false, grid=true,
            ylims=CORN_YLIMS, left_margin=18Plots.mm, bottom_margin=12Plots.mm,
            right_margin=22Plots.mm, top_margin=8Plots.mm,
            guidefontsize=20, tickfontsize=14)

        for (key, _, color, lstyle) in corn_info
            xs, ps = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                key == :feedstock_corn_cs && sol.q_feedstock[:corn_cs] <= 1e-6 && continue
                push!(xs, get_x(s, policy_type))
                push!(ps, sol.p_f[key])
            end
            isempty(xs) && continue
            plot!(p, xs, ps, linewidth=3, color=color, linestyle=lstyle)
        end
        add_vlines!(p, policy_type, vlines; annotate_y=CORN_YLIMS[2] * 0.96)

        pr = Plots.twinx(p)
        plot!(pr, ylabel="\$/lb (Soyoil)", yguidefontcolor=:darkgreen,
            ylims=SOYOIL_YLIMS, guidefontsize=20, tickfontsize=14, legend=false, grid=false)
        for (key, _, color, lstyle) in soyoil_info
            xs, ps = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                key == :feedstock_soy_cs && sol.q_feedstock[:soy_cs] <= 1e-6 && continue
                push!(xs, get_x(s, policy_type))
                push!(ps, sol.p_f[key])
            end
            isempty(xs) && continue
            plot!(pr, xs, ps, linewidth=3, color=color, linestyle=lstyle)
        end
        push!(plots_list, p)
    end

    leg_items = vcat([(lbl, color) for (_, lbl, color, _) in corn_info],
        [(lbl, color) for (_, lbl, color, _) in soyoil_info])
    return plot(plot(plots_list..., layout=(2, 2)),
        make_legend_panel(leg_items, ncols=4, fontsize=14),
        layout=grid(2, 1, heights=[0.93, 0.07]), size=(2200, 1600),
        plot_title="",
        plot_titlefontsize=24, plot_titlefontweight=:bold, margin=10Plots.mm)
end

# ── Implicit tax ─────────────────────────────────────────────────────────────
function plot_implicit_tax_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions

    fuel_info = [(:jet_fuel, "Jet Fuel", :black, :solid), (:saf_atj_conv, "Conv ATJ-SAF", :blue, :solid),
        (:saf_atj_cs, "CS ATJ-SAF", :red, :solid), (:saf_hefa_conv, "Conv HEFA-SAF", :green, :solid),
        (:saf_hefa_cs, "CS HEFA-SAF", :orange, :solid), (:saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple, :solid)]
    RFS_GROUP = [:saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    xlims_map = Dict(:carbontax => (0.0, 700.0), :rfs => (0.0, 0.9), :lcfs => (0.0, 0.5), :taxcredit => (0.0, 75.0))
    ylims_map = Dict(:carbontax => (0.0, 10.0), :rfs => (-5.0, 2.0), :lcfs => (-10.0, 10.0), :taxcredit => (-55.0, 1.0))
    it_key_map = Dict(:carbontax => :carbon_tax, :rfs => :rfs_avi, :lcfs => :lcfs, :taxcredit => :tax_credit)

    plots_list = []
    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xlims_cur = xlims_map[policy_type]
        ylims_cur = ylims_map[policy_type]
        it_key = it_key_map[policy_type]
        THRESH = (ylims_cur[2] - ylims_cur[1]) * 0.05

        p = plot(xlabel=xlabel, ylabel="Implicit Tax/Subsidy (\$/gallon)", title=title,
            titlefontsize=28, titlefontweight=:bold, legend=false, grid=true,
            xlims=xlims_cur, ylims=ylims_cur,
            left_margin=22Plots.mm, bottom_margin=15Plots.mm,
            right_margin=55Plots.mm, top_margin=10Plots.mm,
            guidefontsize=22, tickfontsize=18)
        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1.5, label="")

        series_data = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()

        fuels_to_plot = policy_type == :rfs ?
                        filter(x -> x[1] in [:jet_fuel, :saf_atj_conv], fuel_info) : fuel_info

        for (g, _, color, lstyle) in fuels_to_plot
            xs, its = Float64[], Float64[]
            for s in sorted
                !haskey(solutions, s) && continue
                sol = solutions[s]
                isnothing(sol) && continue
                !hasproperty(sol, :implicit_taxes) && continue
                isnothing(sol.implicit_taxes) && continue
                !haskey(sol.implicit_taxes, g) && continue
                xval = get_x(s, policy_type)
                isnothing(xval) && continue
                xlims_cur[1] <= xval <= xlims_cur[2] || continue
                push!(xs, xval)
                push!(its, sol.implicit_taxes[g][it_key])
            end
            isempty(xs) && continue
            series_data[g] = (xs, its)
            plot!(p, xs, its, linewidth=3.0, color=color, linestyle=lstyle)
        end

        # RFS: 나머지 SAF 대표값
        if policy_type == :rfs
            g_rep = RFS_GROUP[1]
            xs, its = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                !hasproperty(sol, :implicit_taxes) && continue
                isnothing(sol.implicit_taxes) && continue
                !haskey(sol.implicit_taxes, g_rep) && continue
                push!(xs, get_x(s, policy_type))
                push!(its, sol.implicit_taxes[g_rep][it_key])
            end
            if !isempty(xs)
                series_data[:all_other_saf] = (xs, its)
                plot!(p, xs, its, linewidth=3.0, color=:darkgray)
            end
        end

        # 끝점 라벨 (overlap 보정)
        endpoint_vals = Dict(g => itvals[sortperm(xs)[end]]
                             for (g, (xs, itvals)) in series_data if !isempty(xs))
        if !isempty(endpoint_vals)
            sorted_by_y = sort(collect(endpoint_vals), by=x -> x[2])
            label_pos = Dict(g => y for (g, y) in sorted_by_y)
            for i in 2:length(sorted_by_y)
                gp, gc = sorted_by_y[i-1][1], sorted_by_y[i][1]
                abs(label_pos[gc] - label_pos[gp]) < THRESH && (label_pos[gc] = label_pos[gp] + THRESH)
            end
            label_map = Dict(:jet_fuel => ("Jet Fuel", :black), :saf_atj_conv => ("Conv ATJ-SAF", :blue),
                :saf_atj_cs => ("CS ATJ-SAF", :red), :saf_hefa_conv => ("Conv HEFA-SAF", :green),
                :saf_hefa_cs => ("CS HEFA-SAF", :orange), :saf_hefa_nonsoy => ("Non-soy HEFA-SAF", :purple),
                :all_other_saf => ("All Other SAF", RGB(0.2, 0.2, 0.2)))
            xw = xlims_cur[2] - xlims_cur[1]
            for (g, (lbl, color)) in label_map
                haskey(label_pos, g) || continue
                yp = label_pos[g]
                ylims_cur[1] - THRESH <= yp <= ylims_cur[2] + THRESH || continue
                annotate!(p, xlims_cur[2] + xw * 0.02, yp, text(lbl, color, :left, 16))
            end
        end
        add_vlines!(p, policy_type, vlines; annotate_y=ylims_cur[2] * 0.93)
        push!(plots_list, p)
    end

    return plot(
        plots_list...,
        layout=(2, 2),
        size=(2600, 1800),
        #plot_title="Implicit Tax / Subsidy on Aviation Fuels by Policy Stringency",
        #plot_titlefontsize=28, plot_titlefontweight=:bold, margin=12Plots.mm
    )
end
display(plot_implicit_tax_by_policy(results_extended_analysis; vlines=vlines_data))

# =================================================================================
# 8. Generate & Display All Plots
# =================================================================================

# ── Aviation/Gasoline/Diesel stacked quantity plots ──────────────────────────
aviation_config = (
    main_fuel=:q_jet_fuel, main_fuel_label="Jet Fuel", ylims=(0, 22),
    biofuel_types=[(:q_saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple),
        (:q_saf_hefa_cs, "Climate-Smart HEFA-SAF", :orange),
        (:q_saf_hefa_conv, "Conventional HEFA-SAF", :green),
        (:q_saf_atj_cs, "Climate-Smart ATJ-SAF", :red),
        (:q_saf_atj_conv, "Conventional ATJ-SAF", :blue)],
    plot_title="Aviation Fuel Production by Policy Stringency", legendcolumns=3)

gasoline_config = (
    main_fuel=:q_gasoline, main_fuel_label="Gasoline", ylims=(0, 200),
    biofuel_types=[(:q_ethanol, "Ethanol", :red)],
    plot_title="Road Gasoline Fuel Production by Policy Stringency", legendcolumns=2)

diesel_config = (
    main_fuel=:q_diesel, main_fuel_label="Diesel", ylims=(30, 55),
    biofuel_types=[(:q_rd_soy, "Soy RD", :red), (:q_rd_nonsoy, "Non-soy RD", :green),
        (:q_biodiesel_soy, "Soy Biodiesel", :blue), (:q_biodiesel_nonsoy, "Non-soy Biodiesel", :purple)],
    plot_title="Diesel Fuel Production by Policy Stringency", legendcolumns=3)

display(plot_fuel_production_stacked(results_df, aviation_config; vlines=vlines_data))
display(plot_fuel_production_stacked(results_df, gasoline_config; vlines=vlines_data))
display(plot_fuel_production_stacked(results_df, diesel_config; vlines=vlines_data))

# ── Food ─────────────────────────────────────────────────────────────────────
p_corn, p_oil, p_meal = plot_food_products_by_policy(results_extended_analysis; vlines=vlines_data)
display(p_corn);
display(p_oil);
display(p_meal);

# ── Welfare & Emissions ───────────────────────────────────────────────────────
display(plot_welfare_summary_by_policy(results_extended_analysis; vlines=vlines_data))
display(plot_emissions_stacked_broken_axis(results_extended_analysis;
    vlines=vlines_data, break_point=1.2, y_max=2.6))

# ── Prices ───────────────────────────────────────────────────────────────────
r_e = params.coeff.r;
β = params.coeff.beta;

display(plot_prices_by_policy(results_extended_analysis,
    [(:jet_fuel, "Jet Fuel", :black, true), (:saf_atj_conv, "ATJ(Conv)", :blue, false),
        (:saf_atj_cs, "ATJ(CS)", :dodgerblue, false), (:saf_hefa_conv, "HEFA(Conv)", :green, false),
        (:saf_hefa_cs, "HEFA(CS)", :limegreen, false), (:saf_hefa_nonsoy, "HEFA(Non-soy)", :purple, false)],
    (sol, g) -> r_e[:jet_fuel] * (g == :jet_fuel ? 1.0 : β[(:saf, :jet_fuel)]) * sol.p_c[:avi],
    (0, 15), sol -> sol.p_c[:avi], (0.0, 0.3), "\$/aviation mile", :red, "Aviation Mile Price",
    ""; vlines=vlines_data))

display(plot_prices_by_policy(results_extended_analysis,
    [(:gasoline, "Gasoline", :black, true), (:ethanol, "Ethanol", :red, false)],
    (sol, g) -> r_e[:gasoline] * (g == :gasoline ? 1.0 : β[(:ethanol, :gasoline)]) * sol.p_c[:gas],
    (0, 4), sol -> sol.p_c[:gas], (0.0, 1.0), "\$/gasoline mile", :orange, "Gasoline Consumer Price",
    ""; vlines=vlines_data))

display(plot_prices_by_policy(results_extended_analysis,
    [(:diesel, "Diesel", :black, true), (:rd_soy, "Soy RD", :red, false),
        (:rd_nonsoy, "Non-Soy RD", :orange, false), (:biodiesel_soy, "Soy Biodiesel", :blue, false),
        (:biodiesel_nonsoy, "Non-Soy Biodiesel", :purple, false)],
    (sol, g) -> r_e[:diesel] * (g == :diesel ? 1.0 : g in (:biodiesel_soy, :biodiesel_nonsoy) ?
                                β[(:biodiesel, :diesel)] : β[(:rd, :diesel)]) * sol.p_c[:die],
    (0, 6.5), sol -> sol.p_c[:die], (0.0, 2.0), "\$/diesel mile", :brown, "Diesel Consumer Price",
    ""; vlines=vlines_data))

display(plot_feedstock_prices_by_policy(results_extended_analysis; vlines=vlines_data))

# =================================================================================
# 9. Sensitivity Analysis (c0 ATJ/HEFA)
# =================================================================================

c0_scenarios = [(atj=0.32, hefa=0.44, label="Low"),
    (atj=2.3, hefa=1.145, label="Kumar TEA"),
    (atj=4.91, hefa=1.56, label="High")]

sensitivity_extended = Dict()
for c0 in c0_scenarios
    println("\nRunning: c0_atj=$(c0.atj), c0_hefa=$(c0.hefa)")
    new_fuel = deepcopy(params.supply.fuel)
    new_fuel[:saf_atj_shared] = merge(new_fuel[:saf_atj_shared], (c0=c0.atj,))
    new_fuel[:saf_hefa_shared] = merge(new_fuel[:saf_hefa_shared], (c0=c0.hefa,))
    new_params = merge(params, (supply=(fuel=new_fuel, land=params.supply.land),))
    _, sols_s = run_extended_analysis(new_params, EXTENDED_POLICY_MATRIX, verbose=false)
    sensitivity_extended[c0.label] = calculate_extended_welfare_analysis(
        sols_s, EXTENDED_POLICY_MATRIX, new_params, scc=190.0)
end

@save joinpath(OUTPUT_DIR, "sensitivity_extended_c0.jld2") sensitivity_extended c0_scenarios
println("✓ Saved sensitivity_extended_c0.jld2")

function plot_saf_stacked_sensitivity(sensitivity_extended, c0_scenarios, fuel_config)
    SAF_COLS = [:q_saf_atj_conv, :q_saf_atj_cs, :q_saf_hefa_conv, :q_saf_hefa_cs, :q_saf_hefa_nonsoy]
    all_plots, label_plots = [], []

    for c0 in c0_scenarios
        df_s = results_to_dataframe(sensitivity_extended[c0.label], EXTENDED_POLICY_MATRIX)
        row_idx = findfirst(c -> c.label == c0.label, c0_scenarios)

        for (col_idx, (policy_type, xcol, xlabel, pol_title)) in enumerate(POLICIES)
            df = sort(filter(r -> r.policy_type == String(policy_type), df_s), xcol)
            xs = df[!, xcol]
            biofuel_sorted = sort(fuel_config.biofuel_types, by=x -> mean(df[!, x[1]]), rev=true)

            p = plot(
                xlabel=row_idx == length(c0_scenarios) ? xlabel : "",
                ylabel=col_idx == 1 ? "(billion gal)" : "",
                title=row_idx == 1 ? pol_title : "",
                titlefontsize=20, titlefontweight=:bold, legend=false, grid=true,
                xlims=extrema(xs), ylims=fuel_config.ylims, margin=5Plots.mm, guidefontsize=10)

            main_vals = df[!, fuel_config.main_fuel]
            plot!(p, xs, main_vals, fillrange=fuel_config.ylims[1],
                fillalpha=0.7, fillcolor=:lightgray, linewidth=1.5, color=:black)
            cum = copy(main_vals)
            for (col, _, color) in biofuel_sorted
                new_cum = cum .+ df[!, col]
                plot!(p, xs, new_cum, fillrange=cum, fillalpha=0.7, fillcolor=color,
                    linewidth=1.5, color=color)
                cum = new_cum
            end
            push!(all_plots, p)
        end

        p_lbl = plot(grid=false, showaxis=false, ticks=false, xlims=(0, 1), ylims=(0, 1), framestyle=:none)
        annotate!(p_lbl, 0.5, 0.5, text(c0.label, :black, :center, 20, rotation=90))
        push!(label_plots, p_lbl)
    end

    combined = []
    for (i, lp) in enumerate(label_plots)
        push!(combined, lp)
        append!(combined, all_plots[(i-1)*4+1:i*4])
    end

    leg_items = vcat([("Jet Fuel", :black)], [(lbl, color) for (_, lbl, color) in fuel_config.biofuel_types])
    return plot(
        plot(combined..., layout=grid(length(c0_scenarios), 5, widths=[0.05, 0.2375, 0.2375, 0.2375, 0.2375])),
        make_legend_panel(leg_items, ncols=fuel_config.legendcolumns, fontsize=15),
        layout=grid(2, 1, heights=[0.97, 0.03]), size=(3000, 2000),
        plot_title="SAF Production by Policy Stringency (c0 Sensitivity)",
        plot_titlefontsize=20, plot_titlefontweight=:bold, margin=8Plots.mm)
end

p_sens = plot_saf_stacked_sensitivity(sensitivity_extended, c0_scenarios, aviation_config)
display(p_sens)
savefig(p_sens, joinpath(OUTPUT_DIR, "saf_stacked_sensitivity_c0.png"))

# ── Dual variables ─────────────────────────────────────────────────────────
function plot_dual_variables_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    dual_info = [
        (:λ_rfs, "λ RFS D6", :steelblue),
        (:λ_blendwall_ethanol, "λ Blendwall (Ethanol)", :orange),
        (:λ_blendwall_biodiesel, "λ Blendwall (Biodiesel)", :green),
        (:λ_nonsoy_capacity, "λ Non-soy Capacity", :red),
    ]

    all_plots = []

    for (dual_key, dual_label, dual_color) in dual_info
        for (policy_type, _, xlabel, pol_title) in POLICIES
            sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
            xs = Float64[]
            ys = Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                push!(xs, get_x(s, policy_type))
                push!(ys, getfield(sol.duals, dual_key))
            end

            row_idx = findfirst(d -> d[1] == dual_key, dual_info)
            col_idx = findfirst(p -> p[1] == policy_type, POLICIES)

            # λ_blendwall_biodiesel 행에만 y축 범위 고정
            ylims_val = dual_key == :λ_blendwall_biodiesel ? (0.0, 0.1) : :auto

            p = plot(
                xlabel=row_idx == length(dual_info) ? xlabel : "",
                ylabel=col_idx == 1 ? dual_label : "",
                title=row_idx == 1 ? pol_title : "",
                titlefontsize=22, titlefontweight=:bold,
                legend=false, grid=true,
                xlims=extrema(xs),
                ylims=ylims_val,
                left_margin=col_idx == 1 ? 20Plots.mm : 5Plots.mm,
                bottom_margin=row_idx == length(dual_info) ? 12Plots.mm : 2Plots.mm,
                top_margin=row_idx == 1 ? 8Plots.mm : 2Plots.mm,
                guidefontsize=20, tickfontsize=13
            )
            plot!(p, xs, ys, linewidth=2.5, color=dual_color)
            hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1.2, label="", alpha=0.6)
            add_vlines!(p, policy_type, vlines;
                annotate_y=isempty(ys) ? 1.0 : maximum(ys) * 0.95)
            push!(all_plots, p)
        end
    end

    return plot(
        plot(all_plots..., layout=(length(dual_info), length(POLICIES))),
        layout=grid(2, 1, heights=[0.96, 0.04]),
        size=(2400, 1800)
    )
end

display(plot_dual_variables_by_policy(results_extended_analysis; vlines=vlines_data))

lcfs_scenarios = sort_scenarios(results_extended_analysis.scenario_groups[:lcfs])
lcfs_scenarios = lcfs_scenarios[1:min(50, length(lcfs_scenarios))]
df_lcfs = DataFrame(
    scenario=String[],
    conv_atj_saf=Float64[],
    nonsoy_hefa_saf=Float64[]
)

for scenario in lcfs_scenarios
    sol = results_extended_analysis.solutions[scenario]
    isnothing(sol) && continue

    push!(df_lcfs, (
        String(scenario),
        sol.q[:saf_atj_conv],
        sol.q[:saf_hefa_nonsoy]
    ))
end

println(df_lcfs)

# =================================================================================
# 10. RFS & LCFS 전용 듀얼 변수 그림
# =================================================================================

function plot_policy_specific_duals(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions

    # RFS 그림
    rfs_sorted = sort_scenarios(results_extended_analysis.scenario_groups[:rfs])
    rfs_xs = [get_x(s, :rfs) for s in rfs_sorted if !isnothing(solutions[s])]
    rfs_ys = [solutions[s].duals.λ_rfs_avi for s in rfs_sorted if !isnothing(solutions[s])]

    p_rfs = plot(
        xlabel="RFS Aviation Mandate (θ_avi)", ylabel="λ RFS Aviation",
        title="RFS Dual Variable", titlefontsize=22, titlefontweight=:bold,
        legend=false, grid=true,
        linewidth=3, color=:steelblue, marker=:circle, markersize=6,
        left_margin=18Plots.mm, bottom_margin=15Plots.mm,
        guidefontsize=18, tickfontsize=14,
        size=(900, 600)
    )
    plot!(p_rfs, rfs_xs, rfs_ys)
    hline!(p_rfs, [0], color=:gray, linestyle=:dot, linewidth=2, label="", alpha=0.6)
    if !isnothing(vlines) && haskey(vlines, :rfs)
        add_vlines!(p_rfs, :rfs, vlines; annotate_y=maximum(rfs_ys) * 0.95)
    end

    # LCFS 그림
    lcfs_sorted = sort_scenarios(results_extended_analysis.scenario_groups[:lcfs])
    lcfs_xs = [get_x(s, :lcfs) for s in lcfs_sorted if !isnothing(solutions[s])]
    lcfs_ys = [solutions[s].duals.λ_lcfs for s in lcfs_sorted if !isnothing(solutions[s])]

    p_lcfs = plot(
        xlabel="LCFS Standard (σ)", ylabel="λ LCFS",
        title="LCFS Dual Variable", titlefontsize=22, titlefontweight=:bold,
        legend=false, grid=true,
        linewidth=3, color=:darkgreen, marker=:circle, markersize=6,
        left_margin=18Plots.mm, bottom_margin=15Plots.mm,
        guidefontsize=18, tickfontsize=14,
        size=(900, 600)
    )
    plot!(p_lcfs, lcfs_xs, lcfs_ys)
    hline!(p_lcfs, [0], color=:gray, linestyle=:dot, linewidth=2, label="", alpha=0.6)
    if !isnothing(vlines) && haskey(vlines, :lcfs)
        add_vlines!(p_lcfs, :lcfs, vlines; annotate_y=maximum(lcfs_ys) * 0.95)
    end

    return plot(p_rfs, p_lcfs, layout=(1, 2), size=(1800, 600),
        plot_title="Policy-Specific Dual Variables",
        plot_titlefontsize=24, plot_titlefontweight=:bold,
        margin=12Plots.mm)
end

display(plot_policy_specific_duals(results_extended_analysis; vlines=vlines_data))

carbontax_df = filter(r -> r.policy_type == "carbontax", results_df)
sort!(carbontax_df, :t)

# 처음 20개 행 확인
first(carbontax_df[!, [:t, :q_saf_atj_conv, :q_saf_atj_cs, :q_saf_hefa_nonsoy]], 20)