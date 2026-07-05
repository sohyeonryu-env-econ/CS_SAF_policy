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

function add_vlines!(p, policy_type, vlines; y_top=nothing, annotate_y=nothing, fontsize=9)
    isnothing(vlines) && return
    haskey(vlines, policy_type) || return
    vline_colors = [:darkred, :darkblue]
    for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
        c = vline_colors[min(j, length(vline_colors))]
        vline!(p, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
        if !isnothing(annotate_y)
            annotate!(p, xval, annotate_y, text(vlabel, c, :center, fontsize))
        end
    end
end

# =================================================================================
# 1. Policy Grid Setup
# =================================================================================

const POLICY_RANGES = (
    t=0:0.1:700,
    θ_avi=0:0.005:0.95,
    σ=0.0:0.0005:0.5,
    p=0:0.05:100.0
)

function create_policy_scenarios()
    scenarios = Dict()
    for t in POLICY_RANGES.t
        scenarios[Symbol("carbontax_$(round(Int,t))")] =
            (t=Float64(t), θ_avi=0.0, σ=0.0, p=0.0, use_ci_threshold=false, recognize_cs=true)
    end
    for θ in POLICY_RANGES.θ_avi
        scenarios[Symbol("rfs_$(round(Int, θ*1000))")] =
            (t=0.0, θ_avi=Float64(θ), σ=0.0, p=0.0, use_ci_threshold=true, recognize_cs=true)
    end
    # ── 추가: use_ci_threshold=false 인 RFS ──
    for θ in POLICY_RANGES.θ_avi
        scenarios[Symbol("rfsnoci_$(round(Int, θ*1000))")] =
            (t=0.0, θ_avi=Float64(θ), σ=0.0, p=0.0, use_ci_threshold=false, recognize_cs=true)
    end
    for σ in POLICY_RANGES.σ
        scenarios[Symbol("lcfs_$(round(Int, σ*1000))")] =
            (t=0.0, θ_avi=0.0, σ=Float64(σ), p=0.0, use_ci_threshold=false, recognize_cs=true)
    end
    for p in POLICY_RANGES.p
        scenarios[Symbol("taxcredit_$(round(Int, p*100))")] =
            (t=0.0, θ_avi=0.0, σ=0.0, p=Float64(p), use_ci_threshold=true, recognize_cs=true)
    end
    scenarios[:statusquo] = (t=0.0, θ_avi=0.0, σ=0.0, p=0.0, use_ci_threshold=true, recognize_cs=true)
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
            rfsnoci=[k for k in keys(valid) if startswith(String(k), "rfsnoci_")],
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
                 for i in 1:(length(ms)-1) if ms[i] * ms[i+1] <= 0]

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

@load joinpath(OUTPUT_DIR, "results_target.jld2") equivalent_policies
ep_3B = equivalent_policies
@load joinpath(OUTPUT_DIR, "results_target_6.jld2") equivalent_policies
ep_6B = equivalent_policies

statusquo_em = results_extended_analysis.solutions[:statusquo].emissions.total
abatement_3B = statusquo_em - ep_3B[:rfs].actual_emission
abatement_6B = statusquo_em - ep_6B[:rfs].actual_emission
vlines_abatement = [(abatement_3B, "3B"), (abatement_6B, "6B")]

vlines_data = Dict(pt => [(ep_3B[pt].policy_value, "3B"), (ep_6B[pt].policy_value, "6B")]
                   for pt in [:carbontax, :rfs, :lcfs, :taxcredit])

# =================================================================================
# 5. MAC Plot
# =================================================================================

function plot_mac_comparison_simple(results_extended_analysis, mac_extended;
    vlines_abatement=nothing, y_max=1080.0, y_min=-250.0, fig_size=(1600, 1200))
    sq_em = results_extended_analysis.solutions[:statusquo].emissions.total
    max_ab = 0.105
    bg = RGB(0.96, 0.96, 0.94)
    policy_colors = [(:carbontax, :blue), (:rfs, :red),
        (:rfsnoci, :orange),
        (:lcfs, :green), (:taxcredit, :purple)]
    policy_labels = Dict(:carbontax => "Carbon Tax", :rfs => "RFS Aviation",
        :rfsnoci => "RFS Aviation (no CI)",
        :lcfs => "LCFS", :taxcredit => "Tax Credit")
    # collect plot data
    plot_data = Dict()
    for (pt, _) in policy_colors
        data = pt in [:taxcredit, :lcfs] ? mac_extended[pt][1:end] : mac_extended[pt]
        ab = [sq_em - d.emission for d in data if sq_em - d.emission <= max_ab]
        prv = [d.mac_private for d in data if sq_em - d.emission <= max_ab]
        soc = [d.mac_social for d in data if sq_em - d.emission <= max_ab]
        isempty(ab) && continue
        idx = sortperm(ab)
        plot_data[pt] = (ab=ab[idx], private=prv[idx], social=soc[idx])
    end
    function make_panel(key, title, show_y)
        p = plot(title=title, titlefontsize=20, titlefontweight=:bold,
            ylabel=show_y ? "MAC (\$/ton CO₂)" : "",
            xlabel="Cumulative Abatement (Billion tons CO₂)",
            legend=false, grid=true,
            xlims=(0, max_ab), ylims=(y_min, y_max),
            yticks=collect(-200:200.0:y_max),
            left_margin=show_y ? 22Plots.mm : 5Plots.mm,
            right_margin=8Plots.mm, bottom_margin=12Plots.mm, top_margin=8Plots.mm,
            guidefontsize=18, tickfontsize=18,
            background_color_inside=bg, background_color=:white)
        for (pt, color) in policy_colors
            haskey(plot_data, pt) || continue
            d = plot_data[pt]
            plot!(p, d.ab, key == :private ? d.private : d.social, linewidth=2.5, color=color)
        end
        return p
    end
    p_leg = plot(legend=:top, legendcolumns=5, grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none, legendfontsize=18, background_color=:white)
    for (pt, color) in policy_colors
        plot!(p_leg, [NaN], [NaN], label=policy_labels[pt], linewidth=3, color=color)
    end
    return plot(plot(make_panel(:private, "Private MAC", true), make_panel(:social, "Social MAC", false), layout=(1, 2)),
        p_leg, layout=grid(2, 1, heights=[0.95, 0.05]),
        size=fig_size, plot_title="",
        plot_titlefontsize=30, plot_titlefontweight=:bold, background_color=:white)
end

display(plot_mac_comparison_simple(results_extended_analysis, mac_extended;
    vlines_abatement=vlines_abatement, y_max=1080.0, y_min=-250.0))

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

function plot_land_use_stacked_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    omega = params.coeff.omega

    # 전체 y축 최대값 통일
    all_max = maximum(
        (solutions[s].l_n + solutions[s].l_cs) * 1000
        for s in keys(solutions) if !isnothing(solutions[s])
    )
    ylims_fixed = (0, all_max * 1.1)

    plots_list = []

    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xs = [get_x(s, policy_type) for s in sorted if !isnothing(solutions[s])]

        total_n = [(solutions[s].l_n) * 1000 for s in sorted if !isnothing(solutions[s])]
        total_cs = [(solutions[s].l_cs) * 1000 for s in sorted if !isnothing(solutions[s])]

        p = plot(xlabel=xlabel, ylabel="Million Acres",
            title=title, titlefontsize=22, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=extrema(xs), ylims=ylims_fixed,
            left_margin=15Plots.mm, bottom_margin=12Plots.mm,
            guidefontsize=18, tickfontsize=14)

        plot!(p, xs, total_n,
            fillrange=0, fillalpha=0.7, fillcolor=:steelblue,
            linewidth=1.5, color=:steelblue, label="")

        plot!(p, xs, total_n .+ total_cs,
            fillrange=total_n, fillalpha=0.7, fillcolor=:orange,
            linewidth=1.5, color=:orange, label="")

        add_vlines!(p, policy_type, vlines; annotate_y=ylims_fixed[2] * 0.97)
        push!(plots_list, p)
    end

    p_leg = make_legend_panel(
        [("Conventional", :steelblue), ("Climate-Smart", :orange)],
        ncols=2, fontsize=16)

    return plot(
        plot(plots_list..., layout=(2, 2)),
        p_leg,
        layout=grid(2, 1, heights=[0.93, 0.07]),
        size=(2200, 1600),
        plot_title="Total Land Use by Policy Stringency",
        plot_titlefontsize=22, plot_titlefontweight=:bold,
        margin=10Plots.mm)
end

display(plot_land_use_stacked_by_policy(results_extended_analysis; vlines=vlines_data))

# ── Stacked fuel production ──────────────────────────────────────────────────
function plot_fuel_production_stacked(results_df, fuel_config; vlines=nothing)
    plots = []
    for (idx, (policy_type, xcol, xlabel, title)) in enumerate(POLICIES)
        show_ylabel = policy_type == :carbontax

        df = sort(filter(r -> r.policy_type == String(policy_type), results_df), xcol)
        x_vals = df[!, xcol]
        x_min = minimum(x_vals)
        x_max = ep_6B[policy_type].policy_value
        biofuel_sorted = fuel_config.biofuel_types

        p = plot(xlabel=xlabel, ylabel=show_ylabel ? "Quantity (billion gallons)" : "", title=title,
            titlefontsize=30, titlefontweight=:bold, legend=false, grid=true,
            xlims=(x_min, x_max), ylims=fuel_config.ylims,
            margin=10Plots.mm, guidefontsize=30,
            left_margin=show_ylabel ? 18Plots.mm : 8Plots.mm,
            bottom_margin=15Plots.mm, tickfontsize=25, labelfontsize=28)
        main_vals = df[!, fuel_config.main_fuel]
        plot!(p, x_vals, main_vals, fillrange=fuel_config.ylims[1],
            fillalpha=0.7, fillcolor=:lightgray, linewidth=1.5, color=:black, label="")
        cumsum_vals = copy(main_vals)
        for (col, _, color) in biofuel_sorted
            col_vals = df[!, col]
            new_cum = similar(cumsum_vals)
            for i in 1:length(cumsum_vals)
                if col_vals[i] < 1e-10
                    new_cum[i] = NaN
                else
                    new_cum[i] = cumsum_vals[i] + col_vals[i]
                end
            end
            mask = .!isnan.(new_cum)
            if any(mask)
                plot!(p, x_vals[mask], new_cum[mask], fillrange=cumsum_vals[mask],
                    fillalpha=0.7, fillcolor=color, linewidth=1.5, color=color, label="")
            end
            cumsum_vals = [col_vals[i] < 1e-10 ? cumsum_vals[i] : new_cum[i] for i in 1:length(new_cum)]
        end
        add_vlines!(p, policy_type, vlines; annotate_y=fuel_config.ylims[2] * 0.97, fontsize=24)        
        push!(plots, p)
    end
    leg_items = vcat([(fuel_config.main_fuel_label, :black)],
        [(lbl, color) for (_, lbl, color) in fuel_config.biofuel_types])
    return plot(plot(plots..., layout=(2, 2)),
        make_legend_panel(leg_items, ncols=fuel_config.legendcolumns, fontsize=26),
        layout=grid(2, 1, heights=[0.92, 0.08]), size=(2500, 1800),
        plot_title="", plot_titlefontsize=30,
        plot_titlefontweight=:bold,
        left_margin=10Plots.mm, right_margin=10Plots.mm,
        top_margin=10Plots.mm, bottom_margin=20Plots.mm)
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
        add_vlines!(p_c, policy_type, vlines; annotate_y=14.5, fontsize=24)
        push!(corn_plots, p_c)

        p_o = plot(); title=title, ylabel="billion lbs", ylims=(0, 15), common_kw...)
        plot!(p_o, xs, soy_oil, linewidth=4, color=:darkgreen)
        add_vlines!(p_o, policy_type, vlines; annotate_y=14.5, fontsize=24)
        push!(oil_plots, p_o)

        p_m = plot(); title=title, ylabel="MMT", ylims=(0, 80), common_kw...)
        plot!(p_m, xs, soy_meal, linewidth=4, color=:brown)
        add_vlines!(p_m, policy_type, vlines; annotate_y=77.0, fontsize=24)
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
        x_max = ep_6B[policy_type].policy_value

        p = plot(xlabel=xlabel, ylabel="\$/bushel (Corn)", title=title,
            yguidefontcolor=:darkorange,
            titlefontsize=22, titlefontweight=:bold, legend=false, grid=true,
            xlims=(0, x_max), ylims=15,
            left_margin=18Plots.mm, bottom_margin=12Plots.mm,
            right_margin=22Plots.mm, top_margin=8Plots.mm,
            guidefontsize=20, tickfontsize=14)
        for (key, _, color, lstyle) in corn_info
            xs, ps = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                xval = get_x(s, policy_type)
                xval <= x_max || continue
                key == :feedstock_corn_cs && sol.q_feedstock[:corn_cs] <= 1e-6 && continue
                push!(xs, xval)
                push!(ps, sol.p_f[key])
            end
            isempty(xs) && continue
            plot!(p, xs, ps, linewidth=3, color=color, linestyle=lstyle)
        end
        add_vlines!(p, policy_type, vlines; annotate_y=CORN_YLIMS[2] * 0.96)

        pr = Plots.twinx(p)
        plot!(pr, ylabel="\$/lb (Soyoil)", yguidefontcolor=:darkgreen,
            xlims=(0, x_max), ylims=2,
            guidefontsize=20, tickfontsize=14, legend=false, grid=false)
        for (key, _, color, lstyle) in soyoil_info
            xs, ps = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                xval = get_x(s, policy_type)
                xval <= x_max || continue
                key == :feedstock_soy_cs && sol.q_feedstock[:soy_cs] <= 1e-6 && continue
                push!(xs, xval)
                push!(ps, sol.p_f[key])
            end
            isempty(xs) && continue
            plot!(pr, xs, ps, linewidth=3, color=color, linestyle=lstyle)
        end
        push!(plots_list, p)
    end
    # linestyle까지 반영한 범례 패널 직접 생성
    leg_items = vcat(
        [(lbl, color, lstyle) for (_, lbl, color, lstyle) in corn_info],
        [(lbl, color, lstyle) for (_, lbl, color, lstyle) in soyoil_info])

    p_leg = plot(legend=:top, legendcolumns=4, grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none, legendfontsize=14)
    for (lbl, color, lstyle) in leg_items
        plot!(p_leg, [NaN], [NaN], label=lbl, linewidth=3, color=color, linestyle=lstyle)
    end

    return plot(plot(plots_list..., layout=(2, 2)),
        p_leg,
        layout=grid(2, 1, heights=[0.93, 0.07]), size=(2200, 1600),
        plot_title="",
        plot_titlefontsize=24, plot_titlefontweight=:bold, margin=10Plots.mm)
end

display(plot_feedstock_prices_by_policy(results_extended_analysis; vlines=vlines_data))

# ── Feedstock prices & land rent (status quo 대비 변화율) ────────────────────
function plot_feedstock_prices_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    sq = solutions[:statusquo]

    sq_corn = sq.p_f[:feedstock_corn_n]
    sq_soy = sq.p_f[:feedstock_soy_n]
    sq_rent = sq.duals.r_land

    corn_info = [(:feedstock_corn_n, "Corn (Conv)", :darkorange, :solid, false, nothing),
        (:feedstock_corn_cs, "Corn (CS)", :darkorange, :dash, true, :corn_cs)]
    soyoil_info = [(:feedstock_soy_n, "Soyoil (Conv)", :darkgreen, :solid, false, nothing),
        (:feedstock_soy_cs, "Soyoil (CS)", :darkgreen, :dash, true, :soy_cs)]

    function row_ylims(getter)
        vals = Float64[]
        for (policy_type, _, _, _) in POLICIES
            sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
            x_max = ep_6B[policy_type].policy_value
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                get_x(s, policy_type) <= x_max || continue
                append!(vals, getter(sol))
            end
        end
        isempty(vals) && return (-1.0, 1.0)
        lo, hi = minimum(vals), maximum(vals)
        pad = (hi - lo) * 0.08
        pad == 0 && (pad = abs(hi) * 0.1 + 1.0)
        return (min(0.0, lo - pad), hi + pad)
    end

    corn_yl = row_ylims(sol -> [(sol.p_f[k] - sq_corn) / sq_corn * 100
                                for (k, _, _, _, is_cs, csk) in corn_info
                                if !(is_cs && sol.q_feedstock[csk] <= 1e-6)])
    soy_yl = row_ylims(sol -> [(sol.p_f[k] - sq_soy) / sq_soy * 100
                               for (k, _, _, _, is_cs, csk) in soyoil_info
                               if !(is_cs && sol.q_feedstock[csk] <= 1e-6)])
    rent_yl = row_ylims(sol -> [(sol.duals.r_land - sq_rent) / sq_rent * 100])

    # feedstock 줄 패널: show_title(윗줄만), ylabel_txt는 1열만 넘김
    function make_feedstock_panels(info, sq_base, ylabel_txt, yl; show_title, show_xlabel)
        panels = []
        for (col, (policy_type, _, xlabel, title)) in enumerate(POLICIES)
            sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
            x_max = ep_6B[policy_type].policy_value

            p = plot(xlabel=show_xlabel ? xlabel : "",
                ylabel=col == 1 ? ylabel_txt : "",
                title=show_title ? title : "",
                titlefontsize=20, titlefontweight=:bold, legend=false, grid=true,
                xlims=(0, x_max), ylims=yl,
                left_margin=col == 1 ? 18Plots.mm : 5Plots.mm,
                bottom_margin=show_xlabel ? 12Plots.mm : 4Plots.mm,
                top_margin=show_title ? 8Plots.mm : 4Plots.mm,
                guidefontsize=18, tickfontsize=14)

            for (key, _, color, lstyle, is_cs, cs_key) in info
                xs, ps = Float64[], Float64[]
                for s in sorted
                    sol = solutions[s]
                    isnothing(sol) && continue
                    xval = get_x(s, policy_type)
                    xval <= x_max || continue
                    is_cs && sol.q_feedstock[cs_key] <= 1e-6 && continue
                    push!(xs, xval)
                    push!(ps, (sol.p_f[key] - sq_base) / sq_base * 100)
                end
                isempty(xs) && continue
                plot!(p, xs, ps, linewidth=3, color=color, linestyle=lstyle)
            end
            hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1, label="")
            add_vlines!(p, policy_type, vlines; annotate_y=yl[2] * 0.93)
            push!(panels, p)
        end
        return panels
    end

    # land rent 줄 (맨 아래): 제목 없음, x라벨 있음
    function make_rent_panels()
        panels = []
        for (col, (policy_type, _, xlabel, title)) in enumerate(POLICIES)
            sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
            x_max = ep_6B[policy_type].policy_value

            p = plot(xlabel=xlabel,
                ylabel=col == 1 ? "% change (Land rent)" : "",
                title="",
                legend=false, grid=true,
                xlims=(0, x_max), ylims=rent_yl,
                left_margin=col == 1 ? 18Plots.mm : 5Plots.mm,
                bottom_margin=12Plots.mm, top_margin=4Plots.mm,
                guidefontsize=18, tickfontsize=14)

            xs, ps = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                xval = get_x(s, policy_type)
                xval <= x_max || continue
                push!(xs, xval)
                push!(ps, (sol.duals.r_land - sq_rent) / sq_rent * 100)
            end
            plot!(p, xs, ps, linewidth=3, color=:black)
            hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1, label="")
            add_vlines!(p, policy_type, vlines; annotate_y=rent_yl[2] * 0.93)
            push!(panels, p)
        end
        return panels
    end

    # 윗줄 corn(제목O, x라벨X), 가운데 soyoil(제목X, x라벨X), 아랫줄 rent(제목X, x라벨O)
    corn_panels = make_feedstock_panels(corn_info, sq_corn, "% change (Corn)", corn_yl;
        show_title=true, show_xlabel=false)
    soy_panels = make_feedstock_panels(soyoil_info, sq_soy, "% change (Soyoil)", soy_yl;
        show_title=false, show_xlabel=false)
    rent_panels = make_rent_panels()

    all_panels = vcat(corn_panels, soy_panels, rent_panels)

    leg_items = vcat(
        [(lbl, color, lstyle) for (_, lbl, color, lstyle, _, _) in corn_info],
        [(lbl, color, lstyle) for (_, lbl, color, lstyle, _, _) in soyoil_info],
        [("Land rent", :black, :solid)])
    p_leg = plot(legend=:top, legendcolumns=5, grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none, legendfontsize=17)
    for (lbl, color, lstyle) in leg_items
        plot!(p_leg, [NaN], [NaN], label=lbl, linewidth=3, color=color, linestyle=lstyle)
    end

    return plot(plot(all_panels..., layout=(3, 4)),
        p_leg,
        layout=grid(2, 1, heights=[0.95, 0.05]), size=(2600, 1900),
        plot_title="",
        plot_titlefontsize=24, plot_titlefontweight=:bold, margin=10Plots.mm)
end

display(plot_feedstock_prices_by_policy(results_extended_analysis; vlines=vlines_data))

# ── Implicit tax ─────────────────────────────────────────────────────────────
function plot_implicit_tax_by_policy(results_extended_analysis, ep_5B; vlines=nothing)
    solutions = results_extended_analysis.solutions

    fuel_info = [(:jet_fuel, "Jet Fuel", :black, :solid), (:saf_atj_conv, "Conv ATJ-SAF", :blue, :solid),
        (:saf_atj_cs, "CS ATJ-SAF", :red, :solid), (:saf_hefa_conv, "Conv HEFA-SAF", :green, :solid),
        (:saf_hefa_cs, "CS HEFA-SAF", :orange, :solid), (:saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple, :solid)]
    RFS_GROUP = [:saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    # 각 정책의 6B stringency 값까지로 xlims 통일 (0 시작)
    xlims_map = Dict(pt => (0.0, ep_6B[pt].policy_value) for pt in [:carbontax, :rfs, :lcfs, :taxcredit])
    it_key_map = Dict(:carbontax => :carbon_tax, :rfs => :rfs_avi, :lcfs => :lcfs, :taxcredit => :tax_credit)

    plots_list = []
    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xlims_cur = xlims_map[policy_type]
        it_key = it_key_map[policy_type]

        fuels_to_plot = policy_type == :rfs ?
                        filter(x -> x[1] in [:jet_fuel, :saf_atj_conv], fuel_info) : fuel_info

        # 먼저 시리즈 데이터 수집 (그리기 전에 ylims 계산 위해)
        series_data = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()
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
                xval = get_x(s, policy_type)
                xlims_cur[1] <= xval <= xlims_cur[2] || continue
                push!(xs, xval)
                push!(its, sol.implicit_taxes[g_rep][it_key])
            end
            !isempty(xs) && (series_data[:all_other_saf] = (xs, its))
        end

        # 5B 범위 내 데이터로 ylims 자동 계산
        all_yvals = Float64[]
        for (_, (_, its)) in series_data
            append!(all_yvals, its)
        end
        if isempty(all_yvals)
            ylims_cur = (-1.0, 1.0)
        else
            ymin, ymax = minimum(all_yvals), maximum(all_yvals)
            pad = (ymax - ymin) * 0.05
            pad == 0 && (pad = abs(ymax) * 0.1 + 1.0)
            ylims_cur = (ymin - pad, ymax + pad)
        end
        THRESH = (ylims_cur[2] - ylims_cur[1]) * 0.05

        p = plot(xlabel=xlabel, ylabel="Implicit Tax/Subsidy (\$/gallon)", title=title,
            titlefontsize=28, titlefontweight=:bold, legend=false, grid=true,
            xlims=xlims_cur, ylims=ylims_cur,
            left_margin=22Plots.mm, bottom_margin=15Plots.mm,
            right_margin=95Plots.mm, top_margin=10Plots.mm,
            guidefontsize=22, tickfontsize=18)
        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1.5, label="")

        # 실제 그리기
        color_map = Dict(g => (color, lstyle) for (g, _, color, lstyle) in fuel_info)
        for (g, (xs, its)) in series_data
            if g == :all_other_saf
                plot!(p, xs, its, linewidth=3.0, color=:darkgray)
            else
                c, ls = color_map[g]
                plot!(p, xs, its, linewidth=3.0, color=c, linestyle=ls)
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
                policy_type == :taxcredit && g == :jet_fuel && continue
                yp = label_pos[g]
                ylims_cur[1] - THRESH <= yp <= ylims_cur[2] + THRESH || continue
                annotate!(p, xlims_cur[2] + xw * 0.02, yp, text(lbl, color, :left, 22))
            end
        end

        # 3B, 6B vline (라벨 크기 조절 가능)
        if !isnothing(vlines) && haskey(vlines, policy_type)
            vline_colors = [:darkred, :darkblue]
            for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                c = vline_colors[min(j, length(vline_colors))]
                vline!(p, [xval], color=c, linestyle=:dash, linewidth=1.8, label="")
                annotate!(p, xval, ylims_cur[2] + 0.5 * THRESH, text(vlabel, c, :center, 16))
            end
        end

        push!(plots_list, p)
    end

    return plot(
        plots_list...,
        layout=(2, 2),
        size=(2600, 1500),
    )
end
display(plot_implicit_tax_by_policy(results_extended_analysis, ep_6B; vlines=vlines_data))


# =================================================================================
# 8. Generate & Display All Plots
# =================================================================================

# ── Aviation/Gasoline/Diesel stacked quantity plots ──────────────────────────
aviation_config = (
    main_fuel=:q_jet_fuel, main_fuel_label="Jet Fuel", ylims=(0, 23),
    biofuel_types=[
        (:q_saf_hefa_cs, "Climate-Smart HEFA-SAF", :orange),
        (:q_saf_hefa_conv, "Conventional HEFA-SAF", :green),
        (:q_saf_atj_cs, "Climate-Smart ATJ-SAF", :red),
        (:q_saf_atj_conv, "Conventional ATJ-SAF", :blue),
        (:q_saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple)],
    plot_title="Aviation Fuel Production by Policy Stringency", legendcolumns=3)

gasoline_config = (
    main_fuel=:q_gasoline, main_fuel_label="Gasoline", ylims=(0, 160),
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


# ── Dual variables ─────────────────────────────────────────────────────────
function plot_dual_variables_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    dual_info = [
        (:λ_rfs, "λ RFS D6", :steelblue),
        (:λ_blendwall_ethanol, "λ Blendwall (Ethanol)", :orange),
        (:λ_blendwall_biodiesel, "λ Blendwall (Biodiesel)", :green),
        #(:λ_nonsoy_capacity, "λ Non-soy Capacity", :red),
    ]
    all_plots = []
    for (dual_key, dual_label, dual_color) in dual_info
        for (policy_type, _, xlabel, pol_title) in POLICIES
            x_upper = vlines[policy_type][2][1]

            sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
            xs = Float64[]
            ys = Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                xval = get_x(s, policy_type)
                xval <= x_upper || continue
                push!(xs, xval)
                push!(ys, getfield(sol.duals, dual_key))
            end
            row_idx = findfirst(d -> d[1] == dual_key, dual_info)
            col_idx = findfirst(p -> p[1] == policy_type, POLICIES)
            ylims_val = dual_key == :λ_blendwall_biodiesel ? (0.0, 0.1) : :auto
            p = plot(
                xlabel=row_idx == length(dual_info) ? xlabel : "",
                ylabel=col_idx == 1 ? dual_label : "",
                title=row_idx == 1 ? pol_title : "",
                titlefontsize=22, titlefontweight=:bold,
                legend=false, grid=true,
                xlims=(minimum(xs), x_upper),
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
# 10. RFS & LCFS dual variable plots
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


# $ per mile price changes relative to status quo
# $ per mile price changes relative to status quo
function plot_mile_price_changes_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    sq = solutions[:statusquo]
    sq_avi = sq.p_c[:avi]
    sq_gas = sq.p_c[:gas]
    sq_die = sq.p_c[:die]
    series_info = [(:avi, sq_avi, "Aviation RPM", :red),
        (:gas, sq_gas, "Gasoline VMT", :orange),
        (:die, sq_die, "Diesel VMT", :green)]

    title_panels = []   # 맨 윗줄: 정책명 4개
    avi_panels = []     # 가운데 줄: aviation 4개
    road_panels = []    # 아랫줄: road 4개

    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        x_max = ep_6B[policy_type].policy_value

        data = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()
        for (key, sq_val, _, _) in series_info
            xs, ys = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                xval = get_x(s, policy_type)
                xval <= x_max || continue
                push!(xs, xval)
                push!(ys, (sol.p_c[key] - sq_val) / sq_val * 100)
            end
            data[key] = (xs, ys)
        end

        # 정책명 패널
        p_title = plot(framestyle=:none, legend=false, grid=false,
            xlims=(0, 1), ylims=(0, 1))
        annotate!(p_title, 0.5, 0.4, text(title, :black, :center, 24))
        push!(title_panels, p_title)

        # aviation 패널 (0~250)
        p_avi = plot(legend=false, grid=true,
            xlims=(0, x_max), ylims=(-10, 250),
            left_margin=8Plots.mm, bottom_margin=4Plots.mm, top_margin=2Plots.mm,
            guidefontsize=14, tickfontsize=16)
        for (key, _, _, color) in series_info
            key == :avi || continue
            xs, ys = data[key]
            isempty(xs) && continue
            plot!(p_avi, xs, ys, linewidth=2.5, color=color)
        end
        hline!(p_avi, [0], color=:gray, linestyle=:dot, linewidth=1, label="")
        annotate!(p_avi, x_max * 0.03, 250 * 0.95, text("%", :black, :left, 15))
        add_vlines!(p_avi, policy_type, vlines; annotate_y=240)
        push!(avi_panels, p_avi)

        # road 패널 (0~25), x축 라벨은 여기(맨 아래)에만
        p_road = plot(xlabel=xlabel, legend=false, grid=true,
            xlims=(0, x_max), ylims=(-2, 25),
            left_margin=8Plots.mm, bottom_margin=10Plots.mm, top_margin=2Plots.mm,
            guidefontsize=14, tickfontsize=16)
        for (key, _, _, color) in series_info
            key in (:gas, :die) || continue
            xs, ys = data[key]
            isempty(xs) && continue
            plot!(p_road, xs, ys, linewidth=2.5, color=color)
        end
        hline!(p_road, [0], color=:gray, linestyle=:dot, linewidth=1, label="")
        annotate!(p_road, x_max * 0.03, 25 * 0.95, text("%", :black, :left, 15))
        add_vlines!(p_road, policy_type, vlines; annotate_y=24)
        push!(road_panels, p_road)
    end

    # 3행 × 4열: 정책명 / aviation / road
    all_panels = vcat(title_panels, avi_panels, road_panels)
    main = plot(all_panels..., layout=grid(3, 4, heights=[0.08, 0.46, 0.46]))

    leg_items = [(lbl, color) for (_, _, lbl, color) in series_info]
    p_leg = make_legend_panel(leg_items, ncols=3, fontsize=15)

    return plot(main, p_leg,
        layout=grid(2, 1, heights=[0.94, 0.06]),
        size=(1500, 1000),
    )
end

display(plot_mile_price_changes_by_policy(results_extended_analysis; vlines=vlines_data))

# Food quantity stacked plots
function plot_food_products_stacked_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions

    CORN_YMAX = 12.0
    OIL_YMAX = 17.0
    MEAL_YMAX = 100.0

    panels = []
    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        x_max = ep_6B[policy_type].policy_value

        xs = Float64[]
        corn_food = Float64[]
        corn_total = Float64[]
        soy_oil = Float64[]
        soy_meal = Float64[]
        for s in sorted
            sol = solutions[s]
            isnothing(sol) && continue
            xval = get_x(s, policy_type)
            xval <= x_max || continue
            ddgs = 0.092 * sol.q[:ethanol] +
                   0.159 * (sol.q[:saf_atj_conv] + sol.q[:saf_atj_cs])
            push!(xs, xval)
            push!(corn_total, sol.x[:corn])
            push!(corn_food, sol.x[:corn] - ddgs)
            push!(soy_oil, sol.x[:soyoil])
            push!(soy_meal, sol.x[:soymeal])
        end

        common_kw = (legend=false, grid=true,
            xlims=(0, x_max),
            left_margin=2Plots.mm, right_margin=0Plots.mm,
            top_margin=16Plots.mm, bottom_margin=9Plots.mm,
            guidefontsize=18, tickfontsize=15)

        p_c = plot(;  xlabel=xlabel, ylims=(0, CORN_YMAX), common_kw...)
        plot!(p_c, xs, corn_food, fillrange=0, fillalpha=0.85,
            fillcolor=:orange, linewidth=1.2, color=:orange)
        plot!(p_c, xs, corn_total, fillrange=corn_food, fillalpha=0.85,
            fillcolor=:red, linewidth=1.2, color=:red)
        annotate!(p_c, x_max * 0.03, CORN_YMAX * 0.99, text("B bu", :black, :left, 15))
        add_vlines!(p_c, policy_type, vlines; annotate_y=CORN_YMAX * 0.96)

        p_o = plot(; ylims=(0, OIL_YMAX), title=title,
            titlefontsize=20, titlefontweight=:bold, common_kw...)
        plot!(p_o, xs, soy_oil, fillrange=0, fillalpha=0.85,
            fillcolor=:darkgreen, linewidth=1.2, color=:darkgreen)
        annotate!(p_o, x_max * 0.03, OIL_YMAX * 0.99, text("B lbs", :black, :left, 15))
        add_vlines!(p_o, policy_type, vlines; annotate_y=OIL_YMAX * 0.96)

        p_m = plot(; ylims=(0, MEAL_YMAX), common_kw...)
        plot!(p_m, xs, soy_meal, fillrange=0, fillalpha=0.85,
            fillcolor=:brown, linewidth=1.2, color=:brown)
        annotate!(p_m, x_max * 0.03, MEAL_YMAX * 0.99, text("MMT", :black, :left, 15))
        add_vlines!(p_m, policy_type, vlines; annotate_y=MEAL_YMAX * 0.96)

        push!(panels, p_c, p_o, p_m)
    end

    leg_items = [("Corn for food", :orange), ("DDGS (corn)", :red),
        ("Soybean oil", :darkgreen), ("Soybean meal", :brown)]
    p_leg = make_legend_panel(leg_items, ncols=4, fontsize=15)

    # 정책 3패널 사이에 좁은 spacer 열
    sp() = plot(framestyle=:none, legend=false, grid=false,
        xlims=(0, 1), ylims=(0, 1))

    top_row = [panels[1], panels[2], panels[3], sp(),
        panels[4], panels[5], panels[6]]
    bot_row = [panels[7], panels[8], panels[9], sp(),
        panels[10], panels[11], panels[12]]
    all_panels = vcat(top_row, bot_row)

    w = 0.155
    sw = 0.03
    col_widths = [w, w, w, sw, w, w, w]
    col_widths = col_widths ./ sum(col_widths)

    main = plot(all_panels..., layout=grid(2, 7, widths=col_widths))

    return plot(main, p_leg,
        layout=grid(2, 1, heights=[0.92, 0.08]),
        size=(2400, 1350),
        left_margin=5Plots.mm, right_margin=8Plots.mm,
        top_margin=5Plots.mm, bottom_margin=16Plots.mm)
end

p_food = plot_food_products_stacked_by_policy(results_extended_analysis; vlines=vlines_data)
display(p_food)
savefig(p_food, joinpath(OUTPUT_DIR, "quantity_food.png"))


# 3B summary table

function summarize_3B_table(results_extended_analysis, ep_3B)
    solutions = results_extended_analysis.solutions
    sq = solutions[:statusquo]

    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    SAF_LABELS = Dict(
        :saf_atj_conv => "ATJ_conv", :saf_atj_cs => "ATJ_cs",
        :saf_hefa_conv => "HEFA_conv", :saf_hefa_cs => "HEFA_cs",
        :saf_hefa_nonsoy => "HEFA_nonsoy")

    sq_jet = sq.q[:jet_fuel]
    sq_saf = sum(sq.q[g] for g in SAF_GOODS)
    sq_rpm = sq.x[:avi]

    # 3B 지점 시나리오 찾기 (policy_value에 가장 가까운 시나리오)
    function nearest_scenario(policy_type, target)
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        best, best_d = nothing, Inf
        for s in sorted
            isnothing(solutions[s]) && continue
            d = abs(get_x(s, policy_type) - target)
            d < best_d && (best_d=d; best=s)
        end
        return best
    end

    # SAF가 처음 도입되는(총량 > 임계) 정책 수준
    function first_saf_level(policy_type; thresh=1e-4)
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        for s in sorted
            sol = solutions[s]
            isnothing(sol) && continue
            sum(sol.q[g] for g in SAF_GOODS) > thresh && return get_x(s, policy_type)
        end
        return NaN
    end

    # 열 구성
    df = DataFrame(
        policy=String[],
        stringency_3B=Float64[],
        jet_sq=Float64[], jet_3B=Float64[],
        jet_abs_change=Float64[], jet_pct_change=Float64[],
        rpm_change=Float64[],
        saf_total_3B=Float64[], saf_abs_change=Float64[],
    )
    # 종류별 갤런, 종류별 share(%) 열 추가
    for g in SAF_GOODS
        df[!, Symbol("q_" * SAF_LABELS[g])] = Float64[]
    end
    for g in SAF_GOODS
        df[!, Symbol("share_" * SAF_LABELS[g])] = Float64[]
    end
    df[!, :first_saf_level] = Float64[]

    for (policy_type, _, _, title) in POLICIES
        target = ep_3B[policy_type].policy_value
        s = nearest_scenario(policy_type, target)
        isnothing(s) && continue
        sol = solutions[s]

        jet = sol.q[:jet_fuel]
        saf_by = [sol.q[g] for g in SAF_GOODS]
        saf_total = sum(saf_by)

        jet_pct = sq_jet > 0 ? (jet - sq_jet) / sq_jet * 100 : NaN
        rpm_pct = sq_rpm > 0 ? (sol.x[:avi] - sq_rpm) / sq_rpm * 100 : NaN

        # 종류별 share: SAF 총량 대비 %
        shares = saf_total > 1e-8 ? [q / saf_total * 100 for q in saf_by] : fill(0.0, length(saf_by))

        row = Any[title, get_x(s, policy_type),
            sq_jet, jet, jet-sq_jet, jet_pct,
            rpm_pct,
            saf_total, saf_total-sq_saf]
        append!(row, saf_by)      # 종류별 갤런
        append!(row, shares)      # 종류별 share
        push!(row, first_saf_level(policy_type))
        push!(df, row)
    end

    return df
end

df_3B = summarize_3B_table(results_extended_analysis, ep_3B)
show(df_3B, allcols=true, allrows=true)
println()

using CSV
CSV.write(joinpath(OUTPUT_DIR, "summary_3B.csv"), df_3B)

# RFS when non soy HEFA-SAF is first introduced
function first_nonsoy_hefa_level(results_extended_analysis; thresh=1e-6)
    solutions = results_extended_analysis.solutions
    sorted = sort_scenarios(results_extended_analysis.scenario_groups[:rfs])
    for s in sorted
        sol = solutions[s]
        isnothing(sol) && continue
        if sol.q[:saf_hefa_nonsoy] > thresh
            return get_x(s, :rfs)
        end
    end
    return NaN
end

θ_first = first_nonsoy_hefa_level(results_extended_analysis)
println("Non-soy HEFA SAF 최초 등장 θ_avi = ", θ_first)

function first_saf_by_type_lcfs(results_extended_analysis; thresh=1e-6)
    solutions = results_extended_analysis.solutions
    sorted = sort_scenarios(results_extended_analysis.scenario_groups[:lcfs])
    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    onset = Dict(g => NaN for g in SAF_GOODS)
    for s in sorted
        sol = solutions[s]
        isnothing(sol) && continue
        σ = get_x(s, :lcfs)
        for g in SAF_GOODS
            isnan(onset[g]) && sol.q[g] > thresh && (onset[g] = σ)
        end
    end

    # σ 오름차순 정렬 (등장 안 한 종류는 뒤로)
    ranked = sort(collect(onset), by=x -> isnan(x[2]) ? Inf : x[2])
    println("LCFS SAF 종류별 최초 등장 σ:")
    for (g, σ) in ranked
        println("  ", g, " => ", isnan(σ) ? "미도입" : σ)
    end
    return ranked
end

ranked = first_saf_by_type_lcfs(results_extended_analysis)
first_type = ranked[1]
println("\n가장 먼저 등장하는 SAF: ", first_type[1], " (σ = ", first_type[2], ")")

function find_max_social_welfare(results_extended_analysis)
    welfare_summary = results_extended_analysis.welfare_summary

    println("각 정책의 Social Welfare 최대 지점:\n")
    results = Dict()
    for (policy_type, _, _, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])

        best_s, best_sw, best_x = nothing, -Inf, NaN
        for s in sorted
            haskey(welfare_summary, s) || continue
            sw = welfare_summary[s].social_welfare
            if sw > best_sw
                best_sw = sw
                best_s = s
                best_x = get_x(s, policy_type)
            end
        end

        results[policy_type] = (scenario=best_s, stringency=best_x, social_welfare=best_sw)
        @printf("%-12s: stringency = %.4f,  social welfare = %.4f\n",
            title, best_x, best_sw)
    end
    return results
end

max_sw = find_max_social_welfare(results_extended_analysis)

function plot_welfare_summary_by_policy(results_extended_analysis; vlines=nothing)
    welfare_summary = results_extended_analysis.welfare_summary
    sectors = [:avi, :gas, :die, :corn, :soyoil, :soymeal]
    sector_colors = Dict(
        :avi => :steelblue,
        :gas => :darkorange,
        :die => :seagreen,
        :corn => :goldenrod,
        :soyoil => :mediumpurple,
        :soymeal => :sienna)

    p = plot(layout=(2, 2), size=(1400, 1000),
        plot_title="",
        left_margin=10Plots.mm, bottom_margin=5Plots.mm)

    for (idx, (policy_type, _, xlabel, title)) in enumerate(POLICIES)
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        valid = [s for s in sorted if haskey(welfare_summary, s)]
        xs = [get_x(s, policy_type) for s in valid]
        ws = [welfare_summary[s] for s in valid]
        n = length(xs)

        # CS sector별 stacked (양수는 위로, 음수는 아래로)
        pos_cum = zeros(n)
        neg_cum = zeros(n)
        for sec in sectors
            vals = [w.cs_by_sector[sec] for w in ws]
            base = similar(vals)
            top = similar(vals)
            for j in 1:n
                if vals[j] >= 0
                    base[j] = pos_cum[j]
                    top[j] = pos_cum[j] + vals[j]
                    pos_cum[j] += vals[j]
                else
                    base[j] = neg_cum[j]
                    top[j] = neg_cum[j] + vals[j]
                    neg_cum[j] += vals[j]
                end
            end
            plot!(p[idx], xs, top, fillrange=base,
                fillalpha=0.65, linewidth=0, color=sector_colors[sec],
                label="CS $(String(sec))",
                xlabel=xlabel, ylabel="Welfare Change (B\$)", title=title,
                titlefontsize=16, titlefontweight=:bold, legend=:best, legendfontsize=9)
        end

        # PS를 CS 양수 누적 위에 얹기 (음수면 음수 누적 아래로)
        ps_vals = [w.ps_land_change for w in ws]
        ps_base = similar(ps_vals)
        ps_top = similar(ps_vals)
        for j in 1:n
            if ps_vals[j] >= 0
                ps_base[j] = pos_cum[j]
                ps_top[j] = pos_cum[j] + ps_vals[j]
            else
                ps_base[j] = neg_cum[j]
                ps_top[j] = neg_cum[j] + ps_vals[j]
            end
        end
        plot!(p[idx], xs, ps_top, fillrange=ps_base,
            fillalpha=0.45, linewidth=0, color=:gray50, label="PS (land)")

        policy_type in [:carbontax, :taxcredit] &&
            plot!(p[idx], xs, [w.gr_change for w in ws], label="Govt Revenue", linewidth=2, color=:green)
        plot!(p[idx], xs, [w.private_surplus for w in ws], label="Private", linewidth=3, linestyle=:dot, color=:black)
        plot!(p[idx], xs, [w.social_welfare for w in ws], label="Social", linewidth=3, linestyle=:dot, color=:red)
        hline!(p[idx], [0], color=:gray, linestyle=:dot, label="", alpha=0.5)

        add_vlines!(p[idx], policy_type, vlines; annotate_y=nothing)
    end
    return p
end

println("\n=== Plotting welfare summary by policy ===")
display(plot_welfare_summary_by_policy(results_extended_analysis; vlines=vlines_data))


# welfare
function plot_welfare_summary_by_policy(results_extended_analysis; vlines=nothing)
    welfare_summary = results_extended_analysis.welfare_summary
    sectors = [:avi, :gas, :die, :corn, :soyoil, :soymeal]
    sector_colors = Dict(
        :avi => RGB(0.20, 0.44, 0.69),
        :gas => RGB(0.99, 0.68, 0.38),
        :die => RGB(0.90, 0.49, 0.13),
        :corn => RGB(0.55, 0.75, 0.43),
        :soyoil => RGB(0.30, 0.60, 0.30),
        :soymeal => RGB(0.14, 0.40, 0.18))

    sector_names = Dict(
        :avi => "Aviation",
        :gas => "Gasoline",
        :die => "Diesel",
        :corn => "Corn",
        :soyoil => "Soyoil",
        :soymeal => "Soymeal")

    gr_color = RGB(0.55, 0.35, 0.17)

    #ylims_map = Dict(
    #    :carbontax => (-100, 100),
    #    :rfs => (-100, 100),
    #    :lcfs => (-100, 100),
    #    :taxcredit => (-200, 150))

        ylims_map = Dict(
        :carbontax => (-50, 50),
        :rfs => (-50, 50),
        :lcfs => (-50, 50),
        :taxcredit => (-50, 50))

    show_ylabel = Dict(:carbontax => true, :rfs => false, :lcfs => true, :taxcredit => false)

    panels = []
    for (idx, (policy_type, _, xlabel, title)) in enumerate(POLICIES)
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        valid = [s for s in sorted if haskey(welfare_summary, s)]
        xs = [get_x(s, policy_type) for s in valid]
        ws = [welfare_summary[s] for s in valid]
        n = length(xs)

        x_upper = vlines[policy_type][1][1]

        keep = [i for i in 1:n if xs[i] <= x_upper]
        if length(keep) > 50
            step = ceil(Int, length(keep) / 50)
            keep = keep[1:step:end]
        end
        valid = valid[keep]
        xs = xs[keep]
        ws = [welfare_summary[s] for s in valid]
        n = length(xs)

        x_pad = 0.02 * x_upper

        pl = plot(xlabel=xlabel, ylabel=show_ylabel[policy_type] ? "Welfare Change (B\$)" : "",
            title=title, titlefontsize=20, titlefontweight=:bold, legend=false,
            xlims=(-x_pad, x_upper), ylims=ylims_map[policy_type],
            guidefontsize=18, tickfontsize=16,
            left_margin=show_ylabel[policy_type] ? 12Plots.mm : 4Plots.mm, bottom_margin=5Plots.mm)

        pos_cum = zeros(n)
        neg_cum = zeros(n)
        for sec in sectors
            vals = [w.cs_by_sector[sec] for w in ws]
            base = similar(vals)
            top = similar(vals)
            for j in 1:n
                if vals[j] >= 0
                    base[j] = pos_cum[j]
                    top[j] = pos_cum[j] + vals[j]
                    pos_cum[j] += vals[j]
                else
                    base[j] = neg_cum[j]
                    top[j] = neg_cum[j] + vals[j]
                    neg_cum[j] += vals[j]
                end
            end
            plot!(pl, xs, top, fillrange=base,
                fillalpha=0.75, linewidth=0, color=sector_colors[sec], label="")
        end

        ps_vals = [w.ps_land_change for w in ws]
        ps_base = similar(ps_vals)
        ps_top = similar(ps_vals)
        for j in 1:n
            if ps_vals[j] >= 0
                ps_base[j] = pos_cum[j]
                ps_top[j] = pos_cum[j] + ps_vals[j]
                pos_cum[j] += ps_vals[j]
            else
                ps_base[j] = neg_cum[j]
                ps_top[j] = neg_cum[j] + ps_vals[j]
                neg_cum[j] += ps_vals[j]
            end
        end
        plot!(pl, xs, ps_top, fillrange=ps_base,
            fillalpha=0.45, linewidth=0, color=:gray50, label="")

        if policy_type in [:carbontax, :taxcredit]
            gr_vals = [w.gr_change for w in ws]
            gr_base = similar(gr_vals)
            gr_top = similar(gr_vals)
            for j in 1:n
                if gr_vals[j] >= 0
                    gr_base[j] = pos_cum[j]
                    gr_top[j] = pos_cum[j] + gr_vals[j]
                    pos_cum[j] += gr_vals[j]
                else
                    gr_base[j] = neg_cum[j]
                    gr_top[j] = neg_cum[j] + gr_vals[j]
                    neg_cum[j] += gr_vals[j]
                end
            end
            plot!(pl, xs, gr_top, fillrange=gr_base,
                fillalpha=0.55, linewidth=0, color=gr_color, label="")
        end

        plot!(pl, xs, [w.private_surplus for w in ws], linewidth=4, linestyle=:dot, color=:black, label="")

        sw_vals = [w.social_welfare for w in ws]
        plot!(pl, xs, sw_vals, linewidth=4, linestyle=:dot, color=:red, label="")

        hline!(pl, [0], color=:gray, linestyle=:dot, label="", alpha=0.5)
        add_vlines!(pl, policy_type, vlines; annotate_y=ylims_map[policy_type][2]*0.9)

        max_idx = argmax(sw_vals)
        scatter!(pl, [xs[max_idx]], [sw_vals[max_idx]], marker=:circle, markersize=8,
            color=:red, markerstrokecolor=:black, markerstrokewidth=1.5, label="")

        push!(panels, pl)
    end

    p_leg = plot(legend=:top, legendcolumns=4, grid=false, showaxis=false,
        ticks=false, xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=14, background_color=:white)
    plot!(p_leg, [NaN], [NaN], seriestype=:shape, fillalpha=0.45,
        linewidth=0, color=:gray50, label="Producer surplus")
    for sec in sectors
        plot!(p_leg, [NaN], [NaN], seriestype=:shape, fillalpha=0.75,
            linewidth=0, color=sector_colors[sec], label="Cons: $(sector_names[sec])")
    end

    plot!(p_leg, [NaN], [NaN], seriestype=:shape, fillalpha=0.55,
        linewidth=0, color=gr_color, label="Govt Revenue")
    plot!(p_leg, [NaN], [NaN], linewidth=4, linestyle=:dot, color=:black, label="Private")
    plot!(p_leg, [NaN], [NaN], linewidth=4, linestyle=:dot, color=:red, label="Social")

    return plot(
        plot(panels..., layout=(2, 2)),
        p_leg, layout=grid(2, 1, heights=[0.86, 0.14]),
        size=(1400, 1150), background_color=:white)
end

println("\n=== Plotting welfare summary by policy ===")
display(plot_welfare_summary_by_policy(results_extended_analysis; vlines=vlines_data))

using DataFrames, Printf

function build_welfare_decomposition(results_extended_analysis, vlines)
    welfare_summary = results_extended_analysis.welfare_summary
    sectors = [:avi, :gas, :die, :corn, :soyoil, :soymeal]
    sector_names = Dict(
        :avi => "Aviation",
        :gas => "Gasoline",
        :die => "Diesel",
        :corn => "Corn",
        :soyoil => "Soyoil",
        :soymeal => "Soymeal")

    function nearest_scenario(policy_type, target_x)
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        valid = [s for s in sorted if haskey(welfare_summary, s)]
        xs = [get_x(s, policy_type) for s in valid]
        idx = argmin(abs.(xs .- target_x))
        return valid[idx], xs[idx]
    end

    # 3B, 6B 순서로 target 라벨을 먼저 정한 뒤 정책별로 채운다
    target_labels = ["3B", "6B"]

    rows = []
    for target_label in target_labels
        for (policy_type, _, _, title) in POLICIES
            entry = findfirst(v -> v[2] == target_label, vlines[policy_type])
            target_x = vlines[policy_type][entry][1]
            s, actual_x = nearest_scenario(policy_type, target_x)
            w = welfare_summary[s]

            row = Dict{String,Any}(
                "Policy" => title,
                "Target" => target_label,
                "Stringency" => round(actual_x, digits=4)
            )
            for sec in sectors
                row["CS_"*sector_names[sec]] = round(w.cs_by_sector[sec], digits=3)
            end
            row["CS_Total"] = round(sum(w.cs_by_sector[sec] for sec in sectors), digits=3)
            row["PS_land"] = round(w.ps_land_change, digits=3)
            row["Govt_Rev"] = round(w.gr_change, digits=3)
            row["Private"] = round(w.private_surplus, digits=3)
            row["Env"] = round(w.env_benefit, digits=3)
            row["Social"] = round(w.social_welfare, digits=3)
            push!(rows, row)
        end
    end

    col_order = vcat(
        ["Policy", "Target", "Stringency"],
        ["CS_"*sector_names[sec] for sec in sectors],
        ["CS_Total"],
        ["PS_land", "Govt_Rev", "Private", "Env", "Social"]
    )

    df = DataFrame()
    for col in col_order
        df[!, col] = [get(r, col, missing) for r in rows]
    end
    return df
end

welfare_table = build_welfare_decomposition(results_extended_analysis, vlines_data)
println("\n=== Welfare decomposition at 3B and 6B targets ===")
show(welfare_table, allrows=true, allcols=true)
println()

# emissions
function plot_emissions_stacked_broken_axis(results_extended_analysis;
    vlines=nothing, break_point=1.2, y_max=2.6)
    solutions = results_extended_analysis.solutions
    GAS_FUELS = [:gasoline, :ethanol]
    DIESEL_FUELS = [:diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    sector_info = [(:gas, "Road (Gasoline)", :steelblue), (:die, "Road (Diesel)", :orange),
        (:food, "Food", :green), (:avi, "Aviation", :red)]

    legend_order = [(:avi, "Aviation", :red), (:gas, "Road (Gasoline)", :steelblue),
        (:die, "Road (Diesel)", :orange), (:food, "Food", :green)]

    tick_step = Dict(:carbontax => 100.0, :rfs => 0.2, :lcfs => 0.05, :taxcredit => 5.0)

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
    for (idx, (policy_type, _, xlabel, title)) in enumerate(POLICIES)
        show_ylabel = idx == 1

        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xs = [get_x(s, policy_type) for s in sorted if !isnothing(solutions[s])]
        vals = Dict(k => [getfield(get_sector_em(solutions[s]), k)
            for s in sorted if !isnothing(solutions[s])]
                    for (k, _, _) in sector_info)

        x_upper = vlines[policy_type][2][1]
        xl = (minimum(xs), x_upper)

        # 정책별 세로 그리드 간격
        step = tick_step[policy_type]
        xt = collect(ceil(xl[1] / step) * step : step : xl[2])

        p_top = plot(title=title, titlefontsize=22, titlefontweight=:bold,
            ylabel=show_ylabel ? "Billion ton CO₂e" : "", legend=false,
            grid=true, gridalpha=0.3, gridcolor=:gray, gridlinewidth=0.5,
            xlims=xl, ylims=(break_point, y_max),
            xticks=(xt, fill("", length(xt))),
            bottom_margin=-4Plots.mm, top_margin=10Plots.mm,
            left_margin=show_ylabel ? 20Plots.mm : 6Plots.mm,
            guidefontsize=18, tickfontsize=16)
        draw_stacked!(p_top, xs, vals)
        hline!(p_top, [break_point], color=:white, linewidth=6, label="")
        add_vlines!(p_top, policy_type, vlines; annotate_y=y_max * 0.97)
        push!(top_plots, p_top)

        p_bot = plot(xlabel=xlabel, legend=false,
            grid=true, gridalpha=0.3, gridcolor=:gray, gridlinewidth=0.5,
            xlims=xl, ylims=(0.0, break_point), yticks=[0.0],
            xticks=xt,
            top_margin=-4Plots.mm, bottom_margin=14Plots.mm,
            left_margin=show_ylabel ? 20Plots.mm : 6Plots.mm,
            guidefontsize=18, tickfontsize=16)
        draw_stacked!(p_bot, xs, vals)
        hline!(p_bot, [break_point], color=:white, linewidth=6, label="")
        add_vlines!(p_bot, policy_type, vlines)
        push!(bot_plots, p_bot)
    end

    p_leg = make_legend_panel([(lbl, color) for (_, lbl, color) in legend_order], ncols=4, fontsize=17)
    return plot(plot(vcat(top_plots, bot_plots)..., layout=grid(2, 4, heights=[0.75, 0.25])),
        p_leg, layout=grid(2, 1, heights=[0.93, 0.07]), size=(2200, 1300),
        plot_title="",
        plot_titlefontsize=20, plot_titlefontweight=:bold, margin=8Plots.mm)
end
display(plot_emissions_stacked_broken_axis(results_extended_analysis; vlines=vlines_data, break_point=1.2, y_max=2.6))