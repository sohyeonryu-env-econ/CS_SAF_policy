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


# Land rent by policy
function plot_land_rent_by_policy(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    plots_list = []

    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xs = [get_x(s, policy_type) for s in sorted if !isnothing(solutions[s])]
        rs = [solutions[s].duals.r_land for s in sorted if !isnothing(solutions[s])]

        p = plot(xlabel=xlabel, ylabel="\$/acre", title=title,
            titlefontsize=22, titlefontweight=:bold,
            legend=false, grid=true,
            xlims=extrema(xs),
            left_margin=15Plots.mm, bottom_margin=12Plots.mm,
            guidefontsize=18, tickfontsize=14)

        plot!(p, xs, rs,
            linewidth=2, color=:black, marker=:circle, markersize=2,
            markerstrokewidth=0, label="Land rent")

        add_vlines!(p, policy_type, vlines; annotate_y=maximum(rs) * 0.97)
        push!(plots_list, p)
    end

    return plot(
        plot(plots_list..., layout=(2, 2)),
        size=(2200, 1500),
        plot_title="Land Rent by Policy Stringency",
        plot_titlefontsize=22, plot_titlefontweight=:bold,
        margin=10Plots.mm)
end

display(plot_land_rent_by_policy(results_extended_analysis; vlines=vlines_data))

# =================================================================================
# Jet fuel reduction at the 3B-equivalent carbon tax
# =================================================================================

sols = results_extended_analysis.solutions

# status quo jet fuel quantity
jet_sq = sols[:statusquo].q[:jet_fuel]

# 3B-equivalent carbon tax stringency
t_3B = ep_3B[:carbontax].policy_value

# find the carbontax scenario whose t is closest to t_3B
carbontax_scenarios = [s for s in keys(sols) if !isnothing(sols[s]) && startswith(String(s), "carbontax_")]
closest = argmin(s -> abs(EXTENDED_POLICY_MATRIX[s].t - t_3B), carbontax_scenarios)

jet_3B = sols[closest].q[:jet_fuel]

pct_change = (jet_3B - jet_sq) / jet_sq * 100

@printf("Status quo jet fuel: %.4f B gal\n", jet_sq)
@printf("3B-equivalent carbon tax (t = %.2f): jet fuel = %.4f B gal\n", EXTENDED_POLICY_MATRIX[closest].t, jet_3B)
@printf("Jet fuel change relative to status quo: %.2f%%\n", pct_change)

# RFS

# 3B-equivalent RFS stringency
RFS_3B = ep_3B[:rfs].policy_value

# RFS scenario 중 theta가 RFS_3B에 가장 가까운 것 찾기
rfs_scenarios = [s for s in keys(sols) if !isnothing(sols[s]) && startswith(String(s), "rfs_")]
closest_rfs = argmin(s -> abs(EXTENDED_POLICY_MATRIX[s].θ_avi - RFS_3B), rfs_scenarios)

# 해당 scenario의 jet fuel
jet_3B = sols[closest_rfs].q[:jet_fuel]
pct_change = (jet_3B - jet_sq) / jet_sq * 100

# SAF 도입량 (ATJ + HEFA, conventional + CS 모두 합산)
sol_3B = sols[closest_rfs]
saf_atj = sol_3B.q[:saf_atj_conv] + sol_3B.q[:saf_atj_cs]
saf_hefa = sol_3B.q[:saf_hefa_conv] + sol_3B.q[:saf_hefa_cs] + sol_3B.q[:saf_hefa_nonsoy]
saf_total = saf_atj + saf_hefa

@printf("Status quo jet fuel: %.4f B gal\n", jet_sq)
@printf("3B-equivalent RFS (θ_avi = %.4f): jet fuel = %.4f B gal\n",
    EXTENDED_POLICY_MATRIX[closest_rfs].θ_avi, jet_3B)
@printf("Jet fuel change relative to status quo: %.2f%%\n", pct_change)
@printf("\n")
@printf("SAF introduction at 3B-equivalent RFS:\n")
@printf("  ATJ SAF (conv + CS):  %.4f B gal\n", saf_atj)
@printf("  HEFA SAF (conv + CS): %.4f B gal\n", saf_hefa)
@printf("  Total SAF:            %.4f B gal\n", saf_total)

sq_sol = sols[:statusquo]

rpm_avi_sq = sq_sol.x[:avi]
rpm_avi_3B = sol_3B.x[:avi]
rpm_avi_change = (rpm_avi_3B - rpm_avi_sq) / rpm_avi_sq * 100

@printf("Aviation RPM (status quo): %.4f\n", rpm_avi_sq)
@printf("Aviation RPM (3B):         %.4f\n", rpm_avi_3B)
@printf("Aviation RPM change: %.2f%%\n", rpm_avi_change)

# LCFS
# LCFS 3B-equivalent
ep_lcfs = ep_3B[:lcfs]
σ_3B = ep_lcfs.policy_value

# ep에 solution/수량이 직접 있으면 그걸 쓰고, 아니면 σ_3B로 scenario를 특정
sol_3B = if hasproperty(ep_lcfs, :solution) && !isnothing(ep_lcfs.solution)
    ep_lcfs.solution
elseif hasproperty(ep_lcfs, :scenario)
    sols[ep_lcfs.scenario]
else
    # grid가 σ를 1000배 정수로 키를 만들었으므로 정확히 그 키를 직접 구성
    sols[Symbol("lcfs_$(round(Int, σ_3B*1000))")]
end

jet_3B = sol_3B.q[:jet_fuel]
jet_pct_change = (jet_3B - jet_sq) / jet_sq * 100

saf_atj = sol_3B.q[:saf_atj_conv] + sol_3B.q[:saf_atj_cs]
saf_hefa = sol_3B.q[:saf_hefa_conv] + sol_3B.q[:saf_hefa_cs] + sol_3B.q[:saf_hefa_nonsoy]
saf_total = saf_atj + saf_hefa

rpm_avi_sq = sq_sol.x[:avi]
rpm_avi_3B = sol_3B.x[:avi]
rpm_avi_change = (rpm_avi_3B - rpm_avi_sq) / rpm_avi_sq * 100

@printf("Status quo jet fuel: %.4f B gal\n", jet_sq)
@printf("3B-equivalent LCFS (σ = %.4f): jet fuel = %.4f B gal\n", σ_3B, jet_3B)
@printf("Jet fuel change relative to status quo: %.2f%%\n", jet_pct_change)
@printf("\n")
@printf("SAF introduction at 3B-equivalent LCFS:\n")
@printf("  ATJ SAF (conv + CS):  %.4f B gal\n", saf_atj)
@printf("  HEFA SAF (conv + CS + non-soy): %.4f B gal\n", saf_hefa)
@printf("  Total SAF:            %.4f B gal\n", saf_total)
@printf("\n")
@printf("Aviation RPM (status quo): %.4f\n", rpm_avi_sq)
@printf("Aviation RPM (3B):         %.4f\n", rpm_avi_3B)
@printf("Aviation RPM change: %.2f%%\n", rpm_avi_change)

# LCFS social welfare = 0 교차점 (양수 → 음수 전환)
lcfs_sorted = sort_scenarios(results_extended_analysis.scenario_groups[:lcfs])
ws = results_extended_analysis.welfare_summary

σ_vals = [EXTENDED_POLICY_MATRIX[s].σ for s in lcfs_sorted if haskey(ws, s)]
sw_vals = [ws[s].social_welfare for s in lcfs_sorted if haskey(ws, s)]

# 부호가 바뀌는 구간 찾아 선형보간으로 교차점 계산
crossings = Float64[]
for i in 1:(length(sw_vals)-1)
    if sw_vals[i] > 0 && sw_vals[i+1] <= 0   # 양수에서 음수(또는 0)로 전환
        σ_cross = σ_vals[i] + (-sw_vals[i] / (sw_vals[i+1] - sw_vals[i])) * (σ_vals[i+1] - σ_vals[i])
        push!(crossings, σ_cross)
    end
end

if isempty(crossings)
    println("부호 전환 없음 (social welfare 범위: $(round(minimum(sw_vals),digits=3)) ~ $(round(maximum(sw_vals),digits=3)))")
else
    for σc in crossings
        @printf("Social welfare가 음수로 전환되는 σ ≈ %.4f\n", σc)
    end
    # 전환 직후 첫 grid 점도 같이 출력
    first_neg_idx = findfirst(<=(0), sw_vals[2:end]) + 1
    @printf("첫 음수 grid 점: σ = %.4f, social welfare = %.4f\n",
        σ_vals[first_neg_idx], sw_vals[first_neg_idx])
end


# Tax Credit 3B-equivalent
# Tax Credit 3B-equivalent: ep가 들고 있는 model에서 직접 추출
sol_3B = extract_solution(ep_3B[:taxcredit].model, :taxcredit_3B)

p_3B = ep_3B[:taxcredit].policy_value

jet_3B = sol_3B.q[:jet_fuel]
jet_pct_change = (jet_3B - jet_sq) / jet_sq * 100

saf_atj = sol_3B.q[:saf_atj_conv] + sol_3B.q[:saf_atj_cs]
saf_hefa = sol_3B.q[:saf_hefa_conv] + sol_3B.q[:saf_hefa_cs] + sol_3B.q[:saf_hefa_nonsoy]
saf_total = saf_atj + saf_hefa

sol_5B = extract_solution(ep_5B[:taxcredit].model, :taxcredit_5B)
saf_hefa_5 = sol_5B.q[:saf_hefa_conv] + sol_5B.q[:saf_hefa_cs] + sol_5B.q[:saf_hefa_nonsoy]

rpm_avi_sq = sq_sol.x[:avi]
rpm_avi_3B = sol_3B.x[:avi]
rpm_avi_change = (rpm_avi_3B - rpm_avi_sq) / rpm_avi_sq * 100

@printf("Status quo jet fuel: %.4f B gal\n", jet_sq)
@printf("3B-equivalent Tax Credit (p = %.2f \$/gal): jet fuel = %.4f B gal\n", p_3B, jet_3B)
@printf("Jet fuel change relative to status quo: %.2f%%\n", jet_pct_change)
@printf("\n")
@printf("SAF introduction at 3B-equivalent Tax Credit:\n")
@printf("  ATJ SAF (conv + CS):  %.4f B gal\n", saf_atj)
@printf("  HEFA SAF (conv + CS + non-soy): %.4f B gal\n", saf_hefa)
@printf("  Total SAF:            %.4f B gal\n", saf_total)
@printf("\n")
@printf("Aviation RPM (status quo): %.4f\n", rpm_avi_sq)
@printf("Aviation RPM (3B):         %.4f\n", rpm_avi_3B)
@printf("Aviation RPM change: %.2f%%\n", rpm_avi_change)

# =================================================================================
# Check for micro-variations in road and food sectors
# =================================================================================

sols = results_extended_analysis.solutions

# 점검할 변수들
road_food_vars = [
    (:gasoline, sol -> sol.q[:gasoline]),
    (:ethanol, sol -> sol.q[:ethanol]),
    (:diesel, sol -> sol.q[:diesel]),
    (:rd_soy, sol -> sol.q[:rd_soy]),
    (:rd_nonsoy, sol -> sol.q[:rd_nonsoy]),
    (:biodiesel_soy, sol -> sol.q[:biodiesel_soy]),
    (:biodiesel_nonsoy, sol -> sol.q[:biodiesel_nonsoy]),
    (:corn_food, sol -> sol.x[:corn]),
    (:soyoil_food, sol -> sol.x[:soyoil]),
    (:soymeal, sol -> sol.x[:soymeal]),
    (:p_corn_n, sol -> sol.p_f[:feedstock_corn_n]),
    (:p_soy_n, sol -> sol.p_f[:feedstock_soy_n]),
    (:r_land, sol -> sol.duals.r_land),
]

println("="^80)
println("Range of road/food variables across each policy (min, max, max-min)")
println("="^80)

for (policy_type, _, _, title) in POLICIES
    sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
    valid = [sols[s] for s in sorted if !isnothing(sols[s])]
    isempty(valid) && continue
    println("\n### $title ###")
    for (name, f) in road_food_vars
        vals = [f(sol) for sol in valid]
        vmin, vmax = minimum(vals), maximum(vals)
        rng = vmax - vmin
        rel = vmin != 0 ? rng / abs(vmin) * 100 : NaN
        @printf("  %-14s  min=%.6f  max=%.6f  range=%.6e  (%.4f%%)\n",
            name, vmin, vmax, rng, rel)
    end
end

# =================================================================================
# Carbon tax: compare road/food before and after SAF introduction
# =================================================================================

saf_goods = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

carbontax_sorted = sort_scenarios(results_extended_analysis.scenario_groups[:carbontax])

println("\n", "="^80)
println("Carbon tax: SAF total and road/food variables by tax rate")
println("="^80)
@printf("%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n",
    "t", "SAF_tot", "gasoline", "ethanol", "rd_soy", "rd_nonsoy", "r_land")

prev_saf = 0.0
threshold_t = nothing
for s in carbontax_sorted
    sol = sols[s]
    isnothing(sol) && continue
    t = EXTENDED_POLICY_MATRIX[s].t
    saf_tot = sum(sol.q[g] for g in saf_goods)
    # SAF가 처음 0에서 양수로 바뀌는 지점 기록
    if isnothing(threshold_t) && prev_saf < 1e-6 && saf_tot >= 1e-6
        threshold_t = t
    end
    prev_saf = saf_tot
    # 너무 많으니 10단위로만 출력
    if round(Int, t) % 10 == 0
        @printf("%-8.1f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.2f\n",
            t, saf_tot, sol.q[:gasoline], sol.q[:ethanol],
            sol.q[:rd_soy], sol.q[:rd_nonsoy], sol.duals.r_land)
    end
end

println("\nSAF introduction threshold: t ≈ \$$(threshold_t)/tonCO2e")

@printf("%-8s %-14s %-14s\n", "t", "gasoline_VMT", "diesel_VMT")
for s in carbontax_sorted
    sol = sols[s]
    isnothing(sol) && continue
    t = EXTENDED_POLICY_MATRIX[s].t
    if round(Int, t) % 50 == 0 || t < 1.0
        @printf("%-8.1f %-14.4f %-14.4f\n", t, sol.x[:gas], sol.x[:die])
    end
end

# 변화량 요약
gas_vmt = [sols[s].x[:gas] for s in carbontax_sorted if !isnothing(sols[s])]
die_vmt = [sols[s].x[:die] for s in carbontax_sorted if !isnothing(sols[s])]
@printf("\nGasoline VMT: min=%.4f max=%.4f range=%.4f (%.4f%%)\n",
    minimum(gas_vmt), maximum(gas_vmt),
    maximum(gas_vmt) - minimum(gas_vmt),
    (maximum(gas_vmt) - minimum(gas_vmt)) / maximum(gas_vmt) * 100)
@printf("Diesel VMT: min=%.4f max=%.4f range=%.4f (%.4f%%)\n",
    minimum(die_vmt), maximum(die_vmt),
    maximum(die_vmt) - minimum(die_vmt),
    (maximum(die_vmt) - minimum(die_vmt)) / maximum(die_vmt) * 100)

@printf("%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n",
    "t", "p_avi", "p_gas", "p_die", "p_corn", "p_soy", "r_land")
for s in carbontax_sorted
    sol = sols[s]
    isnothing(sol) && continue
    t = EXTENDED_POLICY_MATRIX[s].t
    if round(Int, t) % 50 == 0 || t < 1.0
        @printf("%-8.1f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.2f\n",
            t, sol.p_c[:avi], sol.p_c[:gas], sol.p_c[:die],
            sol.p_f[:feedstock_corn_n], sol.p_f[:feedstock_soy_n], sol.duals.r_land)
    end
end

# 변화량 요약
for (name, f) in [
    ("p_avi", sol -> sol.p_c[:avi]),
    ("p_gas", sol -> sol.p_c[:gas]),
    ("p_die", sol -> sol.p_c[:die]),
    ("p_corn", sol -> sol.p_f[:feedstock_corn_n]),
    ("p_soy", sol -> sol.p_f[:feedstock_soy_n]),
    ("r_land", sol -> sol.duals.r_land),
]
    vals = [f(sols[s]) for s in carbontax_sorted if !isnothing(sols[s])]
    vmin, vmax = minimum(vals), maximum(vals)
    @printf("%-8s min=%.5f max=%.5f range=%.5f (%.4f%%)\n",
        name, vmin, vmax, vmax - vmin, (vmax - vmin) / vmin * 100)
end

function plot_consumer_prices_combined(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    price_info = [
        (sol -> sol.p_c[:avi], "RPM", :red),
        (sol -> sol.p_c[:gas], "Gasoline VMT", :orange),
        (sol -> sol.p_c[:die], "Diesel VMT", :green),
    ]
    sq = solutions[:statusquo]
    plots_list = []
    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])

        # tax credit만 y축 0~1, 나머지는 0~0.5
        ymax = policy_type == :taxcredit ? 1.0 : 0.4

        p = plot(xlabel=xlabel, ylabel="\$/mile", title=title,
            titlefontsize=25, titlefontweight=:bold, legend=false, grid=true,
            ylims=(0.0, ymax),
            xlims=(policy_type == :lcfs ? (-0.005, 0.2) : :auto),
            left_margin=18Plots.mm, bottom_margin=15Plots.mm,
            top_margin=8Plots.mm,
            guidefontsize=23, tickfontsize=18)
        for (f, _, color) in price_info
            xs, ps = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                push!(xs, get_x(s, policy_type))
                push!(ps, f(sol))
            end
            isempty(xs) && continue
            idx = sortperm(xs)
            xs, ps = xs[idx], ps[idx]
            plot!(p, xs, ps, linewidth=5, color=color, label="")
            x0 = xs[1]
            y0 = f(sq)
            scatter!(p, [x0], [y0], color=color, markersize=9,
                markerstrokewidth=0, label="")
        end
        add_vlines!(p, policy_type, vlines; annotate_y=ymax * 0.96)
        push!(plots_list, p)
    end
    leg_items = [("Aviation RPM", :red), ("Gasoline VMT", :orange), ("Diesel VMT", :green)]
    p_leg = make_legend_panel(leg_items, ncols=3, fontsize=20)
    return plot(
        plot(plots_list..., layout=(2, 2)),
        p_leg,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2500, 1800),
        plot_title="",
        plot_titlefontsize=30, plot_titlefontweight=:bold,
        margin=10Plots.mm)
end

display(plot_consumer_prices_combined(results_extended_analysis; vlines=vlines_data))

function plot_consumer_prices_pct_change(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    price_info = [
        (sol -> sol.p_c[:avi], "Aviation RPM", :red),
        (sol -> sol.p_c[:gas], "Gasoline VMT", :orange),
        (sol -> sol.p_c[:die], "Diesel VMT", :green),
    ]
    sq = solutions[:statusquo]

    plots_list = []
    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        # 정책별 y축 상한
        ymax = policy_type == :carbontax ? 360.0 :
               policy_type == :taxcredit ? 150.0 : 80.0

        p = plot(xlabel=xlabel, ylabel="Price change (%)", title=title,
            titlefontsize=25, titlefontweight=:bold, legend=false, grid=true,
            ylims=(-5.0, ymax),
            xlims=(policy_type == :lcfs ? (-0.005, 0.2) : :auto),
            left_margin=18Plots.mm, bottom_margin=15Plots.mm,
            top_margin=8Plots.mm, guidefontsize=23, tickfontsize=18)

        for (f, _, color) in price_info
            base = f(sq)
            xs, ps = Float64[], Float64[]
            for s in sorted
                sol = solutions[s]
                isnothing(sol) && continue
                push!(xs, get_x(s, policy_type))
                push!(ps, (f(sol) - base) / base * 100)
            end
            isempty(xs) && continue
            idx = sortperm(xs)
            plot!(p, xs[idx], ps[idx], linewidth=5, color=color, label="")

        end
        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1.5, label="")
        add_vlines!(p, policy_type, vlines; annotate_y=ymax * 0.96)
        push!(plots_list, p)
    end

    leg_items = [("Aviation RPM", :red), ("Gasoline VMT", :orange), ("Diesel VMT", :green)]
    p_leg = make_legend_panel(leg_items, ncols=3, fontsize=20)
    return plot(plot(plots_list..., layout=(2, 2)), p_leg,
        layout=grid(2, 1, heights=[0.95, 0.05]), size=(2500, 1800),
        plot_title="", plot_titlefontweight=:bold, margin=10Plots.mm)
end

display(plot_consumer_prices_pct_change(results_extended_analysis; vlines=vlines_data))

function plot_price_change_av_road(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    sq = solutions[:statusquo]

    avi_color = RGB(0.84, 0.19, 0.15)
    gas_color = RGB(0.90, 0.62, 0.0)
    die_color = RGB(0.13, 0.54, 0.13)

    # 정책별 좌(aviation), 우(road) y축 상한
    avi_ymax = Dict(:carbontax => 360.0, :rfs => 80.0, :lcfs => 80.0, :taxcredit => 5.0)
    road_ymax = Dict(:carbontax => 10.0, :rfs => 10.0, :lcfs => 10.0, :taxcredit => 160.0)

    pct(f, sol, base) = (f(sol) - base) / base * 100

    all_plots = []
    for (policy_type, _, xlabel, title) in POLICIES
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        valid = [s for s in sorted if !isnothing(solutions[s])]
        xs = [get_x(s, policy_type) for s in valid]
        idx = sortperm(xs)
        xs = xs[idx]
        xlims_cur = policy_type == :lcfs ? (0.0, 0.2) : extrema(xs)

        function add_vl!(pp, ymax)
            if !isnothing(vlines) && haskey(vlines, policy_type)
                vcolors = [RGB(0.55, 0.0, 0.0), RGB(0.0, 0.0, 0.55)]
                xspan = xlims_cur[2] - xlims_cur[1]
                for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                    c = vcolors[min(j, 2)]
                    vline!(pp, [xval], color=c, linestyle=:dash, linewidth=1.6, label="")
                    annotate!(pp, xval + xspan*0.02, ymax*0.93, text(vlabel, c, :left, 12))
                end
            end
        end

        # 왼쪽: aviation RPM
        base_avi = sq.p_c[:avi]
        rpm = [pct(s -> s.p_c[:avi], solutions[v], base_avi) for v in valid][idx]
        ymax_a = avi_ymax[policy_type]
        p_avi = plot(xlabel=xlabel, ylabel="$title\nRPM change (%)",
            titlefontsize=18, legend=false, grid=true, gridalpha=0.18,
            ylims=(-3.0, ymax_a), xlims=xlims_cur, framestyle=:box,
            left_margin=16Plots.mm, bottom_margin=10Plots.mm, top_margin=6Plots.mm,
            guidefontsize=16, tickfontsize=13)
        plot!(p_avi, xs, rpm, linewidth=3.5, color=avi_color, label="")
        hline!(p_avi, [0], color=RGB(0.6, 0.6, 0.6), linestyle=:dash, linewidth=1, label="")
        add_vl!(p_avi, ymax_a)
        push!(all_plots, p_avi)

        # 오른쪽: road VMT
        base_gas = sq.p_c[:gas];
        base_die = sq.p_c[:die]
        gasv = [pct(s -> s.p_c[:gas], solutions[v], base_gas) for v in valid][idx]
        diev = [pct(s -> s.p_c[:die], solutions[v], base_die) for v in valid][idx]
        ymax_r = road_ymax[policy_type]
        p_road = plot(xlabel=xlabel, ylabel="VMT change (%)",
            titlefontsize=18, legend=false, grid=true, gridalpha=0.18,
            ylims=(-3.0, ymax_r), xlims=xlims_cur, framestyle=:box,
            left_margin=16Plots.mm, bottom_margin=10Plots.mm, top_margin=6Plots.mm,
            guidefontsize=16, tickfontsize=13)
        plot!(p_road, xs, gasv, linewidth=3.5, color=gas_color, label="")
        plot!(p_road, xs, diev, linewidth=3.5, color=die_color, label="")
        hline!(p_road, [0], color=RGB(0.6, 0.6, 0.6), linestyle=:dash, linewidth=1, label="")
        add_vl!(p_road, ymax_r)
        push!(all_plots, p_road)
    end

    leg_items = [("Aviation RPM", avi_color),
        ("Gasoline VMT", gas_color),
        ("Diesel VMT", die_color)]
    p_leg = make_legend_panel(leg_items, ncols=3, fontsize=16)

    return plot(plot(all_plots..., layout=(4, 2)), p_leg,
        layout=grid(2, 1, heights=[0.96, 0.04]), size=(2000, 2600),
        plot_title="", margin=6Plots.mm, background_color=:white)
end

display(plot_price_change_av_road(results_extended_analysis; vlines=vlines_data))



function plot_price_paired_2x2(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions
    sq = solutions[:statusquo]

    avi_color = RGB(0.84, 0.19, 0.15)
    gas_color = RGB(0.90, 0.62, 0.0)
    die_color = RGB(0.13, 0.54, 0.13)

    avi_ymax = Dict(:carbontax => 250.0, :rfs => 250.0, :lcfs => 250.0, :taxcredit => 250.0)
    road_ymax = Dict(:carbontax => 110.0, :rfs => 110.0, :lcfs => 110.0, :taxcredit => 110.0)

    xlim_map = Dict(:carbontax => (0.0, 500.0), :rfs => (0.0, 0.9),
        :lcfs => (0.0, 0.2), :taxcredit => (0.0, 80.0))

    pct(f, sol, base) = (f(sol) - base) / base * 100

    function make_pair(policy_type, xlabel, title)
        sorted = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xlims_cur = xlim_map[policy_type]

        valid = [s for s in sorted if !isnothing(solutions[s]) &&
                                      xlims_cur[1] <= get_x(s, policy_type) <= xlims_cur[2]]
        xs = [get_x(s, policy_type) for s in valid]
        idx = sortperm(xs)
        xs = xs[idx]

        function add_vl!(pp, ymax)
            if !isnothing(vlines) && haskey(vlines, policy_type)
                vcolors = [RGB(0.55, 0.0, 0.0), RGB(0.0, 0.0, 0.55)]
                xspan = xlims_cur[2] - xlims_cur[1]
                for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                    (xlims_cur[1] <= xval <= xlims_cur[2]) || continue
                    c = vcolors[min(j, 2)]
                    vline!(pp, [xval], color=c, linestyle=:dash, linewidth=1.3, label="")
                    annotate!(pp, xval + xspan*0.05, ymax*0.88,
                        text(vlabel, c, :left, 8))
                end
            end
        end

        base_avi = sq.p_c[:avi]
        rpm = [pct(s -> s.p_c[:avi], solutions[v], base_avi) for v in valid][idx]
        ymax_a = avi_ymax[policy_type]
        p_avi = plot(legend=false, grid=false,
            ylims=(-ymax_a*0.04, ymax_a), xlims=xlims_cur, framestyle=:box,
            xlabel=xlabel, guidefontsize=10, tickfontsize=8,
            left_margin=2Plots.mm, right_margin=0Plots.mm,
            bottom_margin=8Plots.mm, top_margin=2Plots.mm)
        plot!(p_avi, xs, rpm, linewidth=3.5, color=avi_color, label="")
        add_vl!(p_avi, ymax_a)
        annotate!(p_avi, xlims_cur[1] + (xlims_cur[2]-xlims_cur[1])*0.06, ymax_a*0.97,
            text("%", :black, :left, 10))

        base_gas = sq.p_c[:gas];
        base_die = sq.p_c[:die]
        gasv = [pct(s -> s.p_c[:gas], solutions[v], base_gas) for v in valid][idx]
        diev = [pct(s -> s.p_c[:die], solutions[v], base_die) for v in valid][idx]
        ymax_r = road_ymax[policy_type]
        p_road = plot(legend=false, grid=false,
            ylims=(-ymax_r*0.04, ymax_r), xlims=xlims_cur, framestyle=:box,
            xlabel=xlabel, guidefontsize=10, tickfontsize=8,
            left_margin=0Plots.mm, right_margin=2Plots.mm,
            bottom_margin=8Plots.mm, top_margin=2Plots.mm)
        plot!(p_road, xs, gasv, linewidth=3.5, color=gas_color, label="")
        plot!(p_road, xs, diev, linewidth=3.5, color=die_color, label="")
        add_vl!(p_road, ymax_r)
        annotate!(p_road, xlims_cur[1] + (xlims_cur[2]-xlims_cur[1])*0.06, ymax_r*0.97,
            text("%", :black, :left, 10))

        graphs = plot(p_avi, p_road, layout=(1, 2))

        # 제목 패널: 여백 0으로 그래프에 바싹 붙임
        p_title = plot(framestyle=:none, legend=false, grid=false,
            xlims=(0, 1), ylims=(0, 1),
            left_margin=0Plots.mm, right_margin=0Plots.mm,
            top_margin=0Plots.mm, bottom_margin=0Plots.mm)
        annotate!(p_title, 0.5, 0.3, text(title, :black, :center, 16))

        return plot(p_title, graphs, layout=grid(2, 1, heights=[0.07, 0.93]))
    end

    pairs_plots = [make_pair(pt, xl, tt) for (pt, _, xl, tt) in POLICIES]

    # 가운데 빈 열을 끼워 좌우 정책 묶음 사이만 벌림
    # 배치: [carbontax] [빈칸] [rfs]
    #       [lcfs]      [빈칸] [taxcredit]
    blank = plot(framestyle=:none, legend=false, grid=false, background_color=:white)

    body = plot(
        pairs_plots[1], blank, pairs_plots[2],
        pairs_plots[3], blank, pairs_plots[4],
        layout=grid(2, 3, widths=[0.49, 0.02, 0.49]))

    leg_items = [("Aviation RPM", avi_color),
        ("Gasoline VMT", gas_color),
        ("Diesel VMT", die_color)]
    p_leg = make_legend_panel(leg_items, ncols=3, fontsize=15)

    return plot(
        body,
        p_leg,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(1800, 1400),
        margin=6Plots.mm, background_color=:white)
end

display(plot_price_paired_2x2(results_extended_analysis; vlines=vlines_data))

# ── Food products (stacked area, 3 panels per policy) ───────────────────────
function plot_food_products_3panel(results_extended_analysis; vlines=nothing)
    solutions = results_extended_analysis.solutions

    corn_food_color = RGB(0.95, 0.61, 0.07)
    corn_ddgs_color = RGB(0.80, 0.16, 0.13)
    oil_color = RGB(0.13, 0.45, 0.13)
    meal_color = RGB(0.55, 0.27, 0.07)

    corn_ymax = 12.0
    oil_ymax = 17.0
    meal_ymax = 100.0

    function make_triple(policy_type, xlabel, title)
        scenario_list = sort_scenarios(results_extended_analysis.scenario_groups[policy_type])
        xs = [get_x(s, policy_type) for s in scenario_list]
        idx = sortperm(xs)
        xs = xs[idx]
        xlims_cur = extrema(xs)

        ddgs = [0.092 * solutions[s].q[:ethanol] +
                0.159 * (solutions[s].q[:saf_atj_conv] + solutions[s].q[:saf_atj_cs])
                for s in scenario_list][idx]
        corn_t = [solutions[s].x[:corn] for s in scenario_list][idx]
        soy_oil = [solutions[s].x[:soyoil] for s in scenario_list][idx]
        soy_meal = [solutions[s].x[:soymeal] for s in scenario_list][idx]

        corn_food = corn_t .- ddgs

        function add_vl!(pp, ymax)
            if !isnothing(vlines) && haskey(vlines, policy_type)
                vcolors = [RGB(0.0, 0.0, 0.0), RGB(0.0, 0.0, 0.55)]
                xspan = xlims_cur[2] - xlims_cur[1]
                for (j, (xval, vlabel)) in enumerate(vlines[policy_type])
                    (xlims_cur[1] <= xval <= xlims_cur[2]) || continue
                    c = vcolors[min(j, 2)]
                    vline!(pp, [xval], color=c, linestyle=:dash, linewidth=1.3, label="")
                    annotate!(pp, xval + xspan*0.04, ymax*0.92, text(vlabel, c, :left, 8))
                end
            end
        end

        # 개별 패널 마진을 아주 컴팩트하게 축소
        base_kw = (legend=false, grid=false, xlims=xlims_cur, framestyle=:box,
            xlabel=xlabel, ylabel="", guidefontsize=9, tickfontsize=8,
            bottom_margin=8Plots.mm, top_margin=6Plots.mm,
            tick_direction=:in, widen=false,
            left_margin=6.5Plots.mm,   # 숫자가 자기 축선 바로 왼쪽에 이쁘게 위치할 최소 마진
            right_margin=1.0Plots.mm)  # 오른쪽 공백 최소화

        function add_unit_label!(pp, unit_str, ymax)
            xspan = xlims_cur[2] - xlims_cur[1]
            annotate!(pp, xlims_cur[1] + xspan*0.03, ymax*0.95, text(unit_str, :black, :left, 8))
        end

        # 1. Corn 패널
        p_c = plot(; ylims=(0, corn_ymax), base_kw...)
        plot!(p_c, xs, corn_food, fillrange=0, fillalpha=0.85,
            color=corn_food_color, linewidth=0, label="")
        plot!(p_c, xs, corn_t, fillrange=corn_food, fillalpha=0.85,
            color=corn_ddgs_color, linewidth=0, label="")
        plot!(p_c, xs, corn_t, color=corn_ddgs_color, linewidth=2, label="")
        add_vl!(p_c, corn_ymax)
        add_unit_label!(p_c, "billion bu", corn_ymax)

        # 2. Soybean Oil 패널
        p_o = plot(; ylims=(0, oil_ymax), base_kw...)
        plot!(p_o, xs, soy_oil, fillrange=0, fillalpha=0.85,
            color=oil_color, linewidth=2, label="")
        add_vl!(p_o, oil_ymax)
        add_unit_label!(p_o, "billion lbs", oil_ymax)

        # 3. Soybean Meal 패널
        p_m = plot(; ylims=(0, meal_ymax), base_kw...)
        plot!(p_m, xs, soy_meal, fillrange=0, fillalpha=0.85,
            color=meal_color, linewidth=2, label="")
        add_vl!(p_m, meal_ymax)
        add_unit_label!(p_m, "MMT", meal_ymax)

        # ★ 핵심: margin=-2.5Plots.mm 옵션을 주어 3개 서브플롯 사이의 내부 기본 간격을 강제로 당겨 좁힙니다.
        graphs = plot(p_c, p_o, p_m, layout=grid(1, 3), link=:none, margin=-2.5Plots.mm)

        p_title = plot(framestyle=:none, legend=false, grid=false,
            xlims=(0, 1), ylims=(0, 1),
            left_margin=0Plots.mm, right_margin=0Plots.mm,
            top_margin=0Plots.mm, bottom_margin=0Plots.mm)
        annotate!(p_title, 0.5, 0.3, text(title, :black, :center, 16))

        return plot(p_title, graphs, layout=grid(2, 1, heights=[0.07, 0.93]))
    end

    triples = [make_triple(pt, xl, tt) for (pt, _, xl, tt) in POLICIES]

    blank = plot(framestyle=:none, legend=false, grid=false, background_color=:white)

    # 중앙 세로 여백(widths의 가운데 0.012)은 그대로 유지하여 좌우 정책 블록 경계선은 살려둡니다.
    body = plot(
        triples[1], blank, triples[2],
        triples[3], blank, triples[4],
        layout=grid(2, 3, widths=[0.494, 0.012, 0.494]))

    leg_items = [("Corn for food", corn_food_color),
        ("DDGS (corn)", corn_ddgs_color),
        ("Soybean oil", oil_color),
        ("Soybean meal", meal_color)]
    p_leg = make_legend_panel(leg_items, ncols=4, fontsize=14)

    return plot(
        body,
        p_leg,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2200, 1400),
        margin=4Plots.mm, background_color=:white)
end

display(plot_food_products_3panel(results_extended_analysis; vlines=vlines_data))