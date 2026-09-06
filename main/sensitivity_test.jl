# elasticity_mac_sensitivity.jl
# avi demand elasticity sensitivity: -0.1, -0.4, -1.0

cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "model_mkt.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "units.jl"))     # metric reporting; model stays in US units
using .Units

import .ModelMkt: params, build_unified_model, extract_solution, is_solved_and_feasible
import .Analysis: calculate_emissions_detail, calculate_implicit_taxes,
    calculate_cs_changes, calculate_ps_land_changes, calculate_ps_nonsoy_changes,
    calculate_gr_changes, calculate_environmental_benefit,
    calculate_total_welfare
using DataFrames, Printf, Plots, JuMP, Statistics

# =================================================================================
# 0. parmas
# =================================================================================

# create_demand_params
function make_demand_params(sigma, p0, x0, p_high)
    A_val = p0 * x0^(-1 / sigma)
    return (k=1 / sigma, A=A_val, s=(p_high / A_val)^sigma)
end

# p0=0.04, x0=1204.79, p_high=500.0  (choke price; must match model_mkt.jl's :avi entry)
function build_params_with_avi_elasticity(base_params, sigma_avi)
    new_avi = make_demand_params(sigma_avi, 0.04, 1204.79, 500.0)
    new_demand = copy(base_params.demand)
    new_demand[:avi] = new_avi
    return merge(base_params, (demand=new_demand,))
end

# =================================================================================
# 1. Grid
# =================================================================================

const POLICY_RANGES_SENS = (
    t=0:1.0:700,
    θ_avi=0:0.003:0.9,
    σ=0.0:0.001:0.5,
    p=0:10.0:5000.0     # tax credit ($/ton CO2e): credit = p*(δ_jet - δ_g)
)

function create_policy_scenarios_sens()
    scenarios = Dict()
    for t in POLICY_RANGES_SENS.t
        scenarios[Symbol("carbontax_$(round(Int, t))")] =
            (t=Float64(t), θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation, use_ci_threshold=true, recognize_cs=true)
    end
    for θ in POLICY_RANGES_SENS.θ_avi
        scenarios[Symbol("rfs_$(round(Int, θ * 1000))")] =
            (t=0.0, θ_avi=Float64(θ), σ=0.0, p=0.0, carbon_tax_scope=:aviation, use_ci_threshold=true, recognize_cs=true)
    end
    for σ in POLICY_RANGES_SENS.σ
        scenarios[Symbol("lcfs_$(round(Int, σ * 1000))")] =
            (t=0.0, θ_avi=0.0, σ=Float64(σ), p=0.0, carbon_tax_scope=:aviation, use_ci_threshold=false, recognize_cs=true)
    end
    for p in POLICY_RANGES_SENS.p
        scenarios[Symbol("taxcredit_$(round(Int, p * 100))")] =
            (t=0.0, θ_avi=0.0, σ=0.0, p=Float64(p), carbon_tax_scope=:aviation, use_ci_threshold=true, recognize_cs=true)
    end
    scenarios[:statusquo] = (t=0.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation, use_ci_threshold=true, recognize_cs=true)
    return scenarios
end

const POLICY_MATRIX_SENS = create_policy_scenarios_sens()

# =================================================================================
# 2. helper functions
# =================================================================================

sort_scenarios(list) = sort(list, by=s -> parse(Int, split(String(s), "_")[2]))

function get_x_sens(s, policy_type, policy_matrix)
    haskey(policy_matrix, s) || return NaN
    c = policy_matrix[s]
    policy_type == :carbontax ? c.t :
    policy_type == :rfs ? c.θ_avi :
    policy_type == :lcfs ? c.σ : c.p
end

# =================================================================================
# 3. Run + welfare + MAC
# =================================================================================

function run_grid_for_elasticity(sigma_avi, policy_matrix; verbose=false)
    p_elastic = build_params_with_avi_elasticity(params, sigma_avi)
    solutions = Dict()
    solved = failed = 0

    for (name, config) in policy_matrix
        try
            model = build_unified_model(p_elastic, config)
            optimize!(model)
            if is_solved_and_feasible(model)
                sol = extract_solution(model, name)
                sol = merge(sol, (emissions=calculate_emissions_detail(sol, p_elastic),))
                if name != :statusquo
                    sol = merge(sol, (implicit_taxes=calculate_implicit_taxes(sol, p_elastic, config),))
                end
                solutions[name] = sol
                solved += 1
            else
                solutions[name] = nothing
                failed += 1
            end
        catch e
            verbose && println("  ✗ $name: $e")
            solutions[name] = nothing
            failed += 1
        end
    end
    @printf("  elasticity %.1f → solved %d / %d\n", sigma_avi, solved, solved + failed)
    return solutions, p_elastic
end

function welfare_and_mac_for_elasticity(solutions, p_elastic; scc=190.0)
    valid = filter(pr -> !isnothing(pr.second), solutions)
    sq = valid[:statusquo]

    cs = calculate_cs_changes(valid, sq, p_elastic)
    ps = calculate_ps_land_changes(valid, sq, p_elastic)
    ps_ns = calculate_ps_nonsoy_changes(valid, sq, p_elastic)
    gr = calculate_gr_changes(valid)
    env = calculate_environmental_benefit(valid, sq, scc)
    welf = calculate_total_welfare(cs, ps, gr, env; ps_nonsoy_changes=ps_ns)

    groups = (
        carbontax=[k for k in keys(valid) if startswith(String(k), "carbontax_")],
        rfs=[k for k in keys(valid) if startswith(String(k), "rfs_")],
        lcfs=[k for k in keys(valid) if startswith(String(k), "lcfs_")],
        taxcredit=[k for k in keys(valid) if startswith(String(k), "taxcredit_")],
    )

    sq_em = sq.emissions.total
    mac = Dict()
    for (pt, slist) in pairs(groups)
        sorted = sort_scenarios(slist)
        data = []
        for i in 2:length(sorted)
            s_i, s_p = sorted[i], sorted[i-1]
            Δem = valid[s_i].emissions.total - valid[s_p].emissions.total
            abs(Δem) < 1e-6 && continue
            Δps = welf[s_i].private_surplus - welf[s_p].private_surplus
            Δsw = welf[s_i].social_welfare - welf[s_p].social_welfare
            push!(data, (
                scenario=s_i,
                emission=valid[s_i].emissions.total,
                mac_private=Δps / Δem,
                mac_social=Δsw / Δem
            ))
        end
        mac[pt] = data
    end

    return (mac=mac, sq_em=sq_em, welfare=welf, solutions=valid)
end

# =================================================================================
# 4. run
# =================================================================================

const ELASTICITIES = [-0.2, -0.4, -0.6]

println("\n=== Running policy grid for each elasticity ===")
results_by_elasticity = Dict{Float64,Any}()
for σ_avi in ELASTICITIES
    sols, p_el = run_grid_for_elasticity(σ_avi, POLICY_MATRIX_SENS)
    results_by_elasticity[σ_avi] = welfare_and_mac_for_elasticity(sols, p_el; scc=190.0)
end


# =================================================================================
# 5. MAC 
# =================================================================================

function plot_mac_panels(results_by_elasticity, elasticities;
    use_social=true, max_ab=0.11, y_min=-250.0, y_max=900.0,
    fig_size=(2400, 1000))

    bg = RGB(0.96, 0.96, 0.94)
    policy_colors = [(:carbontax, :blue), (:rfs, :red), (:lcfs, :green), (:taxcredit, :purple)]
    policy_labels = Dict(:carbontax => "Carbon Tax", :rfs => "Volumetric mandate",
        :lcfs => "CI standard", :taxcredit => "Tax Credit")
    key = use_social ? :mac_social : :mac_private

    function make_panel(σ_avi, show_y)
        res = results_by_elasticity[σ_avi]
        sq_em = res.sq_em

        p = plot(title="ε = $(σ_avi)",
            titlefontsize=22, titlefontweight=:bold,
            ylabel=show_y ? "MAC (\$/tonne CO₂)" : "",
            xlabel="Cumulative Abatement (B tonne CO₂e)",
            legend=false, grid=true,
            xlims=(0, max_ab), ylims=(y_min, y_max),
            yticks=collect(-200:200.0:y_max),
            left_margin=show_y ? 22Plots.mm : 6Plots.mm,
            right_margin=8Plots.mm, bottom_margin=14Plots.mm, top_margin=10Plots.mm,
            guidefontsize=20, tickfontsize=15,
            background_color_inside=bg, background_color=:white)

        for (pt, color) in policy_colors
            data = pt in [:taxcredit, :lcfs] ? res.mac[pt][2:end] : res.mac[pt]
            ab = [sq_em - d.emission for d in data if sq_em - d.emission <= max_ab]
            val = [getfield(d, key) for d in data if sq_em - d.emission <= max_ab]
            isempty(ab) && continue
            idx = sortperm(ab)
            plot!(p, ab[idx], val[idx], linewidth=2.5, color=color)
        end
        hline!(p, [0], color=:gray, linestyle=:dot, linewidth=2.0, label="", alpha=0.8)
        return p
    end

    panels = [make_panel(elasticities[i], i == 1) for i in 1:length(elasticities)]

    p_leg = plot(legend=:top, legendcolumns=4, grid=false, showaxis=false,
        ticks=false, xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=14, background_color=:white)
    for (pt, color) in policy_colors
        plot!(p_leg, [NaN], [NaN], label=policy_labels[pt], linewidth=3, color=color)
    end

    return plot(
        plot(panels..., layout=(1, length(elasticities))),
        p_leg, layout=grid(2, 1, heights=[0.9, 0.1]),
        size=fig_size,
        plot_title=use_social ? "Social MAC by Aviation Demand Elasticity" :
                   "Private MAC by Aviation Demand Elasticity",
        plot_titlefontsize=24, plot_titlefontweight=:bold,
        background_color=:white)
end

println("\n=== Plotting MAC panels ===")
display(plot_mac_panels(results_by_elasticity, ELASTICITIES; use_social=true))

# =================================================================================
# 7. Aviation fuel stacked quantity
# =================================================================================

function plot_aviation_stacked_grid(results_by_elasticity, elasticities;
    max_ab_policy=nothing, fig_size=(2600, 1700))

    policy_meta = [
        (:carbontax, :t, "Carbon Tax (\$/tonne CO₂e)"),
        (:rfs, :θ_avi, "Volumetric mandate (θ_avi)"),
        (:lcfs, :σ, "CI standard (σ)"),
        (:taxcredit, :p, "Tax Credit (\$/tonne CO₂e)"),
    ]

    main_fuel = (:jet_fuel, "Jet Fuel", :lightgray, :black)
    saf_layers = [
        (:saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple),
        (:saf_hefa_cs, "Climate-Smart HEFA-SAF", :orange),
        (:saf_hefa_conv, "Conventional HEFA-SAF", :green),
        (:saf_atj_cs, "Climate-Smart ATJ-SAF", :red),
        (:saf_atj_conv, "Conventional ATJ-SAF", :blue),
    ]

    function get_policy_x(config, pt)
        pt == :carbontax ? config.t :
        pt == :rfs ? config.θ_avi :
        pt == :lcfs ? config.σ : config.p
    end

    ymax = 0.0
    for σ_avi in elasticities
        res = results_by_elasticity[σ_avi]
        for s in keys(res.solutions)
            sol = res.solutions[s]
            isnothing(sol) && continue
            tot = Units.gal_to_L(sol.q[:jet_fuel] + sum(sol.q[g] for (g, _, _) in saf_layers))
            tot > ymax && (ymax = tot)
        end
    end
    ylims_fixed = (0, ymax * 1.05)

    function make_panel(σ_avi, pt, xlabel, show_y, show_title, pol_title)
        res = results_by_elasticity[σ_avi]
        slist = [s for s in keys(res.solutions)
                       if startswith(String(s), String(pt) * "_") && !isnothing(res.solutions[s])]
        xvals = [get_policy_x(POLICY_MATRIX_SENS[s], pt) for s in slist]
        idx = sortperm(xvals)
        slist = slist[idx]
        xs = xvals[idx]

        x_min, x_max = isempty(xs) ? (0.0, 1.0) : extrema(xs)

        p = plot(
            xlabel=xlabel,
            ylabel=show_y ? (Units.METRIC ? "Quantity (billion liters)" :
                                            "Quantity (billion gallons)") : "",
            title=show_title ? pol_title : "",
            titlefontsize=20, titlefontweight=:bold,
            legend=false,
            grid=true,
            gridcolor=:gray,
            gridalpha=0.7,
            gridlinewidth=0.6,
            xlims=(x_min, x_max), ylims=ylims_fixed,
            left_margin=show_y ? 18Plots.mm : 5Plots.mm,
            bottom_margin=12Plots.mm, top_margin=show_title ? 8Plots.mm : 3Plots.mm,
            right_margin=5Plots.mm,
            guidefontsize=17, tickfontsize=13)

        isempty(slist) && return p

        jet = [Units.gal_to_L(res.solutions[s].q[:jet_fuel]) for s in slist]
        plot!(p, xs, jet, fillrange=0, fillalpha=0.7,
            fillcolor=main_fuel[3], linewidth=1.2, color=main_fuel[4], label="")

        cum = copy(jet)
        for (g, _, color) in saf_layers
            layer = [Units.gal_to_L(res.solutions[s].q[g]) for s in slist]
            new_cum = cum .+ layer
            mask = layer .> 1e-10
            if any(mask)
                plot!(p, xs[mask], new_cum[mask], fillrange=cum[mask],
                    fillalpha=0.7, fillcolor=color, linewidth=1.2,
                    color=color, label="")
            end
            cum = new_cum
        end
        return p
    end

    panels = []
    for (ri, σ_avi) in enumerate(elasticities)
        for (ci, (pt, _, xlabel)) in enumerate(policy_meta)
            show_title = ri == 1
            show_y = ci == 1
            pol_title = ci == 1 ? "$(policy_meta[ci][1])  (ε=$(σ_avi))" :
                        String(policy_meta[ci][1])
            ptitle = show_title ? uppercasefirst(String(pt)) : ""
            p = make_panel(σ_avi, pt, ri == length(elasticities) ? xlabel : "",
                show_y, show_title, ptitle)
            if ci == 1
                annotate!(p, xlims(p)[1], ylims_fixed[2] * 0.9,
                    text("ε = $(σ_avi)", :black, :left, 16))
            end
            push!(panels, p)
        end
    end

    # legend
    leg_items = vcat([(main_fuel[2], main_fuel[3])],
        [(lbl, color) for (_, lbl, color) in saf_layers])
    p_leg = plot(legend=:top, legendcolumns=6, grid=false, showaxis=false,
        ticks=false, xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=13, background_color=:white)
    for (lbl, color) in leg_items
        plot!(p_leg, [NaN], [NaN], label=lbl, linewidth=6, color=color)
    end

    return plot(
        plot(panels..., layout=(length(elasticities), 4)),
        p_leg, layout=grid(2, 1, heights=[0.93, 0.07]),
        size=fig_size,
        plot_title="Aviation Fuel Production by Policy and Elasticity",
        plot_titlefontsize=24, plot_titlefontweight=:bold,
        background_color=:white)
end

println("\n=== Plotting aviation fuel stacked grid (4 policy × 3 elasticity) ===")
display(plot_aviation_stacked_grid(results_by_elasticity, ELASTICITIES))



# =================================================================================
# 8c. SAF share + total RPM
# =================================================================================

function plot_share_and_rpm_stacked(results_by_elasticity, elasticities; fig_size=(2200, 1200))

    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    r_jet = params.coeff.r[:jet_fuel]
    β_saf = params.coeff.beta[(:saf, :jet_fuel)]

    policy_meta = [
        (:carbontax, "Carbon Tax (\$/tonne CO₂e)", "Carbon Tax"),
        (:rfs, "Volumetric mandate (θ_avi)", "Volumetric mandate"),
        (:lcfs, "CI standard (σ)", "CI standard"),
        (:taxcredit, "Tax Credit (\$/tonne CO₂e)", "Tax Credit"),
    ]
    ela_colors = Dict(elasticities[1] => :black,
        elasticities[2] => :dodgerblue,
        elasticities[3] => :crimson)

    function get_policy_x(config, pt)
        pt == :carbontax ? config.t :
        pt == :rfs ? config.θ_avi :
        pt == :lcfs ? config.σ : config.p
    end
    saf_share(sol) = 100 * r_jet * β_saf * sum(sol.q[g] for g in SAF_GOODS) / sol.x[:avi]

    # RPM 
    rpm_min, rpm_max = Inf, -Inf
    for σ_avi in elasticities, s in keys(results_by_elasticity[σ_avi].solutions)
        sol = results_by_elasticity[σ_avi].solutions[s]
        isnothing(sol) && continue
        rpm_min = min(rpm_min, sol.x[:avi]);
        rpm_max = max(rpm_max, sol.x[:avi])
    end
    rpm_lims = (floor(rpm_min/100)*100, ceil(rpm_max/100)*100)

    function series(pt, σ_avi, valfn)
        res = results_by_elasticity[σ_avi]
        slist = [s for s in keys(res.solutions)
                       if startswith(String(s), String(pt)*"_") && !isnothing(res.solutions[s])]
        isempty(slist) && return (Float64[], Float64[])
        xv = [get_policy_x(POLICY_MATRIX_SENS[s], pt) for s in slist]
        idx = sortperm(xv)
        slist, xv = slist[idx], xv[idx]
        return xv, [valfn(res.solutions[s]) for s in slist]
    end

    top, bot = [], []
    for (pt, xlabel, ptitle) in policy_meta
        show_y = pt == :carbontax

        # SAF share
        p_s = plot(ylabel=show_y ? "SAF blend share (%)" : "", title=ptitle,
            titlefontsize=21, titlefontweight=:bold,
            legend=false, grid=true, ylims=(-5, 100), xticks=:none,
            left_margin=show_y ? 16Plots.mm : 4Plots.mm,
            bottom_margin=2Plots.mm, top_margin=8Plots.mm, right_margin=8Plots.mm,
            guidefontsize=20, tickfontsize=16)
        for σ_avi in elasticities
            xv, yv = series(pt, σ_avi, saf_share)
            isempty(xv) && continue
            plot!(p_s, xv, yv, linewidth=2.8, color=ela_colors[σ_avi])
        end
        push!(top, p_s)

        # total RPM
        p_r = plot(xlabel=xlabel,
            ylabel=show_y ? (Units.METRIC ? "Total air travel (billion passenger-km)" :
                                            "Total RPM (billion miles)") : "",
            legend=false, grid=true, ylims=(0, Units.mile_to_km(1380)),
            left_margin=show_y ? 16Plots.mm : 4Plots.mm,
            bottom_margin=14Plots.mm, top_margin=2Plots.mm, right_margin=8Plots.mm,
            guidefontsize=20, tickfontsize=16)
        for σ_avi in elasticities
            xv, yv = series(pt, σ_avi, sol -> Units.mile_to_km(sol.x[:avi]))
            isempty(xv) && continue
            plot!(p_r, xv, yv, linewidth=2.8, color=ela_colors[σ_avi])
        end
        push!(bot, p_r)
    end

    p_leg = plot(legend=:top, legendcolumns=3, grid=false, showaxis=false,
        ticks=false, xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=18, background_color=:white)
    for σ_avi in elasticities
        plot!(p_leg, [NaN], [NaN], label="ε = $(σ_avi)", linewidth=3, color=ela_colors[σ_avi])
    end

    return plot(
        plot(vcat(top, bot)..., layout=grid(2, 4)),
        p_leg, layout=grid(2, 1, heights=[0.94, 0.06]),
        size=fig_size,
        #plot_title = "SAF Blend Share (top) and Total RPM (bottom) by Policy and Elasticity",
        #plot_titlefontsize = 22, plot_titlefontweight = :bold,
        background_color=:white)
end

println("\n=== Plotting share + RPM stacked ===")
display(plot_share_and_rpm_stacked(results_by_elasticity, ELASTICITIES))

function plot_share_rpm_mac_stacked(results_by_elasticity, elasticities;
    use_social=true, max_ab=0.11, mac_ymin=-250.0, mac_ymax=900.0,
    fig_size=(2200, 1650))

    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    r_jet = params.coeff.r[:jet_fuel]
    β_saf = params.coeff.beta[(:saf, :jet_fuel)]

    policy_meta = [
        (:carbontax, "Carbon Tax (\$/tonne CO₂e)", "Carbon Tax"),
        (:rfs, "Volumetric mandate (θ_avi)", "Volumetric mandate"),
        (:lcfs, "CI standard (σ)", "CI standard"),
        (:taxcredit, "Tax Credit (\$/tonne CO₂e)", "Tax Credit"),
    ]
    ela_colors = Dict(elasticities[1] => :black,
        elasticities[2] => :dodgerblue,
        elasticities[3] => :crimson)

    mac_key = use_social ? :mac_social : :mac_private
    mac_xlabel = "Abatement (Mt CO₂e)"

    function get_policy_x(config, pt)
        pt == :carbontax ? config.t :
        pt == :rfs ? config.θ_avi :
        pt == :lcfs ? config.σ : config.p
    end
    saf_share(sol) = 100 * r_jet * β_saf * sum(sol.q[g] for g in SAF_GOODS) / sol.x[:avi]

    function series(pt, σ_avi, valfn)
        res = results_by_elasticity[σ_avi]
        slist = [s for s in keys(res.solutions)
                       if startswith(String(s), String(pt)*"_") && !isnothing(res.solutions[s])]
        isempty(slist) && return (Float64[], Float64[])
        xv = [get_policy_x(POLICY_MATRIX_SENS[s], pt) for s in slist]
        idx = sortperm(xv)
        slist, xv = slist[idx], xv[idx]
        return xv, [valfn(res.solutions[s]) for s in slist]
    end

    top, mid, bot = [], [], []
    for (pt, xlabel, ptitle) in policy_meta
        show_y = pt == :carbontax

        # SAF share
        p_s = plot(xlabel=xlabel, ylabel=show_y ? "SAF blend share (%)" : "", title=ptitle,
            titlefontsize=21, titlefontweight=:bold,
            legend=false, grid=true, ylims=(-5, 100),
            left_margin=show_y ? 16Plots.mm : 4Plots.mm,
            bottom_margin=13Plots.mm, top_margin=8Plots.mm, right_margin=8Plots.mm,
            guidefontsize=18, tickfontsize=15)
        for σ_avi in elasticities
            xv, yv = series(pt, σ_avi, saf_share)
            isempty(xv) && continue
            plot!(p_s, xv, yv, linewidth=2.8, color=ela_colors[σ_avi])
        end
        push!(top, p_s)

        # total RPM
        p_r = plot(xlabel=xlabel,
            ylabel=show_y ? (Units.METRIC ? "Total air travel (billion passenger-km)" :
                                            "Total RPM (billion miles)") : "",
            legend=false, grid=true, ylims=(0, Units.mile_to_km(1380)),
            left_margin=show_y ? 16Plots.mm : 4Plots.mm,
            bottom_margin=13Plots.mm, top_margin=6Plots.mm, right_margin=8Plots.mm,
            guidefontsize=18, tickfontsize=15)
        for σ_avi in elasticities
            xv, yv = series(pt, σ_avi, sol -> Units.mile_to_km(sol.x[:avi]))
            isempty(xv) && continue
            plot!(p_r, xv, yv, linewidth=2.8, color=ela_colors[σ_avi])
        end
        push!(mid, p_r)

        # MAC
        p_m = plot(xlabel=mac_xlabel,
            ylabel=show_y ? "MAC (\$/tonne CO₂)" : "",
            legend=false, grid=true,
            xlims=(0, max_ab*1000), ylims=(mac_ymin, mac_ymax),
            yticks=collect(-200:200.0:mac_ymax),
            left_margin=show_y ? 16Plots.mm : 4Plots.mm,
            bottom_margin=16Plots.mm, top_margin=6Plots.mm, right_margin=8Plots.mm,
            guidefontsize=18, tickfontsize=15)
        for σ_avi in elasticities
            res = results_by_elasticity[σ_avi]
            sq_em = res.sq_em
            data = pt in [:taxcredit, :lcfs] ? res.mac[pt][1:end] : res.mac[pt]
            ab = [(sq_em - d.emission)*1000 for d in data if sq_em - d.emission <= max_ab]
            val = [getfield(d, mac_key) for d in data if sq_em - d.emission <= max_ab]
            isempty(ab) && continue
            idx = sortperm(ab)
            plot!(p_m, ab[idx], val[idx], linewidth=2.8, color=ela_colors[σ_avi])
        end
        hline!(p_m, [0], color=:gray, linestyle=:dot, linewidth=2.0, alpha=0.8)
        push!(bot, p_m)
    end

    p_leg = plot(legend=:top, legendcolumns=3, grid=false, showaxis=false,
        ticks=false, xlims=(0, 1), ylims=(0, 1), framestyle=:none,
        legendfontsize=18, background_color=:white)
    for σ_avi in elasticities
        plot!(p_leg, [NaN], [NaN], label="ε = $(σ_avi)", linewidth=3, color=ela_colors[σ_avi])
    end

    return plot(
        plot(vcat(top, mid, bot)..., layout=grid(3, 4)),
        p_leg, layout=grid(2, 1, heights=[0.95, 0.05]),
        size=fig_size,
        background_color=:white)
end

println("\n=== Plotting share + RPM + MAC stacked ===")
display(plot_share_rpm_mac_stacked(results_by_elasticity, ELASTICITIES; use_social=true))