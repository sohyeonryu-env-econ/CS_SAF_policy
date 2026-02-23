# Zero-profit decomposition
cd(@__DIR__)
include(joinpath(@__DIR__, "SAFModel.jl"))
using .SAFModel
using JLD2, DataFrames, CSV
const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"


# =================================================================================
# 1. Unpack parameters from SAFModel
# =================================================================================

coeff = params.coeff
fuel_cost = params.supply.fuel

ω = coeff.omega;
κ = coeff.kappa;
γ = coeff.gamma;
α = coeff.alpha;
β = coeff.beta;
δ = coeff.delta;
δ_mj = coeff.delta_mj;
r_energy = coeff.r;
θ_road = coeff.theta;
baselineCI = coeff.baselineCI;
soybean_to_oil = coeff.soybean_to_oil
soybean_to_meal = coeff.soybean_to_meal
nonsoy_feedstock_price = coeff.nonsoy_feedstock_price
hefa_saf_premium = coeff.hefa_saf_premium

SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
BIODIESEL_GOODS = [:biodiesel_soy, :biodiesel_nonsoy]
RD_GOODS = [:rd_soy, :rd_nonsoy]
AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
ALL_FUEL_GOODS = params.sets.fuel_goods

# =================================================================================
# 2. Load results
# =================================================================================

@load joinpath(OUTPUT_DIR, "results_base_welfare.jld2") results_base_analysis policy_configs_base

all_target_results = Dict()
for target_saf in [3.0, 5.0]
    suffix = target_saf == 3.0 ? "" : "_$(Int(target_saf))"
    fd = load(joinpath(OUTPUT_DIR, "results_equivalent_emissions_complete$(suffix).jld2"))
    all_target_results[target_saf] = (
        results=fd["results_equiv_emission_analysis"],
        configs=fd["policy_configs_emission"],
    )
end

# =================================================================================
# 3. Scenario registry
# =================================================================================

scenario_order = [(:statusquo, :base), (:carbontax, :base)]
scenario_labels = Dict(
    (:statusquo, :base) => "Status Quo",
    (:carbontax, :base) => "First Best CarbonTax",
)
plabel = Dict(:carbontax => "CarbonTax", :rfs => "RFS", :lcfs => "LCFS", :taxcredit => "TaxCredit")

for target_saf in [3.0, 5.0], policy in [:carbontax, :rfs, :lcfs, :taxcredit]
    suffix = target_saf == 3.0 ? "" : "_$(Int(target_saf))B"
    key = (policy, Symbol("equiv_emission_$(Int(target_saf))"))
    push!(scenario_order, key)
    scenario_labels[key] = "EquivEmission$(suffix)_$(plabel[policy])"
end

function get_sol_config(scenario, group)
    group == :base && return results_base_analysis[scenario], getproperty(policy_configs_base, scenario)
    target_saf = parse(Float64, match(r"equiv_emission_(\d+)", String(group)).captures[1])
    td = all_target_results[target_saf]
    return td.results[scenario], getproperty(td.configs, scenario)
end

sv(x) = ismissing(x) ? 0.0 : Float64(x)

# =================================================================================
# 4. Upstream decomposition
# =================================================================================

function decompose_upstream(sol, label)
    p_f = sol.p_f
    p_c = sol.p_c
    r_land = sv(sol.duals.r_land)

    function farmer_row(type)
        g_corn = type == :n ? :feedstock_corn_n : :feedstock_corn_cs
        g_soy = type == :n ? :feedstock_soy_n : :feedstock_soy_cs
        mr_corn = ω * γ[g_corn] * sv(p_f[g_corn])
        mr_soy_oil = (1 - ω) * γ[g_soy] * sv(p_f[g_soy]) * soybean_to_oil
        mr_soy_meal = (1 - ω) * γ[g_soy] * sv(p_c[:soymeal]) * soybean_to_meal
        mr_total = mr_corn + mr_soy_oil + mr_soy_meal
        kap = type == :n ? 0.0 : κ
        (Scenario=label, Farmer=type == :n ? "Conventional" : "ClimSmart",
            r_land=r_land, MR_total=mr_total, MR_corn=mr_corn,
            MR_soy_oil=mr_soy_oil, MR_soy_meal=mr_soy_meal,
            kappa=kap, ZP_residual=r_land + kap - mr_total)
    end

    rows = [farmer_row(:n)]
    sv(sol.l_cs) > 1e-6 && push!(rows, farmer_row(:cs))
    return rows
end

# =================================================================================
# 5. Downstream decomposition
# =================================================================================

function calc_mc(key, qty)
    fc = fuel_cost[key]
    fc.c0 + fc.c1 * qty + fc.c2 * max(0.0, qty - fc.v)^2
end

function decompose_downstream(sol, config, label)
    p_f = sol.p_f
    p_c = sol.p_c
    q = sol.q
    d = sol.duals
    λ_rfs = sv(d.λ_rfs)
    λ_rfs_avi = sv(d.λ_rfs_avi)
    λ_lcfs = sv(d.λ_lcfs)
    λ_bw_eth = sv(d.λ_blendwall_ethanol)
    λ_bw_bd = sv(d.λ_blendwall_biodiesel)
    λ_nonsoy = sv(d.λ_nonsoy_capacity)

    mc_atj = calc_mc(:saf_atj_shared, sv(q[:saf_atj_conv]) + sv(q[:saf_atj_cs]))
    mc_hefa = calc_mc(:saf_hefa_shared,
        sum(sv(q[g]) for g in [:saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy, :rd_soy, :rd_nonsoy]))
    mc_bd = calc_mc(:biodiesel_shared, sv(q[:biodiesel_soy]) + sv(q[:biodiesel_nonsoy]))

    ppu(g) =
        g == :jet_fuel ? r_energy[:jet_fuel] * sv(p_c[:avi]) :
        g in SAF_GOODS ? r_energy[:jet_fuel] * β[(:saf, :jet_fuel)] * sv(p_c[:avi]) :
        g == :gasoline ? r_energy[:gasoline] * sv(p_c[:gas]) :
        g == :ethanol ? r_energy[:gasoline] * β[(:ethanol, :gasoline)] * sv(p_c[:gas]) :
        g == :diesel ? r_energy[:diesel] * sv(p_c[:die]) :
        g in BIODIESEL_GOODS ? r_energy[:diesel] * β[(:biodiesel, :diesel)] * sv(p_c[:die]) :
        g in RD_GOODS ? r_energy[:diesel] * β[(:rd, :diesel)] * sv(p_c[:die]) : 0.0

    process_mc(g) =
        g in (:saf_atj_conv, :saf_atj_cs) ? mc_atj :
        g in (:saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy) ? mc_hefa :
        g in BIODIESEL_GOODS ? mc_bd :
        g in RD_GOODS ? mc_hefa - hefa_saf_premium :
        calc_mc(g, sv(q[g]))

    feedstock_cost(g) =
        g == :saf_atj_conv ? α[:saf_atj_conv] * sv(p_f[:feedstock_corn_n]) :
        g == :saf_atj_cs ? α[:saf_atj_cs] * sv(p_f[:feedstock_corn_cs]) :
        g == :saf_hefa_conv ? α[:saf_hefa_conv] * sv(p_f[:feedstock_soy_n]) :
        g == :saf_hefa_cs ? α[:saf_hefa_cs] * sv(p_f[:feedstock_soy_cs]) :
        g in (:saf_hefa_nonsoy, :biodiesel_nonsoy, :rd_nonsoy) ? α[g] * nonsoy_feedstock_price :
        g == :ethanol ? α[:ethanol] * sv(p_f[:feedstock_corn_n]) :
        g == :biodiesel_soy ? α[:biodiesel_soy] * sv(p_f[:feedstock_soy_n]) :
        g == :rd_soy ? α[:rd_soy] * sv(p_f[:feedstock_soy_n]) : 0.0

    byproduct(g) =
        g in (:saf_atj_conv, :saf_atj_cs) ? -0.159 * sv(p_f[:feedstock_corn_n]) :
        g == :ethanol ? -0.092 * sv(p_f[:feedstock_corn_n]) : 0.0

    function pa_breakdown(g)
        ct = config.t * (g in AVIATION_FUELS ? δ[g] : 0.0)
        tc = g in SAF_GOODS ? -tax_credit_rate(δ_mj[g], baselineCI, config.p) : 0.0
        lc = λ_lcfs * (g in AVIATION_FUELS ? -((1 - config.σ) * δ[:jet_fuel] - δ[g]) : 0.0)
        ra = λ_rfs_avi * (g == :jet_fuel ? config.θ_avi :
                          (g in SAF_GOODS && δ[g] <= 0.5 * δ[:jet_fuel]) ? -1.6 : 0.0)
        rr = λ_rfs * ((g == :gasoline || g == :diesel) ? θ_road :
                      g == :ethanol ? -1.0 : g in BIODIESEL_GOODS ? -1.5 :
                      g in RD_GOODS ? -1.7 : 0.0)
        bw = λ_bw_eth * (g == :gasoline ? -0.1 : g == :ethanol ? 0.9 : 0.0) +
             λ_bw_bd * (g == :diesel ? -0.05 : g in BIODIESEL_GOODS ? 0.95 : 0.0)
        ns = λ_nonsoy * (g == :saf_hefa_nonsoy ? α[:saf_hefa_nonsoy] :
                         g == :biodiesel_nonsoy ? α[:biodiesel_nonsoy] :
                         g == :rd_nonsoy ? α[:rd_nonsoy] : 0.0)
        (ct=ct, tc=tc, lc=lc, ra=ra, rr=rr, bw=bw, ns=ns, total=ct + tc + lc + ra + rr + bw + ns)
    end

    rows = []
    for g in ALL_FUEL_GOODS
        sv(q[g]) > 1e-6 || continue
        mc = process_mc(g)
        fc = feedstock_cost(g)
        bp = byproduct(g)
        pa = pa_breakdown(g)
        push!(rows, (
            Scenario=label, Good=string(g), Qty_Bgal=sv(q[g]),
            Price_per_unit=ppu(g), ProcessMC=mc, FeedstockCost=fc,
            ByproductDeduct=bp, PolicyAdj_total=pa.total,
            PA_CarbonTax=pa.ct, PA_TaxCredit=pa.tc, PA_LCFS=pa.lc,
            PA_RFS_avi=pa.ra, PA_RFS_road=pa.rr,
            PA_BlendWall=pa.bw, PA_NonsoyCapacity=pa.ns,
            ZP_residual=mc + fc + bp + pa.total - ppu(g),
        ))
    end
    return rows
end

# =================================================================================
# 6. Run and save
# =================================================================================

up_rows, dn_rows = [], []
for (scenario, group) in scenario_order
    label = scenario_labels[(scenario, group)]
    sol, config = get_sol_config(scenario, group)
    append!(up_rows, decompose_upstream(sol, label))
    append!(dn_rows, decompose_downstream(sol, config, label))
end

df_up = DataFrame(up_rows)
df_down = DataFrame(dn_rows)

for df in (df_up, df_down), col in names(df)
    eltype(df[!, col]) <: Number && (df[!, col] = round.(df[!, col], digits=4))
end

CSV.write(joinpath(OUTPUT_DIR, "decomp_upstream.csv"), df_up)
CSV.write(joinpath(OUTPUT_DIR, "decomp_downstream.csv"), df_down)

println("✓ decomp_upstream.csv   ($(nrow(df_up)) rows)")
println("✓ decomp_downstream.csv ($(nrow(df_down)) rows)")
println("Upstream   max |ZP_residual| : ", maximum(abs, df_up.ZP_residual))
println("Downstream max |ZP_residual| : ", maximum(abs, df_down.ZP_residual))

# =================================================================================
# 7. Plot upstream zero-profit decomposition
# =================================================================================

begin
    using Plots

    bar_spec_up = [
        ("Status Quo", "Conventional", 1.0, "Status Quo", ""),
        ("First Best CarbonTax", "Conventional", 2.5, "1st Best CT", ""),
        ("EquivEmission_CarbonTax", "Conventional", 4.5, "CT", "3B RFS Equiv"),
        ("EquivEmission_RFS", "Conventional", 5.5, "RFS\n(Conv)", "3B RFS Equiv"),
        ("EquivEmission_RFS", "ClimSmart", 6.1, "RFS\n(CS)", "3B RFS Equiv"),
        ("EquivEmission_LCFS", "Conventional", 7.0, "LCFS", "3B RFS Equiv"),
        ("EquivEmission_TaxCredit", "Conventional", 7.9, "TC", "3B RFS Equiv"),
        ("EquivEmission_5B_CarbonTax", "Conventional", 9.9, "CT", "5B RFS Equiv"),
        ("EquivEmission_5B_RFS", "Conventional", 10.9, "RFS\n(Conv)", "5B RFS Equiv"),
        ("EquivEmission_5B_RFS", "ClimSmart", 11.5, "RFS\n(CS)", "5B RFS Equiv"),
        ("EquivEmission_5B_LCFS", "Conventional", 12.4, "LCFS", "5B RFS Equiv"),
        ("EquivEmission_5B_TaxCredit", "Conventional", 13.3, "TC", "5B RFS Equiv"),
    ]

    C_CORN = colorant"#F0C14B"
    C_MEAL = colorant"#1B5E20"
    C_OIL = colorant"#55A868"
    C_KAPPA = colorant"#4C72B0"
    BAR_W = 0.40

    csv_scen_up = unique(df_up.Scenario)

    xs_up = Float64[]
    xlbls_up = String[]
    grplbls_up = String[]
    r_land_up = Float64[]
    corn_v = Float64[]
    meal_v = Float64[]
    oil_v = Float64[]
    kap_v = Float64[]

    for (scen, ftype, xpos, xlbl, grp) in bar_spec_up
        scen in csv_scen_up || continue
        row = filter(r -> r.Scenario == scen && r.Farmer == ftype, df_up)
        nrow(row) == 0 && continue
        push!(xs_up, xpos)
        push!(xlbls_up, xlbl)
        push!(grplbls_up, grp)
        push!(r_land_up, row[1, :r_land])
        push!(corn_v, row[1, :MR_corn])
        push!(meal_v, row[1, :MR_soy_meal])
        push!(oil_v, row[1, :MR_soy_oil])
        push!(kap_v, abs(row[1, :kappa]))
    end

    n_up = length(xs_up)
    ymax = maximum(r_land_up) * 1.15
    κ_max = maximum(kap_v)           # = 19
    y_grp = -(κ_max + 80)            # group label sits 35 units below κ bottom
    ylim_lo = y_grp - 70              # a bit more room below the label

    function rect_shape(x_center, y_bottom, y_top, half_w)
        l = x_center - half_w
        r = x_center + half_w
        Shape([l, r, r, l, l], [y_bottom, y_bottom, y_top, y_top, y_bottom])
    end

    fig_up = plot(
        size=(1200, 560),
        legend=:outertopright,
        legendfontsize=8, tickfontsize=7,
        guidefontsize=9, titlefontsize=11,
        ylabel="\$/acre",
        title="Farmer's Zero-Profit Condition Decomposition",
        bottom_margin=40Plots.px, left_margin=15Plots.px, right_margin=5Plots.px,
        grid=:y, gridalpha=0.3,
        xlims=(0.0, 14.5),
        ylims=(ylim_lo, ymax),
        xticks=(xs_up, xlbls_up),
    )

    labeled = Dict(:corn => false, :meal => false, :oil => false, :kap => false)

    for i in 1:n_up
        x = xs_up[i]
        hw = BAR_W / 2
        c = corn_v[i]
        m = meal_v[i]
        o = oil_v[i]
        k = kap_v[i]

        lbl = labeled[:corn] ? "" : "MR Corn"
        plot!(fig_up, rect_shape(x, 0.0, c, hw);
            seriestype=:shape, color=C_CORN, linecolor=:white, linewidth=0.3, label=lbl)
        labeled[:corn] = true

        lbl = labeled[:meal] ? "" : "MR Soy Meal"
        plot!(fig_up, rect_shape(x, c, c + m, hw);
            seriestype=:shape, color=C_MEAL, linecolor=:white, linewidth=0.3, label=lbl)
        labeled[:meal] = true

        lbl = labeled[:oil] ? "" : "MR Soy Oil"
        plot!(fig_up, rect_shape(x, c + m, c + m + o, hw);
            seriestype=:shape, color=C_OIL, linecolor=:white, linewidth=0.3, label=lbl)
        labeled[:oil] = true

        if k > 1e-6
            lbl = labeled[:kap] ? "" : "κ (CS adoption cost)"
            plot!(fig_up, rect_shape(x, -k, 0.0, hw);
                seriestype=:shape, color=C_KAPPA, linecolor=:white, linewidth=0.3, label=lbl)
            labeled[:kap] = true
        end
    end

    scatter!(fig_up, xs_up, r_land_up;
        marker=:diamond, markersize=6, color=:black, label="land rent")
    hline!(fig_up, [0.0]; color=:black, linewidth=0.8, label="")

    # group labels: y_grp is just below κ bar, above ylim bottom
    for grp in unique(filter(!isempty, grplbls_up))
        idxs = findall(g -> g == grp, grplbls_up)
        mid = (minimum(xs_up[idxs]) + maximum(xs_up[idxs])) / 2
        annotate!(fig_up, mid, y_grp, text(grp, :center, 9))
    end

    #savefig(fig_up, joinpath(OUTPUT_DIR, "decomp_upstream_plot.png"))
    println("✓ decomp_upstream_plot.png saved")
    display(fig_up)
end

# =================================================================================
# 8. Plot downstream zero-profit decomposition (4x3 grid, 11 fuel goods)
# =================================================================================

begin
    using Plots

    # ── bar layout (same x positions as upstream) ─────────────────────────────────
    bar_spec_dn = [
        ("Status Quo", 1.0, "SQ", ""),
        ("First Best CarbonTax", 2.0, "1st CT", ""),
        ("EquivEmission_CarbonTax", 3.2, "CT", "3B"),
        ("EquivEmission_RFS", 4.0, "RFS", "3B"),
        ("EquivEmission_LCFS", 4.8, "LCFS", "3B"),
        ("EquivEmission_TaxCredit", 5.6, "TC", "3B"),
        ("EquivEmission_5B_CarbonTax", 6.8, "CT", "5B"),
        ("EquivEmission_5B_RFS", 7.6, "RFS", "5B"),
        ("EquivEmission_5B_LCFS", 8.4, "LCFS", "5B"),
        ("EquivEmission_5B_TaxCredit", 9.2, "TC", "5B"),
    ]

    # ── 11 fuel goods to plot ─────────────────────────────────────────────────────
    # (good_str_in_csv, subplot_title)
    # ATJ SAF and Soy HEFA SAF: use conv or CS depending on which exists per scenario
    # represented as a list of candidate goods — first with qty>0 wins per scenario
    goods_to_plot = [
        (["jet_fuel"], "Jet Fuel"),
        (["saf_atj_conv", "saf_atj_cs"], "ATJ SAF"),
        (["saf_hefa_conv", "saf_hefa_cs"], "Soy HEFA SAF"),
        (["saf_hefa_nonsoy"], "Non-Soy HEFA SAF"),
        (["gasoline"], "Gasoline"),
        (["ethanol"], "Ethanol"),
        (["diesel"], "Diesel"),
        (["rd_soy"], "Soy RD"),
        (["rd_nonsoy"], "Non-Soy RD"),
        (["biodiesel_soy"], "Soy BD"),
        (["biodiesel_nonsoy"], "Non-Soy BD"),
    ]

    # ── stack components ──────────────────────────────────────────────────────────
    # positive: ProcessMC, FeedstockCost+ByproductDeduct (net), PA_NonsoyCapacity
    # can be negative: PA_CarbonTax, PA_RFS_avi, PA_LCFS, PA_TaxCredit,
    #                  PA_RFS_road, PA_BlendWall
    COMP_DEFS = [
        # (label,             cols_to_sum,                                    always_pos)
        ("Processing cost", [:ProcessMC], true),
        ("Feedstock cost", [:FeedstockCost, :ByproductDeduct], true),
        ("SAF Policy cost", [:PA_CarbonTax, :PA_RFS_avi, :PA_LCFS, :PA_TaxCredit], false),
        ("Road RFS constraint", [:PA_RFS_road], false),
        ("Blendwall constraint", [:PA_BlendWall], false),
        ("Non-Soy Capacity constraint", [:PA_NonsoyCapacity], true),
    ]

    COMP_COLORS = [
        colorant"#909090",   # Processing   — dark grey
        colorant"#F0C14B",   # Feedstock    — corn yellow
        colorant"#C44E52",   # SAF Policy   — red
        colorant"#A8C8A0",   # Road RFS     — muted green
        colorant"#B8B0D0",   # Blend Wall   — muted purple
        colorant"#A8CCD8",   # Non-Soy Cap  — muted cyan
    ]

    BAR_W = 0.55
    csv_scen_dn = unique(df_down.Scenario)

    # ── helper: rect shape ────────────────────────────────────────────────────────
    function rect_dn(x_center, y_bottom, y_top, hw)
        l = x_center - hw
        r = x_center + hw
        Shape([l, r, r, l, l], [y_bottom, y_bottom, y_top, y_top, y_bottom])
    end

    # ── build 4x3 layout ─────────────────────────────────────────────────────────
    n_rows = 3
    n_cols = 4
    fig_dn = plot(
        layout=grid(n_rows, n_cols),
        size=(1600, 900),
        left_margin=8Plots.px,
        right_margin=4Plots.px,
        top_margin=4Plots.px,
        bottom_margin=25Plots.px,
        legend=false,
    )

    xs_all = [b[2] for b in bar_spec_dn]
    xlbls_all = [b[3] for b in bar_spec_dn]

    # global legend proxy (drawn on panel 1 only)
    legend_added = falses(length(COMP_DEFS))

    for (pi, (good_candidates, gtitle)) in enumerate(goods_to_plot)

        sub = filter(r -> r.Good in good_candidates, df_down)

        # ── collect bar values ────────────────────────────────────────────────────
        xs_g = Float64[]
        xlbls_g = String[]
        grps_g = String[]
        # comp_vals[c][i] = net value of component c for bar i
        comp_vals = [Float64[] for _ in COMP_DEFS]
        price_g = Float64[]

        for (scen, xpos, xlbl, grp) in bar_spec_dn
            # pick whichever candidate good has qty>0 for this scenario
            row = DataFrame()
            for cand in good_candidates
                cand_row = filter(r -> r.Scenario == scen && r.Good == cand, df_down)
                if nrow(cand_row) > 0
                    row = cand_row
                    break
                end
            end
            if nrow(row) == 0
                # leave gap — push NaN so x positions stay consistent
                push!(xs_g, xpos)
                push!(xlbls_g, xlbl)
                push!(grps_g, grp)
                push!(price_g, NaN)
                for k in eachindex(COMP_DEFS)
                    push!(comp_vals[k], NaN)
                end
                continue
            end
            push!(xs_g, xpos)
            push!(xlbls_g, xlbl)
            push!(grps_g, grp)
            push!(price_g, row[1, :Price_per_unit])
            for (k, (_, cols, _)) in enumerate(COMP_DEFS)
                push!(comp_vals[k], sum(row[1, c] for c in cols))
            end
        end

        n_bars = length(xs_g)

        # y range
        pos_top = [sum(max(comp_vals[k][i], 0.0) for k in eachindex(COMP_DEFS);
            init=0.0) for i in 1:n_bars if !isnan(comp_vals[1][i])]
        neg_bot = [sum(min(comp_vals[k][i], 0.0) for k in eachindex(COMP_DEFS);
            init=0.0) for i in 1:n_bars if !isnan(comp_vals[1][i])]
        ymax_g = isempty(pos_top) ? 10.0 : max(maximum(pos_top), maximum(filter(!isnan, price_g))) * 1.15
        ymin_raw = isempty(neg_bot) ? 0.0 : minimum(neg_bot)
        # always show at least 8% of ymax as negative space so small negatives are visible
        ymin_g = min(ymin_raw * 1.15, -ymax_g * 0.08)

        plot!(fig_dn[pi];
            title=gtitle,
            titlefontsize=8,
            tickfontsize=6,
            xlims=(0.5, 9.9),
            ylims=(ymin_g, ymax_g),
            xticks=(xs_g, xlbls_g),
            grid=:y, gridalpha=0.3,
            legend=false,
        )

        hline!(fig_dn[pi], [0.0]; color=:black, linewidth=0.6, label="")

        # ── draw bars ─────────────────────────────────────────────────────────────
        for i in 1:n_bars
            isnan(comp_vals[1][i]) && continue   # skip missing scenarios
            x = xs_g[i]
            hw = BAR_W / 2

            pos_bottom = 0.0
            neg_bottom = 0.0

            for (k, (lbl, _, always_pos)) in enumerate(COMP_DEFS)
                v = comp_vals[k][i]
                isnan(v) && continue
                col = COMP_COLORS[k]

                if always_pos || v >= 0
                    # stack above zero
                    plot!(fig_dn[pi], rect_dn(x, pos_bottom, pos_bottom + v, hw);
                        seriestype=:shape, color=col,
                        linecolor=:white, linewidth=0.2, label="")
                    pos_bottom += v
                else
                    # stack below zero
                    plot!(fig_dn[pi], rect_dn(x, neg_bottom + v, neg_bottom, hw);
                        seriestype=:shape, color=col,
                        linecolor=:white, linewidth=0.2, label="")
                    neg_bottom += v
                end
            end
        end

        # price_per_unit diamond
        valid_xs = [xs_g[i] for i in 1:n_bars if !isnan(price_g[i])]
        valid_pr = [price_g[i] for i in 1:n_bars if !isnan(price_g[i])]
        scatter!(fig_dn[pi], valid_xs, valid_pr;
            marker=:diamond, markersize=4, color=:black, label="")

        # group label (3B / 5B): place just inside bottom of ylims, below all bars
        grp_y = ymin_g + (ymax_g - ymin_g) * 0.04   # 4% up from bottom
        for grp in unique(filter(!isempty, grps_g))
            idxs = findall(g -> g == grp, grps_g)
            mid = (minimum(xs_g[idxs]) + maximum(xs_g[idxs])) / 2
            annotate!(fig_dn[pi], mid, grp_y, text(grp, :center, 6))
        end
    end

    # ── legend panel (slot 12, empty otherwise) ───────────────────────────────────
    plot!(fig_dn[12]; axis=false, grid=false, ticks=false, background_color=:white)

    # draw legend manually as colored rectangles + text
    legend_x = 0.1
    legend_y_start = 0.85
    dy = 0.13
    entries = [(lbl, col) for ((lbl, _, _), col) in zip(COMP_DEFS, COMP_COLORS)]
    push!(entries, ("\$/gallon", colorant"#000000"))

    for (j, (lbl, col)) in enumerate(entries)
        y = legend_y_start - (j - 1) * dy
        if lbl == "\$/gallon"
            scatter!(fig_dn[12], [legend_x + 0.05], [y];
                marker=:diamond, markersize=6, color=col,
                xlims=(0, 1), ylims=(0, 1), label="")
        else
            plot!(fig_dn[12],
                Shape([legend_x, legend_x + 0.1, legend_x + 0.1, legend_x, legend_x],
                    [y - 0.04, y - 0.04, y + 0.04, y + 0.04, y - 0.04]);
                seriestype=:shape, color=col, linecolor=:white,
                xlims=(0, 1), ylims=(0, 1), label="")
        end
        annotate!(fig_dn[12], legend_x + 0.15, y, text(lbl, :left, 8))
    end

    #savefig(fig_dn, joinpath(OUTPUT_DIR, "decomp_downstream_plot.png"))
    println("✓ decomp_downstream_plot.png saved")
    display(fig_dn)
end