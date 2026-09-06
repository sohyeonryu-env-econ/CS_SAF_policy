# implicit_tax_cases.jl
# Implicit tax / subsidy curves by case (recognize_cs x use_ci_threshold).
#
#
# To run: include it from the same folder as model_mkt.jl / analysis.jl.

cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "model_mkt.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "units.jl"))     # metric reporting; model stays in US units
using .Units

import .ModelMkt: params, build_unified_model, extract_solution
import .Analysis: calculate_implicit_taxes
using Printf, Plots, JuMP, JLD2

include(joinpath(@__DIR__, "paths.jl"))
using .Paths
Paths.setup()

const FIGURE_DIR = Paths.FIGURE_DIR
const DATA_DIR = Paths.DATA_DIR

# Grid cache. The sweep is stored with a spec (grid + case design + units + version), and
# a matching spec on the next run skips the re-solve. Same approach as fig_rpm_lcfs.jl.
#
# Note: model changes are NOT part of the spec. After editing model_mkt.jl or
# calculate_implicit_taxes in analysis.jl, bump CACHE_VERSION.
const CACHE_FILE = "results_implicit_tax_cases.jld2"
const CACHE_VERSION = 1

# =================================================================================
# 1. Sweep grids
# =================================================================================
const THETA_GRID = 0.0:0.005:0.70    # RFS aviation mandate (about 70 Mt of abatement at theta = 0.615)
const SIGMA_GRID = 0.0:0.0025:0.15   # LCFS standard      (about 70 Mt of abatement at sigma = 0.127)

mkcfg(; t=0.0, θ_avi=0.0, σ=0.0, p=0.0, recognize_cs, use_ci_threshold) = (
    t=t, θ_avi=θ_avi, σ=σ, p=p,
    carbon_tax_scope=:aviation,
    use_ci_threshold=use_ci_threshold,
    recognize_cs=recognize_cs,
)

# =================================================================================
# 2. Solve one case x one policy sweep
#    Returns: (xs, Dict(fuel => implicit tax component vector))
# =================================================================================
const AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
    :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

function sweep(policy::Symbol, grid, recognize_cs, use_ci_threshold)
    key = policy == :rfs ? :rfs_avi : :lcfs
    xs = Float64[]
    ys = Dict(g => Float64[] for g in AVIATION_FUELS)
    nfail = 0
    for v in grid
        config = policy == :rfs ?
                 mkcfg(; θ_avi=Float64(v), recognize_cs=recognize_cs, use_ci_threshold=use_ci_threshold) :
                 mkcfg(; σ=Float64(v), recognize_cs=recognize_cs, use_ci_threshold=use_ci_threshold)
        local sol
        try
            model = build_unified_model(params, config)
            optimize!(model)
            if !is_solved_and_feasible(model)
                nfail += 1
                continue
            end
            sol = extract_solution(model, Symbol("$(policy)_$(v)"))
        catch e
            nfail += 1
            continue
        end
        it = calculate_implicit_taxes(sol, params, config)
        push!(xs, Float64(v))
        for g in AVIATION_FUELS
            push!(ys[g], Units.price_gal_to_L(it[g][key]))
        end
    end
    @printf("    %-5s  recognize_cs=%-5s use_ci_threshold=%-5s : solved %d / %d\n",
        policy, recognize_cs, use_ci_threshold, length(xs), length(grid))
    return (xs=xs, ys=ys)
end

# load_or_sweep(name, policy, grid, cases): solve every entry of `cases` over `grid` and
# return Dict(case => sweep result). A matching cache is reused. `name` is the key that
# separates the two figures (rfs / lcfs) inside the cache file.
function load_or_sweep(name, policy, grid, cases)
    cache_path = joinpath(DATA_DIR, CACHE_FILE)
    want = (grid=(first(grid), step(grid), last(grid)),
        cases=[(c[1], c[2], c[3]) for c in cases],
        v=CACHE_VERSION, metric=Units.METRIC)

    store = isfile(cache_path) ? JLD2.load(cache_path) : Dict{String,Any}()
    if get(store, name * "_spec", nothing) == want && haskey(store, name * "_res")
        println("Cache reused: ", name, "  ", cache_path)
        return store[name*"_res"]
    end

    res = Dict(c[1] => sweep(policy, grid, c[2], c[3]) for c in cases)

    # One file holds both rfs and lcfs, so read the existing content, merge, and rewrite.
    store[name*"_res"] = res
    store[name*"_spec"] = want
    JLD2.jldopen(cache_path, "w") do f
        for (k, v) in store
            f[k] = v
        end
    end
    println("Saved: ", name, "  ", cache_path)
    return res
end

# =================================================================================
# 3. Figures
#    Under the RFS every fuel earns the same credit, -1.6*gamma, so the eligible fuel lines
#    coincide exactly. Line widths are stepped and styles mixed so all of them stay visible.
# =================================================================================
const FUEL_STYLE = [
    (:jet_fuel, "Jet Fuel", :black, :solid),
    (:saf_atj_conv, "Conv ATJ-SAF", :blue, :solid),
    (:saf_atj_cs, "CS ATJ-SAF", :red, :dash),
    (:saf_hefa_conv, "Conv HEFA-SAF", :green, :solid),
    (:saf_hefa_cs, "CS HEFA-SAF", :orange, :dash),
    (:saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple, :dot),
]
# Overlap handling: lines drawn later are thinner
const LW = Dict(:jet_fuel => 4.5, :saf_atj_conv => 7.0, :saf_atj_cs => 5.5,
    :saf_hefa_conv => 4.0, :saf_hefa_cs => 3.0, :saf_hefa_nonsoy => 2.0)

function make_legend_panel(items; ncols=length(items), fontsize=15)
    p = plot(legend=:top, legendcolumns=ncols, grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none, legendfontsize=fontsize)
    for (g, label, color, ls) in items
        plot!(p, [NaN], [NaN], label=label, linewidth=4, color=color, linestyle=ls)
    end
    return p
end

# ys is already converted where it is built, so the shared y limits follow automatically.
const IT_YLABEL = Units.METRIC ? "Implicit Tax/Subsidy (\$/liter)" :
                  "Implicit Tax/Subsidy (\$/gallon)"

function case_panel(res, ptitle, xlabel, ylabel, xlims_cur, ylims_cur; show_y=true)
    p = plot(title=ptitle, titlefontsize=21, titlefontweight=:bold,
        xlabel=xlabel, ylabel=show_y ? ylabel : "",
        legend=false, grid=true, xlims=xlims_cur, ylims=ylims_cur,
        left_margin=show_y ? 20Plots.mm : 6Plots.mm,
        right_margin=8Plots.mm, bottom_margin=14Plots.mm, top_margin=8Plots.mm,
        guidefontsize=17, tickfontsize=14)
    hline!(p, [0], color=:gray, linestyle=:dot, linewidth=1.5)
    for (g, _, color, ls) in FUEL_STYLE
        isempty(res.xs) && continue
        plot!(p, res.xs, res.ys[g], linewidth=LW[g], color=color, linestyle=ls)
    end
    return p
end

# =================================================================================
# 4. Figure 1: RFS aviation, 3 cases
# =================================================================================
println("\n[Volumetric mandate] sweeping θ_avi ...")
rfs_cases = [
    (:case1, false, false, "Case 1: no CS, no CI threshold"),
    (:case3, true, false, "Case 3: recognize CS, no CI threshold"),
    (:case4, true, true, "Case 4: recognize CS, 50% CI threshold"),
]
rfs_res = load_or_sweep("rfs", :rfs, THETA_GRID, rfs_cases)

# One shared y axis across the three panels (the point is comparing magnitudes across cases)
rfs_all = vcat([vcat([r.ys[g] for g in AVIATION_FUELS]...) for r in values(rfs_res)]...)
rfs_ylims = let lo = minimum(rfs_all), hi = maximum(rfs_all), pad = 0.08 * (maximum(rfs_all) - minimum(rfs_all))
    (lo - pad, hi + pad)
end
@printf("  Volumetric mandate ylims = (%.2f, %.2f)\n", rfs_ylims...)

fig_rfs = plot(
    plot([case_panel(rfs_res[c[1]], c[4], "Volumetric mandate (θ_avi)",
            IT_YLABEL, (0.0, maximum(THETA_GRID)), rfs_ylims;
            show_y=(i == 1)) for (i, c) in enumerate(rfs_cases)]..., layout=(1, 3)),
    make_legend_panel(FUEL_STYLE, ncols=6),
    layout=grid(2, 1, heights=[0.9, 0.1]), size=(2100, 750),
    plot_title="Implicit Tax / Subsidy under the Volumetric mandate",
    plot_titlefontsize=24, plot_titlefontweight=:bold)
display(fig_rfs)
savefig(fig_rfs, joinpath(FIGURE_DIR, "implicit_tax_rfs_cases.png"))
println("  saved: ", joinpath(FIGURE_DIR, "implicit_tax_rfs_cases.png"))

# =================================================================================
# 5. Figure 2: LCFS, 2 cases
# =================================================================================
println("\n[CI standard] sweeping σ ...")
lcfs_cases = [
    (:case1, false, false, "Case 1: no CS, no CI threshold"),
    (:case3, true, false, "Case 3: recognize CS, no CI threshold"),
]
lcfs_res = load_or_sweep("lcfs", :lcfs, SIGMA_GRID, lcfs_cases)

lcfs_all = vcat([vcat([r.ys[g] for g in AVIATION_FUELS]...) for r in values(lcfs_res)]...)
lcfs_ylims = let lo = minimum(lcfs_all), hi = maximum(lcfs_all), pad = 0.08 * (hi - lo)
    (lo - pad, hi + pad)
end
@printf("  CI standard ylims = (%.2f, %.2f)\n", lcfs_ylims...)

lcfs_xmax = maximum(maximum(r.xs) for r in values(lcfs_res))
fig_lcfs = plot(
    plot([case_panel(lcfs_res[c[1]], c[4], "CI standard (σ)",
            IT_YLABEL, (0.0, lcfs_xmax), lcfs_ylims;
            show_y=(i == 1)) for (i, c) in enumerate(lcfs_cases)]..., layout=(1, 2)),
    make_legend_panel(FUEL_STYLE, ncols=3),
    layout=grid(2, 1, heights=[0.85, 0.15]), size=(1500, 830),
    plot_title="Implicit Tax / Subsidy under the CI standard",
    plot_titlefontsize=24, plot_titlefontweight=:bold)
display(fig_lcfs)
savefig(fig_lcfs, joinpath(FIGURE_DIR, "implicit_tax_lcfs_cases.png"))
println("  saved: ", joinpath(FIGURE_DIR, "implicit_tax_lcfs_cases.png"))

# =================================================================================
# 6. Case differences as numbers (the lines overlap, so the figure alone does not show them)
# =================================================================================
function report(res_dict, cases, label, xlab, probe)
    println("\n" * "="^96)
    println("$label : implicit tax/subsidy at $(xlab) = $(probe)  (\$/gallon)")
    println("="^96)
    @printf("%-18s", "fuel")
    for c in cases
        @printf("%22s", String(c[1]))
    end
    println()
    for (g, lbl, _, _) in FUEL_STYLE
        @printf("%-18s", lbl)
        for c in cases
            r = res_dict[c[1]]
            i = argmin(abs.(r.xs .- probe))
            @printf("%22.4f", r.ys[g][i])
        end
        println()
    end
end

report(rfs_res, rfs_cases, "Volumetric mandate", "θ_avi", 0.30)
report(rfs_res, rfs_cases, "Volumetric mandate", "θ_avi", 0.90)
report(lcfs_res, lcfs_cases, "CI standard", "σ", 0.10)
report(lcfs_res, lcfs_cases, "CI standard", "σ", 0.20)

println("\nDone.")
