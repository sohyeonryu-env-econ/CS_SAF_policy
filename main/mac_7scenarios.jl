# mac_7scenarios.jl
#
# MAC (marginal abatement cost) curves for the seven scenarios (b) to (i) of
# 3B_outcomes.jl, on one figure. Left panel = private MAC, right panel = social MAC.
#
# All seven (policy, case) combinations used in the tables and figures go on one panel:
# color marks the policy family, line style marks CS crediting and the threshold.
#
# Scenario letters are those of SCEN in 3B_outcomes.jl:
#   (a) status quo, which has no MAC by construction
#   (d) RFS crediting CS, dropped because its solution matches (c) and the curves overlap
#
# Nothing is read from jld2: every solution is re-solved on a grid sweep. See POLICY_GRID
# for the grid sizes.

cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "scenarios.jl"))
using .Scenarios
include(joinpath(@__DIR__, "model_mkt.jl"))
include(joinpath(@__DIR__, "analysis.jl"))

import .ModelMkt: params, build_unified_model, extract_solution, is_solved_and_feasible
import .Analysis: calculate_emissions_detail, calculate_implicit_taxes,
    calculate_cs_changes, calculate_ps_land_changes, calculate_ps_nonsoy_changes,
    calculate_ps_fossil_changes,
    calculate_gr_changes, calculate_environmental_benefit, calculate_total_welfare

using JuMP, Printf, CairoMakie, JLD2, CSV, DataFrames

include(joinpath(@__DIR__, "paths.jl"))
using .Paths
Paths.setup()

const FIGURE_DIR = Paths.FIGURE_DIR
const OUTPUT_DIR = Paths.DATA_DIR
const SCC = 190.0

# The common-abatement anchor of 3B_outcomes.jl, drawn as a vertical dashed line.
# Read from the benchmark file rather than hard coded: a model change moves the anchor,
# and a constant would leave the dashed line at the old spot, disagreeing with the tables.
const ANCHOR_MT = let f = joinpath(OUTPUT_DIR, "results_unified_benchmark.jld2")
    isfile(f) ? round(JLD2.load(f, "bench_info").target_reduction * 1000, digits=2) : 40.45
end

# =================================================================================
# 1. Scenario definitions
# =================================================================================
#
# color is the policy family, linestyle the design (crediting CS / threshold).
# grid is that policy's stringency sweep. The upper limit is set at roughly 70 Mt of
# abatement. The figure's x limit is 60 Mt, so there is 17% of headroom; going wider means
# solving off screen.
#
# Stringency that reaches 70 Mt (bisection, taken from the slowest case):
#   carbontax 310.6   rfs 0.615   lcfs 0.127   taxcredit 1138.1
# The limits are those rounded up. To widen, raise these numbers; the grid is part of the
# spec, so the cache re-solves on its own.

# Attach color, line style and sweep grid to SCEN from scenarios.jl.
# (a) status quo has no MAC by construction, and (d) matches (c) closely enough that the
# curves overlap, so both are dropped. Flatten label: the legend must be single-line.
const SCEN7 = [merge(s, Scenarios.CASE_FLAGS[s.case],
                   (label=Scenarios.one_line(s), color=c, style=st, grid=g))
               for (s, (c, st, g)) in zip(
    Scenarios.except(Scenarios.SCEN, "(a)", "(d)"),
    [(:blue, :solid, 0.0:1.0:350.0),
     (:red, :solid, 0.0:0.002:0.7),
     (:red, :dash, 0.0:0.002:0.7),
     (:green, :solid, 0.0:0.001:0.15),
     (:green, :dash, 0.0:0.001:0.15),
     (:purple, :solid, 0.0:8.0:1300.0),
     (:purple, :dash, 0.0:8.0:1300.0)])]

mkcfg(policy, v; recognize_cs, use_ci_threshold) = (
    t=policy === :carbontax ? v : 0.0,
    θ_avi=policy === :rfs ? v : 0.0,
    σ=policy === :lcfs ? v : 0.0,
    p=policy === :taxcredit ? v : 0.0,
    carbon_tax_scope=:aviation,
    use_ci_threshold=use_ci_threshold,
    recognize_cs=recognize_cs,
)

# =================================================================================
# 2. Solve one scenario grid and build its MAC
# =================================================================================

# solve_point(config, policy): solve the model at one stringency and return it with the
# fields the welfare calculation needs. Returns nothing if it does not solve.
function solve_point(config, policy)
    model = build_unified_model(params, config)
    optimize!(model)
    is_solved_and_feasible(model) || return nothing
    sol = extract_solution(model, policy)
    return merge(sol, (
        emissions=calculate_emissions_detail(sol, params),
        implicit_taxes=calculate_implicit_taxes(sol, params, config),
    ))
end

# scenario_mac(s): solve the whole stringency grid of scenario `s` and build private and
# social MAC from finite differences of adjacent points.
# Returns (abatement_mt, mac_private, mac_social).
#
# MAC is defined as delta welfare / delta emissions. Abatement makes delta emissions
# negative, so MAC > 0 wherever welfare falls, and reads as cost per ton.
function scenario_mac(s)
    kw = (recognize_cs=s.recognize_cs, use_ci_threshold=s.use_ci_threshold)

    sq = solve_point(mkcfg(s.policy, 0.0; kw...), :statusquo)
    isnothing(sq) && error("$(s.tag): the status quo did not solve.")
    sq_em = sq.emissions.total

    keys_ = Symbol[]
    sols = Dict{Symbol,Any}(:statusquo => sq)
    vals = Float64[]
    for (i, v) in enumerate(s.grid)
        sol = solve_point(mkcfg(s.policy, v; kw...), s.policy)
        isnothing(sol) && continue
        # The key MUST be "<policy>_<i>". calculate_gov_revenue_change in analysis.jl
        # picks the government revenue term by matching the scenario NAME STRING
        # (startswith "carbontax" / "taxcredit"). A name like :p1 silently zeroes GR and
        # the whole private MAC comes out wrong.
        k = Symbol("$(s.policy)_$(i)")
        sols[k] = sol
        push!(keys_, k)
        push!(vals, v)
    end
    @printf("  %-4s %-36s solved %4d / %4d\n", s.tag, s.label, length(keys_), length(s.grid))

    cs = calculate_cs_changes(sols, sq, params; scenarios=keys_)
    psl = calculate_ps_land_changes(sols, sq, params; scenarios=keys_)
    psn = calculate_ps_nonsoy_changes(sols, sq, params; scenarios=keys_)
    psf = calculate_ps_fossil_changes(sols, sq, params; scenarios=keys_)
    gr = calculate_gr_changes(sols; scenarios=keys_)
    env = calculate_environmental_benefit(sols, sq, SCC; scenarios=keys_)
    w = calculate_total_welfare(cs, psl, gr, env; ps_nonsoy_changes=psn, ps_fossil_changes=psf,
        scenarios=keys_)

    ab, mp, ms = Float64[], Float64[], Float64[]
    for i in 2:length(keys_)
        a, b = keys_[i], keys_[i-1]
        Δem = sols[a].emissions.total - sols[b].emissions.total
        abs(Δem) < 1e-9 && continue          # stretch where more stringency does not move emissions
        push!(ab, (sq_em - sols[a].emissions.total) * 1000)
        push!(mp, (w[a].private_surplus - w[b].private_surplus) / Δem)
        push!(ms, (w[a].social_welfare - w[b].social_welfare) / Δem)
    end

    idx = sortperm(ab)
    return (ab=ab[idx], private=mp[idx], social=ms[idx])
end

# smooth(v, window): damp the residual jitter left in the finite-difference MAC with a
# centered moving average. The finer the grid, the smaller both the numerator and the
# denominator of delta welfare / delta emissions, and the larger the numerical noise.
# window = 1 leaves the raw values.
function smooth(v, window)
    window <= 1 && return v
    h = window ÷ 2
    return [mean_of(v, max(1, i - h):min(length(v), i + h)) for i in eachindex(v)]
end
mean_of(v, r) = sum(@view v[r]) / length(r)

# =================================================================================
# 3. Computation
# =================================================================================

println("\n" * "="^80)
println("MAC grid sweep (7 scenarios)")
println("="^80)

const SMOOTH_WINDOW = 5

# Grid cache.
#
# The seven scenario grids come to about 2,457 points, each solving the model and the
# welfare calculation. But the common edit is a line color or an axis range followed by a
# re-run. So the MAC curves are stored alongside a spec (per-scenario grid and design,
# plus a version), and a matching spec on the next run skips the re-solve. Same approach
# as fig_rpm_lcfs.jl.
#
# Note: changes to the model itself are NOT part of the spec. After editing model_mkt.jl
# or the welfare functions in analysis.jl, bump CACHE_VERSION.
const CACHE_FILE = "results_mac_7scenarios.jld2"
const CACHE_VERSION = 1

function load_or_build_curves()
    cache_path = joinpath(OUTPUT_DIR, CACHE_FILE)
    want = (scen=[(s.tag, s.policy, s.recognize_cs, s.use_ci_threshold,
                   first(s.grid), step(s.grid), last(s.grid)) for s in SCEN7],
        v=CACHE_VERSION, scc=SCC)

    if isfile(cache_path)
        cached = JLD2.load(cache_path)
        if haskey(cached, "curves") && get(cached, "spec", nothing) == want
            println("Cache reused: ", cache_path)
            return cached["curves"]
        end
        println("Cache does not match the current spec, recomputing.")
    end

    t0 = time()
    cs = [(s=s, d=scenario_mac(s)) for s in SCEN7]
    @printf("\nTotal time %.1f s\n", time() - t0)

    spec = want
    curves = cs
    @save cache_path curves spec
    println("Saved: ", cache_path)
    return cs
end

curves = load_or_build_curves()

# =================================================================================
# 4. Figure: one social MAC panel
# =================================================================================
#
# private is social shifted by SCC (social = private - SCC), so the shapes match. The
# paper figure keeps social only. Switch field=:private to draw the other.
#
# The legend stacks one column per policy family: (b) | (c),(e) | (f),(g) | (h),(i).
# Makie fills nbanks column-first, so a blank entry after (b) empties the bottom slot of
# the first column.

"""
    mac_ylims(curves, field, xmax; pad, step)

y range that holds every curve out to `xmax`, rounded outward to a multiple of `step`.

Fixing the range at (-200, 200) clipped the tax credit curves the moment the model changed:
their MAC keeps climbing past the old ceiling.  Reading the range off the data means the
window follows the curves instead of cutting them.

`step` is the rounding unit of the *window*, not the tick spacing.  Rounding out to the tick
spacing left almost a full tick of dead air above and below the curves, so the window snaps
to a much finer grid and the ticks are laid inside it separately (see `plot_mac`).
"""
function mac_ylims(curves, field, xmax; pad=0.02, step=10.0)
    lo, hi = Inf, -Inf
    for c in curves
        y = smooth(getproperty(c.d, field), SMOOTH_WINDOW)
        keep = c.d.ab .<= xmax
        any(keep) || continue
        lo = min(lo, minimum(y[keep]))
        hi = max(hi, maximum(y[keep]))
    end
    (isfinite(lo) && isfinite(hi)) || return (-200.0, 200.0)
    m = (hi - lo) * pad
    return (floor((lo - m) / step) * step, ceil((hi + m) / step) * step)
end

function plot_mac(; field=:social, xmax=60.0, ylims=nothing, ystep=100.0)
    ylims = isnothing(ylims) ? mac_ylims(curves, field, xmax) : ylims
    # Ticks are set apart from the axis limits. The limits round to a finer unit than
    # ystep, so ylims[1]:ystep:ylims[2] would put ticks off the multiples of 100 (-195, -95, ...).
    yt = (ceil(ylims[1] / ystep)*ystep):ystep:(floor(ylims[2] / ystep)*ystep)
    fig = Figure(size=(1000, 520), fontsize=14)

    ax = Axis(fig[1, 1];
        xlabel="Cumulative abatement (Mt CO₂e)",
        ylabel="Social MAC (\$/tonne CO₂e)",
        xticks=0:20:xmax, yticks=yt,
        xgridvisible=true, ygridvisible=true,
        topspinevisible=false, rightspinevisible=false,
        limits=((0, xmax), ylims))
    hlines!(ax, [0.0]; color=(:black, 0.4), linewidth=0.8)
    # The anchor line has to be distinct from both the grid lines and the curves. A faint
    # dotted line disappears into the grid, and :dash is the pattern of the CS-crediting
    # curves and would read as data. So :dashdot, which no curve uses, heavy and dark.
    vlines!(ax, [ANCHOR_MT]; color=(:black, 0.55), linewidth=2.0, linestyle=:dashdot)
    text!(ax, ANCHOR_MT, ylims[2] - 0.03 * (ylims[2] - ylims[1]); text=" anchor $(ANCHOR_MT) Mt",
        align=(:left, :top), fontsize=11, color=:gray20)

    for c in curves
        y = smooth(getproperty(c.d, field), SMOOTH_WINDOW)
        keep = c.d.ab .<= xmax
        lines!(ax, c.d.ab[keep], y[keep];
            color=c.s.color, linestyle=c.s.style, linewidth=2.0)
    end

    # Legend layout: one column per policy family. blank holds the empty slot in column 1.
    by_tag = Dict(c.s.tag => c for c in curves)
    blank = LineElement(color=:transparent)
    order = ["(b)", nothing, "(c)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    elems = [isnothing(t) ? blank :
             LineElement(color=by_tag[t].s.color, linestyle=by_tag[t].s.style, linewidth=2.5)
             for t in order]
    labels = [isnothing(t) ? "" : "$(t) $(by_tag[t].s.label)" for t in order]

    # Center the title and legend on the FIGURE. The y label sits in the axis protrusion
    # rather than in the column width, so both drift right if left alone (as in 3B_outcomes.jl).
    Makie.update_state_before_display!(fig)
    lp = ax.layoutobservables.protrusions[].left
    rp = ax.layoutobservables.protrusions[].right
    full = Outside(-lp, -rp, 0, 0)

    Legend(fig[2, 1], elems, labels;
        orientation=:horizontal, nbanks=2, framevisible=true,
        framecolor=:gray65, framewidth=0.8, tellwidth=false, alignmode=full,
        labelsize=11, colgap=14, padding=(12, 12, 6, 6))

    return fig
end

fig_mac = plot_mac()
save(joinpath(FIGURE_DIR, "fig_mac_7scenarios.png"), fig_mac; px_per_unit=3)
save(joinpath(FIGURE_DIR, "fig_mac_7scenarios.pdf"), fig_mac)
println("✓ Saved: ", joinpath(FIGURE_DIR, "fig_mac_7scenarios.png"))

# Print the MAC near the anchor so it can be checked against the AAC in the tables.
println("\nMAC near the anchor ($(ANCHOR_MT) Mt):")
@printf("  %-4s %-36s %12s %12s\n", "", "", "private", "social")
for c in curves
    isempty(c.d.ab) && continue
    i = argmin(abs.(c.d.ab .- ANCHOR_MT))
    @printf("  %-4s %-36s %12.1f %12.1f\n", c.s.tag, c.s.label,
        smooth(c.d.private, SMOOTH_WINDOW)[i], smooth(c.d.social, SMOOTH_WINDOW)[i])
end

# =================================================================================
# 5. Save the curve data (so numbers can be pulled back out for tables and text)
# =================================================================================

const RESULT_DIR = Paths.TABLE_DIR

open(joinpath(RESULT_DIR, "mac_7scenarios.csv"), "w") do io
    println(io, "tag,label,abatement_mt,mac_private,mac_social,mac_private_raw,mac_social_raw")
    for c in curves
        sp = smooth(c.d.private, SMOOTH_WINDOW)
        ss = smooth(c.d.social, SMOOTH_WINDOW)
        for i in eachindex(c.d.ab)
            println(io, join((c.s.tag, "\"$(c.s.label)\"", c.d.ab[i],
                    sp[i], ss[i], c.d.private[i], c.d.social[i]), ","))
        end
    end
end
println("✓ Saved: ", joinpath(RESULT_DIR, "mac_7scenarios.csv"))
