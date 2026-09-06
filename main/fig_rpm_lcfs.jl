# fig_rpm_lcfs.jl
#
# Change in aviation passenger demand (RPM) as the CI standard (LCFS) tightens.
#
#   x axis: sigma (CI standard), over the range that reads as policy (0 to ZOOM_MAX)
#   y axis: change in aviation passenger demand from the status quo, in levels. One
#           left axis only; percentages are in the console table and in pct_f / pct_g
#           of the CSV
#   lines : (f) CI standard: without crediting CS   [case1, recognize_cs=false]
#           (g) CI standard: crediting CS           [case3, recognize_cs=true ]
#
#   Case numbers and labels follow fig_stacked_paths.jl / unified_benchmark.jl.
#   Color (green) and line style (solid = no CS, dashed = CS) follow mac_7scenarios.jl.
#
# ---------------------------------------------------------------------------------
# The kink at sigma about 0.026
# ---------------------------------------------------------------------------------
# It is a jump in slope, not a discontinuity: the marginal pathway changes there.
#
#   sigma <= 0.026 : SAF comes from non-soy HEFA alone and the two cases coincide
#                    (crediting CS does not change that pathway's CI). As that pathway
#                    hits its capacity (q about 0.62 B gal), lambda_lcfs climbs steeply,
#                    265 -> 300 -> 420.
#   sigma >  0.026 : a second pathway has to enter, and which one differs by case.
#                    (f) takes conventional ATJ, (g) takes CS ATJ (plus a little CS HEFA).
#                    Crediting CS meets the same sigma at a lower lambda_lcfs (378 versus
#                    459 at sigma = 0.04, about 18% lower). That lowers the pass-through
#                    into aviation fuel prices, so (g) loses less RPM than (f).
#
# ---------------------------------------------------------------------------------
# Could extended_grid.jl be reused?
# ---------------------------------------------------------------------------------
# Only half of it. The LCFS grid there is
#
#     (t=0.0, theta_avi=0.0, sigma=sigma, p=0.0, use_ci_threshold=false, recognize_cs=true)
#
# with recognize_cs pinned true. So that grid holds only case3 = (g); there is no
# recognize_cs=false grid for (f). (The rfsnoci_* runs turn off use_ci_threshold, not CS
# crediting.) On top of that its sigma key is `lcfs_$(round(Int, sigma*1000))` while the
# grid step is 0.0003, so keys collide and only 501 of 1668 grid points survive. And the
# aviation panel (plot_price_paired_2x2) draws the price p_c[:avi], not demand.
#
# So both cases are re-solved directly on a shared sigma grid. At 11 ms per point,
# 1201 points x 2 cases finishes in under 30 seconds.
#
# Output:
#   FIGURE_DIR/fig_rpm_lcfs.png / .pdf
#   DATA_DIR/results_rpm_lcfs.jld2    (grid solution cache, reused if the spec matches)
#   TABLE_DIR/results_rpm_lcfs.csv    (exactly the numbers drawn)

cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "model_mkt.jl"))
include(joinpath(@__DIR__, "units.jl"))     # metric reporting; model stays in US units
using .Units

import .ModelMkt: params, build_unified_model, extract_solution, is_solved_and_feasible

using JLD2, JuMP, Printf, DataFrames, CSV, CairoMakie

include(joinpath(@__DIR__, "paths.jl"))
using .Paths
Paths.setup()

const DATA_DIR   = Paths.DATA_DIR
const TABLE_DIR  = Paths.TABLE_DIR
const FIGURE_DIR = Paths.FIGURE_DIR
const BENCH_FILE = "results_unified_benchmark.jld2"
const CACHE_FILE = "results_rpm_lcfs.jld2"
const CSV_FILE = "results_rpm_lcfs.csv"

# Sigma grid. The upper limit 0.3 matches the LCFS sweep in mac_7scenarios.jl. Both cases
# solve out to 0.5 (checked), but past 0.3 RPM is below -40%, which is no longer a policy
# range. The figure shows up to ZOOM_MAX; the rest survives only in the CSV and the
# console table.
const SIGMA_MAX = 0.15
const NGRID = 601             # evenly spaced from 0 to SIGMA_MAX (step 0.00025)
const ZOOM_MAX = 0.08         # sigma limit in the figure (same as the LCFS ticks in fig_stacked_paths.jl).
                              # The grid is solved out to SIGMA_MAX, so changing this alone needs no re-solve.
                              # SIGMA_MAX 0.15 is set at the roughly 70 Mt abatement point (sigma = 0.127),
                              # about twice ZOOM_MAX, leaving room to widen the figure.
const CACHE_VERSION = 1

# =================================================================================
# 1. Case definitions
# =================================================================================

const CASES = [
    (tag="(f)", case=:case1, label="CI standard: without crediting CS",
        recognize_cs=false, use_ci_threshold=false,
        color=RGBf(0.05, 0.42, 0.20), style=:solid),
    (tag="(g)", case=:case3, label="CI standard: crediting CS",
        recognize_cs=true, use_ci_threshold=false,
        color=RGBf(0.35, 0.70, 0.36), style=:dash),
]

mkcfg(σ; recognize_cs, use_ci_threshold) = (
    t=0.0, θ_avi=0.0, σ=Float64(σ), p=0.0,
    carbon_tax_scope=:aviation,
    use_ci_threshold=use_ci_threshold,
    recognize_cs=recognize_cs)

# =================================================================================
# 2. Grid computation (cached)
# =================================================================================

# solve_sigma_path(c, sigma_s): solve case c over the sigma grid and return aviation
# passenger demand. Points that fail to solve stay NaN so the figure breaks there.
# Values are returned in reporting units, through Units (B passenger-km under metric).
function solve_sigma_path(c, σs)
    rpm = fill(NaN, length(σs))
    λ = fill(NaN, length(σs))
    for (i, σ) in enumerate(σs)
        model = build_unified_model(params,
            mkcfg(σ; recognize_cs=c.recognize_cs, use_ci_threshold=c.use_ci_threshold))
        optimize!(model)
        is_solved_and_feasible(model) || (@warn "infeasible" tag = c.tag σ = σ; continue)
        sol = extract_solution(model, :lcfs)
        rpm[i] = Units.mile_to_km(sol.x[:avi])
        λ[i] = sol.duals.λ_lcfs
    end
    return (rpm=rpm, λ=λ)
end

function load_or_build_paths()
    cache_path = joinpath(DATA_DIR, CACHE_FILE)
    # Units belong in the spec so a US <-> metric toggle invalidates the cache
    # automatically (same reason as fig_stacked_paths.jl).
    want = (σ_max=SIGMA_MAX, n=NGRID, v=CACHE_VERSION, metric=Units.METRIC,
        tags=[c.tag for c in CASES])

    if isfile(cache_path)
        cached = JLD2.load(cache_path)
        if haskey(cached, "paths") && haskey(cached, "spec") && cached["spec"] == want
            println("Cache reused: ", cache_path)
            return cached["σs"], cached["paths"]
        end
        println("Cache does not match the current spec, recomputing.")
    end

    σs = collect(range(0.0, SIGMA_MAX, length=NGRID))
    paths = Dict{String,Any}()
    for c in CASES
        t0 = time()
        paths[c.tag] = solve_sigma_path(c, σs)
        @printf("  %s %-38s  %d points  (%.1fs)\n", c.tag, c.label, NGRID, time() - t0)
    end
    spec = want
    @save cache_path paths spec σs
    println("Saved: ", cache_path)
    return σs, paths
end

println("\nSigma grid: 0 to $(SIGMA_MAX), $(NGRID) points x $(length(CASES)) cases")
σs, paths = load_or_build_paths()

# The status quo is sigma = 0 in both cases (crediting CS does not move the no-policy equilibrium).
const SQ_RPM = paths["(f)"].rpm[1]
@assert isapprox(SQ_RPM, paths["(g)"].rpm[1]; rtol=1e-10) "the two cases differ at sigma = 0"

const TRAVEL_UNIT = Units.METRIC ? "billion passenger-km" : "billion RPM"
@printf("status quo air travel = %.1f %s\n", SQ_RPM, TRAVEL_UNIT)

Δrpm(tag) = paths[tag].rpm .- SQ_RPM
pct_rpm(tag) = 100 .* (paths[tag].rpm ./ SQ_RPM .- 1)

# =================================================================================
# 3. Point of common abatement (marked when available)
# =================================================================================
#
# unified_benchmark.jl has already solved the sigma at which each case reaches the common
# abatement. Without that file the marks are skipped and the figure is drawn as usual.

const bench = let f = joinpath(DATA_DIR, BENCH_FILE)
    if isfile(f)
        d = load(f)
        acr, bi = d["all_case_results"], d["bench_info"]
        σ_of = Dict{String,Float64}()
        for c in CASES
            ep = get(acr[c.case].equivalent_policies, :lcfs, nothing)
            isnothing(ep) || (σ_of[c.tag] = ep.config.σ)
        end
        (σ_of=σ_of, target_mt=bi.target_reduction * 1000)
    else
        println("$(BENCH_FILE) not found, skipping the common-abatement marks.")
        (σ_of=Dict{String,Float64}(), target_mt=NaN)
    end
end

# Value at the grid point nearest that sigma (so the marker sits exactly on the line)
function at_sigma(tag, σ)
    i = argmin(abs.(σs .- σ))
    return σs[i], Δrpm(tag)[i], pct_rpm(tag)[i]
end

# The sigma where the two cases start to diverge, found from the data rather than hard
# coded. Below it SAF is non-soy HEFA alone and crediting CS changes nothing.
const SPLIT_TOL = 0.5      # billion passenger-km
const σ_split = let d = abs.(Δrpm("(g)") .- Δrpm("(f)"))
    i = findfirst(v -> !isnan(v) && v > SPLIT_TOL, d)
    isnothing(i) ? NaN : σs[i]
end
@printf("\nCases diverge at sigma = %.4f (below it, non-soy HEFA only)\n", σ_split)

if !isempty(bench.σ_of)
    @printf("\nPoint of common abatement, %.2f Mt CO2e\n", bench.target_mt)
    for c in CASES
        haskey(bench.σ_of, c.tag) || continue
        _, d, p = at_sigma(c.tag, bench.σ_of[c.tag])
        @printf("  %s σ* = %.5f   ΔRPM = %+8.2f %s (%+.2f%%)\n",
            c.tag, bench.σ_of[c.tag], d, TRAVEL_UNIT, p)
    end
end

# =================================================================================
# 4. Console table and CSV
# =================================================================================

println("\nAviation passenger demand change by sigma (", TRAVEL_UNIT, ")")
println("   sigma    (f) delta   (f) %      (g) delta  (g) %     gap (g-f)")
for σ in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
    σ > SIGMA_MAX && continue
    _, df_, pf = at_sigma("(f)", σ)
    _, dg, pg = at_sigma("(g)", σ)
    @printf("  %.3f  %9.2f  %+8.2f%%  %9.2f  %+8.2f%%  %9.2f\n", σ, df_, pf, dg, pg, dg - df_)
end

let df = DataFrame(sigma=σs,
        rpm_f=paths["(f)"].rpm, d_rpm_f=Δrpm("(f)"), pct_f=pct_rpm("(f)"),
        rpm_g=paths["(g)"].rpm, d_rpm_g=Δrpm("(g)"), pct_g=pct_rpm("(g)"),
        lambda_f=paths["(f)"].λ, lambda_g=paths["(g)"].λ)
    CSV.write(joinpath(TABLE_DIR, CSV_FILE), df)
    println("\n✓ Saved: ", joinpath(TABLE_DIR, CSV_FILE))
end

# =================================================================================
# 5. Figure
# =================================================================================
#
# A single panel. Sigma is drawn only over the policy-relevant range (0 to ZOOM_MAX).
# There is one y axis, on the left, in level change (B passenger-km). Percentages are in
# the console table and in pct_f / pct_g of the CSV rather than on a right axis.
#
# The grid is still solved out to sigma = 0.3 and stored in the cache and the CSV.
# Widening or narrowing the drawn range is a change to ZOOM_MAX alone, with no re-solve.

const MARK_COLOR = RGBf(0.55, 0.05, 0.05)
const Y_LAB = "Change in air travel ($(TRAVEL_UNIT))"
const X_LAB = "CI standard (σ)"

# Set the y range from the data over that sigma range (4% padding).
function ylims_for(σ_hi)
    ok = σs .<= σ_hi * 1.0001
    lo = minimum(minimum(filter(!isnan, Δrpm(c.tag)[ok])) for c in CASES)
    pad = 0.04 * abs(lo)
    return (lo - pad, pad)
end

# The two ΔRPM lines (plus the common-abatement points and the split). One left axis.
function draw_panel!(gp, σ_hi; mark_target, show_split)
    ylo, yhi = ylims_for(σ_hi)

    ax = Axis(gp;
        xlabel=X_LAB, ylabel=Y_LAB,
        xlabelsize=15, ylabelsize=15, xticklabelsize=13, yticklabelsize=13,
        xticks=σ_hi <= 0.1 ? collect(0:0.02:σ_hi) : collect(0:0.05:σ_hi),
        topspinevisible=false, rightspinevisible=false,
        limits=((0, σ_hi * 1.002), (ylo, yhi)))

    hlines!(ax, [0.0]; color=(:black, 0.45), linewidth=0.8)

    # Where the two cases diverge. To the left of it the lines coincide exactly, and
    # without the mark the figure looks like a single dashed line.
    if show_split && isfinite(σ_split) && σ_split <= σ_hi
        vlines!(ax, [σ_split]; color=(:gray45, 0.8), linestyle=:dash, linewidth=1.4)
        text!(ax, σ_split, yhi;
            text="paths separate\n(non-soy HEFA capacity binds)",
            align=(:left, :top), offset=(6, -4), fontsize=11, color=:gray30)
    end

    ok = σs .<= σ_hi * 1.0001
    for c in CASES
        d = Δrpm(c.tag)
        good = ok .& .!isnan.(d)
        lines!(ax, σs[good], d[good];
            color=c.color, linestyle=c.style, linewidth=3.0)
    end

    # Common abatement: a vertical dashed line at each case sigma* and a dot on the line.
    if mark_target
        for c in CASES
            haskey(bench.σ_of, c.tag) || continue
            σx, dy, _ = at_sigma(c.tag, bench.σ_of[c.tag])
            (σx <= σ_hi) || continue
            vlines!(ax, [σx]; color=(MARK_COLOR, 0.55), linestyle=:dot, linewidth=1.6)
            scatter!(ax, [σx], [dy]; color=MARK_COLOR, markersize=11,
                strokecolor=:white, strokewidth=1.2)
            text!(ax, σx, dy; text=c.tag, align=(:right, :top),
                offset=(-8, -4), fontsize=13, color=MARK_COLOR)
        end
        # Text inside the figure stays in English. The Makie default font (TeX Gyre Heros)
        # has no Hangul glyphs, so Korean text kills the render step.
        text!(ax, 0.02, ylo;
            text=@sprintf("markers: common abatement of %.2f Mt CO\u2082e", bench.target_mt),
            align=(:left, :bottom), offset=(0, 8), fontsize=12, color=MARK_COLOR)
    end

    return ax
end

function plot_rpm_lcfs()
    fig = Figure(size=(760, 560), fontsize=15)

    draw_panel!(fig[1, 1], ZOOM_MAX;
        mark_target=!isempty(bench.σ_of), show_split=true)

    elems = [LineElement(color=c.color, linestyle=c.style, linewidth=3.0) for c in CASES]
    labels = ["$(c.tag) $(c.label)" for c in CASES]
    Legend(fig[2, 1], elems, labels;
        orientation=:horizontal, nbanks=2, framevisible=true, framecolor=:gray65,
        framewidth=0.8, labelsize=14, patchsize=(34, 14), colgap=18,
        tellwidth=false, tellheight=true)

    Label(fig[0, 1],
        "Air travel response along the CI standard,\nwith and without crediting carbon-smart practices";
        fontsize=17, font=:bold, justification=:center,
        tellwidth=false, padding=(0, 0, 0, 6))

    rowgap!(fig.layout, 1, 6)
    rowgap!(fig.layout, 2, 10)
    return fig
end

mkpath(FIGURE_DIR)
fig = plot_rpm_lcfs()
save(joinpath(FIGURE_DIR, "fig_rpm_lcfs.png"), fig; px_per_unit=3)
save(joinpath(FIGURE_DIR, "fig_rpm_lcfs.pdf"), fig)
println("✓ Saved: ", joinpath(FIGURE_DIR, "fig_rpm_lcfs.png"))
