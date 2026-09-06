# fig_stacked_paths.jl
#
# Redraws the content of Figure 1 (the bars in fig_jet_saf_by_scenario) as
# stringency paths.
#
#   Bar figure: the jet fuel / SAF mix at one point per scenario, the point of equal
#               abatement.
#   This one  : the same scenarios followed continuously from the status quo (= (a))
#               up to that point. The right-hand edge of each panel is that row of the
#               bar figure.
#
#   Two versions:
#     levels : stacked areas in levels (B gal). The y axis is cut to 15-21 so the SAF
#              bands are readable (no scenario carries information in 0-15; an axis
#              break mark is drawn). The heavy black line on top of the stack is total
#              aviation fuel.
#     change : change from the status quo. Below zero, the fall in fossil jet fuel;
#              above zero, SAF added. The heavy dashed line (right axis) is the change
#              in aviation demand (RPM).
#
#   Three knobs control size and layout:
#     Figure(size=...)  overall size
#     MAX_PANEL_COLS    how many reference columns the widest panel in a row spans
#                       (raise it and the figure grows sideways)
#     NGRID             grid points per panel (lower it and steps smear into diagonals)
#
#   Colors are exactly those of aviation_config / plot_fuel_production_stacked in
#   extended_grid.jl (jet fuel lightgray fill with a black outline, SAF at fillalpha
#   0.7, same stacking order).
#
# Run first:
#   julia main/unified_benchmark.jl   ->  results_unified_benchmark.jld2
#
# Output:
#   FIGURE_DIR/fig_stacked_paths_levels.png/.pdf
#   FIGURE_DIR/fig_stacked_paths_change.png/.pdf
#   DATA_DIR/results_stacked_paths.jld2   (grid solution cache, reused if the spec matches)

cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "scenarios.jl"))
using .Scenarios
include(joinpath(@__DIR__, "model_mkt.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
include(joinpath(@__DIR__, "units.jl"))     # metric reporting; model stays in US units
using .Units

import .ModelMkt: params, build_unified_model, extract_solution, is_solved_and_feasible
import .Analysis: calculate_emissions_detail

using JLD2, JuMP, Printf, CairoMakie

include(joinpath(@__DIR__, "paths.jl"))
using .Paths
Paths.setup()

const OUTPUT_DIR = Paths.DATA_DIR
const FIGURE_DIR = Paths.FIGURE_DIR
const BENCH_FILE = "results_unified_benchmark.jld2"
const CACHE_FILE = "results_stacked_paths.jld2"

# The grid has to be fine. The IRA panel has a genuine discontinuity (non-soy HEFA jumps
# from 0 to 0.6 B gal), and on a coarse grid that jump takes a whole cell and is drawn as
# a diagonal. A point solves in about 2 ms, so 401 points x 8 panels takes under 10 s.
const NGRID = 401         # grid points per panel (evenly spaced from 0 to x_max)
const CACHE_VERSION = 4   # bump when the cached content changes
                          # (v2: added aviation demand RPM, v3: fossil CES supply + redefined
                          #  tax credit + rho_pretreat, v4: cache stored in metric. Units are
                          #  now part of the spec, so a US <-> metric toggle alone no longer
                          #  needs a bump.)

# =================================================================================
# 1. Anchor data: the stringency at which each scenario hits the common abatement is the x limit
# =================================================================================

isfile(joinpath(OUTPUT_DIR, BENCH_FILE)) ||
    error("$(BENCH_FILE) not found. Run unified_benchmark.jl first.")

@load joinpath(OUTPUT_DIR, BENCH_FILE) all_case_results base_results bench_info

const SQ = base_results[:statusquo]
const SQ_EMISSIONS = SQ.emissions.total
const TARGET_REDUCTION_MT = bench_info.target_reduction * 1000     # M ton CO2e

# =================================================================================
# 2. Panel definitions: names and order match the bar figure (SCEN in 3B_outcomes.jl)
# =================================================================================
#
# (a) status quo has no panel: x = 0 in every panel is (a).
# One row per policy family: carbon tax / three RFS / two LCFS / two IRA.

# Panels in the same family share a tick format (only the x limit differs, not the unit).
const POLICY_XTICKS = Dict(
    :rfs => 0:0.05:0.30,
    :lcfs => 0:0.02:0.08,
)   # IRA axes are not unified, so their ticks are left automatic per panel

const POLICY_XLABEL = Dict(
    :carbontax => "Carbon tax (\$/tonne CO₂e)",
    :rfs => "Volumetric mandate (θ_avi)",
    :lcfs => "CI standard (σ)",
    :taxcredit => "Tax credit (\$/tonne CO₂e)",
)

# Panel titles fold after the colon. With the policy name spelled "Volumetric mandate",
# a single line runs past the panel width and collides with the neighbouring title. The
# console log and the width check use s.label directly, so they stay on one line.
wrap_title(lab) = replace(lab, ": " => ":\n")

# Attach panel placement (row/col) to SCEN from scenarios.jl. (a) status quo is the
# starting point of every path, not a panel of its own, so it is dropped.
# Flatten label to one line: panel titles must be single-line.
const SCEN = [merge(s, (label=Scenarios.one_line(s), row=rc[1], col=rc[2]))
              for (s, rc) in zip(Scenarios.except(Scenarios.SCEN, "(a)"),
    [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (4, 1), (4, 2)])]

# Per-case (credit CS, CI threshold) settings, same numbering as unified_benchmark.jl
const CASE_FLAGS = Scenarios.CASE_FLAGS

make_config(policy, value, recognize_cs, use_ci_threshold) =
    (t=policy == :carbontax ? value : 0.0,
        θ_avi=policy == :rfs ? value : 0.0,
        σ=policy == :lcfs ? value : 0.0,
        p=policy == :taxcredit ? value : 0.0,
        use_ci_threshold=use_ci_threshold,
        recognize_cs=recognize_cs)

stringency_of(policy, config) =
    policy == :carbontax ? config.t :
    policy == :rfs ? config.θ_avi :
    policy == :lcfs ? config.σ : config.p

# The stringency at which each scenario hits the common abatement (the dashed line)
x_target_of(s) = stringency_of(s.policy,
    all_case_results[s.case].equivalent_policies[s.policy].config)

# Each panel is drawn only up to its own target. Within a row (same policy, same unit)
# panel width is instead made proportional to the x range, so "pixels per dollar" or
# "per 0.01 sigma" is constant along the row. Unifying the axis range and leaving the
# right side empty would equalise tick spacing but only add whitespace; giving every
# panel the same width would hide differences like the IRA 22.85 versus 8.49. With
# proportional widths (h) is drawn 2.7 times as long as (i) and the gap is visible.
# (This is ggplot facet + scales="free_x", space="free_x".)
x_max_of(s) = x_target_of(s)

# =================================================================================
# 3. Grid computation (cached)
# =================================================================================
#
# Colors and order follow aviation_config in extended_grid.jl. Bottom to top:
#   jet fuel (lightgray), non-soy HEFA (purple), CS HEFA (orange), conv HEFA (green),
#   → CS ATJ(red) → conv ATJ(blue)

const JET = :jet_fuel
const JET_LABEL = "Fossil jet fuel"
const JET_FILL = :lightgray
const JET_LINE = :black
const FILL_ALPHA = 0.7

const SAF_PATHWAYS = [
    (:saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple),
    (:saf_hefa_cs, "CS HEFA-SAF", :orange),
    (:saf_hefa_conv, "Conv HEFA-SAF", :green),
    (:saf_atj_cs, "CS ATJ-SAF", :red),
    (:saf_atj_conv, "Conv ATJ-SAF", :blue),
]
const STACK_GOODS = vcat([JET], [g for (g, _, _) in SAF_PATHWAYS])

# solve_path(s, x_max, n): solve scenario s at n points of stringency from 0 to x_max
# and return the aviation fuel mix (jet fuel + 5 SAF pathways) and aviation passenger
# demand (RPM). Points that fail to solve stay NaN so the figure breaks there.
function solve_path(s, x_max, n)
    flags = CASE_FLAGS[s.case]
    xs = collect(range(0.0, x_max, length=n))
    q = fill(NaN, n, length(STACK_GOODS))
    rpm = fill(NaN, n)
    last_sol = nothing
    for (i, xv) in enumerate(xs)
        config = make_config(s.policy, xv, flags.recognize_cs, flags.use_ci_threshold)
        model = build_unified_model(params, config)
        optimize!(model)
        is_solved_and_feasible(model) || (@warn "infeasible" tag = s.tag x = xv; continue)
        sol = extract_solution(model, s.policy)
        for (k, g) in enumerate(STACK_GOODS)
            q[i, k] = Units.gal_to_L(sol.q[g])
        end
        rpm[i] = Units.mile_to_km(sol.x[:avi])
        i == n && (last_sol = sol)
    end
    # End-point check: does it really finish at the common abatement?
    red_mt = isnothing(last_sol) ? NaN :
             (SQ_EMISSIONS - calculate_emissions_detail(last_sol, params).total) * 1000
    return (xs=xs, q=q, rpm=rpm, end_reduction_mt=red_mt)
end

function load_or_build_paths()
    cache_path = joinpath(OUTPUT_DIR, CACHE_FILE)
    # The cache holds values already converted by Units. Units belong in the spec so a
    # US <-> metric toggle invalidates it automatically. Without that, a cache filled in
    # gallons is drawn on a litre axis and the data falls below the panel, leaving it blank.
    want = Dict(s.tag => (x_max=x_target_of(s), n=NGRID, v=CACHE_VERSION,
                          metric=Units.METRIC) for s in SCEN)

    if isfile(cache_path)
        cached = JLD2.load(cache_path)
        if haskey(cached, "paths") && haskey(cached, "spec") && cached["spec"] == want
            println("Cache reused: ", cache_path)
            return cached["paths"]
        end
        println("Cache does not match the current spec, recomputing.")
    end

    paths = Dict{String,Any}()
    for s in SCEN
        xm = x_target_of(s)
        t0 = time()
        paths[s.tag] = solve_path(s, xm, NGRID)
        @printf("  %s %-34s x_max = %10.5f   end-point abatement = %7.3f Mt   (%.1fs)\n",
            s.tag, s.label, xm, paths[s.tag].end_reduction_mt, time() - t0)
    end
    spec = want
    @save cache_path paths spec
    println("Saved: ", cache_path)
    return paths
end

println("\nCommon target abatement = $(round(TARGET_REDUCTION_MT, digits=3)) Mt CO2e")
println("Grid computation per panel ($(NGRID) points/panel)")
paths = load_or_build_paths()

# Summary quantities (used to set axis ranges straight from the data)
saf_total(P) = vec(sum(view(P.q, :, 2:size(P.q, 2)), dims=2))
total_fuel(P) = view(P.q, :, 1) .+ saf_total(P)

const SQ_JET = paths["(b)"].q[1, 1]          # x = 0 is the status quo in every panel
const SQ_RPM = paths["(b)"].rpm[1]
const SQ_TOTAL = SQ_JET

@printf("status quo: jet = %.3f %s, air travel = %.1f B %s\n",
    SQ_JET, Units.METRIC ? "B liters" : "B gal", SQ_RPM, Units.METRIC ? "passenger-km" : "RPM")

# =================================================================================
# 4. Shared drawing helpers
# =================================================================================

const TARGET_LABEL = @sprintf("%.2f Mt CO₂e", TARGET_REDUCTION_MT)
const MARK_COLOR = RGBf(0.55, 0.05, 0.05)

fill_c(c) = (c, FILL_ALPHA)

# Mark where the common abatement is reached. Label the first panel only, dashes elsewhere.
function mark_target!(ax, xt, y_label; show_label=false)
    vlines!(ax, [xt]; color=MARK_COLOR, linestyle=:dash, linewidth=2.0)
    show_label && text!(ax, xt, y_label; text=TARGET_LABEL, align=(:right, :top),
        offset=(-6, -4), fontsize=13, color=MARK_COLOR)
end

# Break mark at the bottom of the y axis (to show it does not start at zero).
function break_mark!(ax, xm, y_lo)
    w = xm * 0.022
    h = (Y_HI_L - y_lo) * 0.055
    y0 = y_lo + h * 0.35
    poly!(ax, Rect2f(-w, y0 - h * 0.75, 2.4w, 1.5h); color=:white, strokewidth=0)
    for dy in (-h * 0.28, h * 0.28)
        lines!(ax, [-w, w], [y0 + dy - h * 0.35, y0 + dy + h * 0.35];
            color=:black, linewidth=1.1)
    end
end

# Row layout. A row is one policy family, and inside it panel width is proportional to the x range.
const ROW_PANELS = [[s for s in SCEN if s.row == r] for r in 1:maximum(t.row for t in SCEN)]

# build_rows!(gp): one independent GridLayout per row. Width allocation happens later,
# in size_rows!, after the axes are added.
build_rows!(gp) = [GridLayout(gp[r, 1]) for r in eachindex(ROW_PANELS)]

# One width rule handles every row.
#   Within a row, panel width is proportional to the x range (same row = same unit, so
#   tick spacing matches).
#   The row as a whole is sized so its widest panel equals one reference column.
# Rows with fewer panels therefore leave the right side empty instead of stretching the
# figure. The reference column is the figure width over GRID_COLS; raising MAX_PANEL_COLS
# widens everything.
const GRID_COLS = 3.0
const MAX_PANEL_COLS = 1.0

function size_rows!(rows)
    for (r, panels) in enumerate(ROW_PANELS)
        ws = [x_max_of(s) for s in panels]
        rel = ws ./ maximum(ws) .* MAX_PANEL_COLS
        for (i, w) in enumerate(rel)
            colsize!(rows[r], i, Auto(w))
        end
        slack = GRID_COLS - sum(rel)
        if slack > 0.02
            Box(rows[r][1, length(panels)+1]; color=:transparent, strokevisible=false)
            colsize!(rows[r], length(panels) + 1, Auto(slack))
        end
    end
end

# For the width check: panel tag -> Axis
const PANEL_AXES = Dict{String,Any}()

# report_widths!(fig): check that one unit of stringency really occupies the same number
# of pixels within a row. The layout has to be resolved once for the values to exist, so
# call this after the figure is built.
function report_widths!(fig)
    Makie.update_state_before_display!(fig)
    println("\nPanel width check (px per unit must match within a row)")
    narrow = String[]
    for panels in ROW_PANELS
        for s in panels
            ax = get(PANEL_AXES, s.tag, nothing)
            isnothing(ax) && continue
            w = widths(ax.scene.viewport[])[1]
            w < 60 && push!(narrow, s.tag)
            @printf("  %s %-34s x_max=%9.5f  width=%6.1f px  %10.2f px/unit\n",
                s.tag, s.label, x_max_of(s), w, w / x_max_of(s))
        end
    end
    isempty(narrow) ||
        error("Panel columns were squeezed ($(join(narrow, ", "))). Widen the figure or " *
              "check elements that demand width (Label/Legend with tellwidth=true).")
end

# Which slot the panel occupies within its row.
col_in_row(s) = findfirst(t -> t.tag == s.tag, ROW_PANELS[s.row])

# The legend has two rows: fossil jet fuel and the total line on top, the five SAF
# pathways below. Seven items on one line run past the figure width and get cut off on
# the right. nbanks=2 fills column-first and cannot split 2/5, so stack two Legends and
# draw one box around both to separate them from the body (as in 3B_outcomes.jl).
#
# Both Legends and the GridLayout holding them use tellwidth=false. Letting them demand
# width once produced a saved figure with the panel columns squeezed to zero.
function add_legend!(fig, row; extra_elems=[], extra_labels=[])
    leg_kw = (orientation=:horizontal, nbanks=1, framevisible=false,
        labelsize=15, patchsize=(20, 14), colgap=16, tellwidth=false,
        padding=(12, 12, 3, 3))

    gl = GridLayout(fig[row, 1]; tellwidth=false, halign=:center)

    Legend(gl[1, 1],
        vcat([PolyElement(color=fill_c(JET_FILL), strokecolor=JET_LINE, strokewidth=1.2)],
            extra_elems),
        vcat([JET_LABEL], extra_labels); leg_kw...)

    Legend(gl[2, 1],
        [PolyElement(color=fill_c(c), strokecolor=c, strokewidth=1.2) for (_, _, c) in SAF_PATHWAYS],
        [l for (_, l, _) in SAF_PATHWAYS]; leg_kw...)

    rowgap!(gl, 1, 0)
    Box(gl[1:2, 1]; color=:transparent, strokecolor=:gray65, strokewidth=0.8)
    return gl
end

# =================================================================================
# 5. Figure A: stacked areas in levels (y axis cut to 15-21)
# =================================================================================

const Y_LO_L = Units.gal_to_L(15.0)
const Y_HI_L = Units.gal_to_L(21.4)

function draw_levels_panel!(ax, s)
    P = paths[s.tag]
    xs, q = P.xs, P.q
    ok = .!isnan.(view(q, :, 1))

    # The axis is cut at Y_LO_L, so the jet fuel area runs from that floor up to q_jet.
    #
    # Areas (bands) are drawn over the whole range without a mask. Where a pathway is
    # zero the upper and lower edges coincide and nothing shows, and a jump from zero to
    # positive is filled with no gap. Selecting only q>0 points, as before, left a whole
    # cell empty between the last zero and the first positive point, a white canyon.
    # Outlines are different: over a zero stretch they would overplot the line below, so
    # they are broken with NaN.
    lower = fill(Y_LO_L, length(xs))
    for (k, _) in enumerate(STACK_GOODS)
        fill_col = k == 1 ? JET_FILL : SAF_PATHWAYS[k-1][3]
        line_col = k == 1 ? JET_LINE : SAF_PATHWAYS[k-1][3]
        v = view(q, :, k)
        upper = k == 1 ? collect(v) : lower .+ v
        band!(ax, xs, lower, upper; color=fill_c(fill_col))
        drawn = k == 1 ? copy(upper) : [(ok[i] && v[i] > 1e-10) ? upper[i] : NaN for i in eachindex(upper)]
        lines!(ax, xs, drawn; color=line_col, linewidth=1.8)
        lower = upper
    end

    # Top of the stack = total aviation fuel in gallons. Its decline shows falling demand
    # for air travel, but gallons and passenger miles are not 1:1. The aviation market
    # clearing condition is
    #     x[:avi] = r_jet * (q_jet + beta * sum(SAF)),   r_jet = 58.958 RPM/gal, beta = 0.973425
    # so a SAF gallon delivers only 97.34% of the transport service of a jet fuel gallon.
    # A larger SAF share therefore needs more gallons for the same passenger miles and
    # this line sits slightly above travel. The two IRA panels show it: passenger miles
    # are pinned at 1208.97 while total gallons rise by 0.032 in (h) and 0.077 in (i),
    # almost exactly sum(SAF)*(1/beta - 1). To read travel directly, look at
    # q_jet + beta*sum(SAF), which is exactly proportional to RPM.
    lines!(ax, xs, lower; color=:black, linewidth=3.2)
end

function plot_levels()
    fig = Figure(size=(1120, 1280), fontsize=15)
    gp = GridLayout(fig[1, 1])
    rows = build_rows!(gp)

    for s in SCEN
        xm = x_max_of(s)
        ax = Axis(rows[s.row][1, col_in_row(s)];
            title="$(s.tag) $(wrap_title(s.label))", titlesize=17, titlealign=:left,
            xlabel=POLICY_XLABEL[s.policy],
            ylabel=s.col == 1 ? (Units.METRIC ? "Aviation fuel (billion liters)" :
                                                 "Aviation fuel (billion gallons)") : "",
            xlabelsize=15, ylabelsize=15,
            xticklabelsize=13, yticklabelsize=13,
            xticks=haskey(POLICY_XTICKS, s.policy) ?
                   collect(POLICY_XTICKS[s.policy]) : Makie.automatic,
            yticks=Units.METRIC ? (60:5:80) : (16:1:21),
            topspinevisible=false, rightspinevisible=false,
            limits=((0, xm * 1.005), (Y_LO_L, Y_HI_L)))
        PANEL_AXES[s.tag] = ax
        draw_levels_panel!(ax, s)
        mark_target!(ax, x_target_of(s), Y_HI_L; show_label=(s.tag == "(b)"))
        break_mark!(ax, xm, Y_LO_L)
    end

    size_rows!(rows)
    add_legend!(fig, 2;
        extra_elems=[LineElement(color=:black, linewidth=3.2)],
        extra_labels=["Total aviation fuel (volume)"])

    Label(fig[0, 1],
        "Aviation fuel mix along each policy path,\n" *
        "from the status quo to the common abatement of $(TARGET_LABEL)";
        fontsize=18, font=:bold, justification=:center,
        tellwidth=false, padding=(0, 0, 0, 6))

    rowgap!(gp, 20)
    for gl in rows
        length(contents(gl)) > 1 && colgap!(gl, 18)
    end
    rowgap!(fig.layout, 2, 10)
    return fig
end

# =================================================================================
# 6. Figure B: change from the status quo
# =================================================================================
#
# Below zero: the fall in fossil jet fuel (gray).  Above zero: SAF added (stacked by pathway).
# Dashed line (right axis): change in aviation passenger demand (RPM), the demand response.
#
# The gap between the lower edge of the gray area (the fall in fossil jet fuel) and the
# top of the SAF stack is the net change in total fuel, which is the demand response.
# Drawing that net change as its own solid line was tried and dropped: it was
# indistinguishable from the black outline of the gray area. Instead the right axis is
# scaled by the status quo RPM/gal ratio, so the dashed line reads against the left fuel
# scale. (Because energy densities differ by fuel, the actual RPM change and the
# converted net fuel change diverge by up to about 7% at the end point.)

const Y_LO_C = Units.gal_to_L(-4.6)
const Y_HI_C = Units.gal_to_L(3.4)
const RPM_COLOR = RGBf(0.15, 0.15, 0.15)

# The right-hand RPM axis sits on the left fuel axis, with ticks converted at the status quo ratio.
const RPM_PER_GAL = SQ_RPM / SQ_JET
# The right-hand travel axis is drawn on the left axis's scale via RPM_PER_GAL, so its
# ticks are in travel units and need their own metric spacing.
const RPM_TICKS = Units.METRIC ? (-400:100:200) : (-250:50:150)

function draw_change_panel!(ax, s)
    P = paths[s.tag]
    xs, q = P.xs, P.q
    ok = .!isnan.(view(q, :, 1))
    zero_line = zeros(length(xs))

    # Below zero: the fall in fossil jet fuel
    djet = view(q, :, 1) .- SQ_JET
    band!(ax, xs, djet, zero_line; color=fill_c(JET_FILL))
    lines!(ax, xs, djet; color=JET_LINE, linewidth=1.8)

    # Above zero: SAF stacked (areas over the full range, lines only where positive, as in the levels panel)
    lower = zeros(length(xs))
    for (k, (_, _, col)) in enumerate(SAF_PATHWAYS)
        v = view(q, :, k + 1)
        upper = lower .+ v
        band!(ax, xs, lower, upper; color=fill_c(col))
        drawn = [(ok[i] && v[i] > 1e-10) ? upper[i] : NaN for i in eachindex(upper)]
        lines!(ax, xs, drawn; color=col, linewidth=1.8)
        lower = upper
    end

    hlines!(ax, [0.0]; color=:gray40, linewidth=0.8)
end

# Actual change in aviation passenger demand (RPM), dashed on the right axis.
function draw_rpm!(axr, s)
    P = paths[s.tag]
    ok = .!isnan.(P.rpm)
    lines!(axr, P.xs[ok], P.rpm[ok] .- SQ_RPM;
        color=RPM_COLOR, linewidth=3.2, linestyle=:dash)
end

function plot_change()
    fig = Figure(size=(1120, 1280), fontsize=15)
    gp = GridLayout(fig[1, 1])
    rows = build_rows!(gp)

    for s in SCEN
        xm = x_max_of(s)
        ax = Axis(rows[s.row][1, col_in_row(s)];
            title="$(s.tag) $(wrap_title(s.label))", titlesize=17, titlealign=:left,
            xlabel=POLICY_XLABEL[s.policy],
            ylabel=s.col == 1 ? (Units.METRIC ? "Change in aviation fuel (billion liters)" :
                                                 "Change in aviation fuel (billion gallons)") : "",
            xlabelsize=15, ylabelsize=15,
            xticklabelsize=13, yticklabelsize=13,
            xticks=haskey(POLICY_XTICKS, s.policy) ?
                   collect(POLICY_XTICKS[s.policy]) : Makie.automatic,
            yticks=Units.METRIC ? (-15:5:10) : (-4:1:3),
            topspinevisible=false,
            limits=((0, xm * 1.005), (Y_LO_C, Y_HI_C)))

        # Right axis: the same data read as aviation demand (RPM)
        show_r = s.col == maximum(t.col for t in SCEN if t.row == s.row)
        axr = Axis(rows[s.row][1, col_in_row(s)];
            yaxisposition=:right,
            ylabel=show_r ? (Units.METRIC ? "Change in air travel (billion passenger-km)" :
                                            "Change in air travel (billion RPM)") : "",
            ylabelsize=15, yticklabelsize=13,
            yticks=(collect(RPM_TICKS), [string(v) for v in RPM_TICKS]),
            limits=((0, xm * 1.005), (Y_LO_C * RPM_PER_GAL, Y_HI_C * RPM_PER_GAL)))
        hidespines!(axr)
        hidexdecorations!(axr)
        axr.yticklabelsvisible = show_r
        axr.yticksvisible = show_r
        axr.ygridvisible = false

        draw_change_panel!(ax, s)
        draw_rpm!(axr, s)
        mark_target!(ax, x_target_of(s), Y_HI_C; show_label=(s.tag == "(b)"))
        translate!(ax.blockscene, 0, 0, 10)
        translate!(axr.blockscene, 0, 0, 20)
    end

    size_rows!(rows)
    add_legend!(fig, 2;
        extra_elems=[LineElement(color=RPM_COLOR, linewidth=3.2, linestyle=:dash)],
        extra_labels=["Air travel, demand response (right axis)"])

    Label(fig[0, 1],
        "Fossil jet fuel displaced, SAF added, and demand response along each policy path,\n" *
        "to the common abatement of $(TARGET_LABEL)";
        fontsize=18, font=:bold, justification=:center,
        tellwidth=false, padding=(0, 0, 0, 6))

    rowgap!(gp, 20)
    for gl in rows
        length(contents(gl)) > 1 && colgap!(gl, 22)
    end
    rowgap!(fig.layout, 2, 10)
    return fig
end

# =================================================================================
# 7. Save
# =================================================================================

mkpath(FIGURE_DIR)

fig_l = plot_levels()
save(joinpath(FIGURE_DIR, "fig_stacked_paths_levels.png"), fig_l; px_per_unit=3)
save(joinpath(FIGURE_DIR, "fig_stacked_paths_levels.pdf"), fig_l)
report_widths!(fig_l)   # measured after saving: the layout is only resolved at render time
println("✓ Saved: ", joinpath(FIGURE_DIR, "fig_stacked_paths_levels.png"))

fig_c = plot_change()
save(joinpath(FIGURE_DIR, "fig_stacked_paths_change.png"), fig_c; px_per_unit=3)
save(joinpath(FIGURE_DIR, "fig_stacked_paths_change.pdf"), fig_c)
println("✓ Saved: ", joinpath(FIGURE_DIR, "fig_stacked_paths_change.png"))
