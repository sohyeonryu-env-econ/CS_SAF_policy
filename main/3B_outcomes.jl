# 3B_outcomes.jl

# Run first: julia main/unified_benchmark.jl   ->  results_unified_benchmark.jld2
#
# Output (TABLE_DIR / FIGURE_DIR):
#   table_A_aviation_3B.csv/.tex      aviation market outcomes + SAF mix
#   table_B_spillover_3B.csv/.tex     road transport and food spillovers
#   table_C_welfare_3B.csv/.tex       welfare decomposition + AAC
#   fig_jet_saf_by_scenario.png/.pdf  two-panel bar chart of jet fuel and SAF
#   fig_spillover_prices.png          land rent, crop price and VMT price changes
#   fig_land_use.png/.pdf             cropland by scenario (conventional / climate-smart)
#   fig_land_use_change.png/.pdf      the same, as change from the status quo

cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "scenarios.jl"))
using .Scenarios
include(joinpath(@__DIR__, "model_mkt.jl"))
include(joinpath(@__DIR__, "analysis.jl"))

import .ModelMkt: params
import .Analysis: SCC

include(joinpath(@__DIR__, "units.jl"))
using .Units

using JLD2, DataFrames, CSV, Printf, CairoMakie

include(joinpath(@__DIR__, "paths.jl"))
using .Paths
Paths.setup()

const DATA_DIR = Paths.DATA_DIR
const TABLE_DIR = Paths.TABLE_DIR
const FIGURE_DIR = Paths.FIGURE_DIR
const BENCH_FILE = "results_unified_benchmark.jld2"

# =================================================================================
# 1. Load the anchor data
# =================================================================================

isfile(joinpath(DATA_DIR, BENCH_FILE)) ||
    error("$(BENCH_FILE) not found. Run unified_benchmark.jl first.")

@load joinpath(DATA_DIR, BENCH_FILE) all_case_results base_results bench_info

const SQ = base_results[:statusquo]
const SQ_EMISSIONS = SQ.emissions.total
const TARGET_REDUCTION = bench_info.target_reduction          # B ton CO2e
const TARGET_REDUCTION_MT = TARGET_REDUCTION * 1000           # M ton CO2e

println("\nAnchor: $(bench_info.policy) / $(bench_info.case) " *
        "(recognize_cs=$(bench_info.recognize_cs), use_ci_threshold=$(bench_info.use_ci_threshold))")
println("  Total SAF        = $(round(Units.gal_to_L(bench_info.total_saf), digits=4)) " *
        "$(Units.METRIC ? "B liters" : "B gal")")
println("  Target abatement = $(round(TARGET_REDUCTION_MT, digits=3)) Mt CO2e")

# =================================================================================
# 2. Scenario definitions: table columns and figure row order, decided only here
# =================================================================================

const SCEN = Scenarios.SCEN

const DESIGN_EQUIVALENCES = [
    (policy=:rfs, a=:case1, b=:case3, shown="(c)",
        dropped="volumetric mandate crediting CS without the CI threshold"),
    (policy=:lcfs, a=:case3, b=:case4, shown="(g)",
        dropped="CI standard crediting CS with the CI threshold"),
    (policy=:taxcredit, a=:case3, b=:case4, shown="(i)",
        dropped="the tax credit crediting CS with the CI threshold"),
]

const SOY_NONSOY_PAIRS = [(:saf_hefa_conv, :saf_hefa_nonsoy),
    (:biodiesel_soy, :biodiesel_nonsoy),
    (:rd_soy, :rd_nonsoy)]

function max_solution_gap(x, y)
    cs_priced = max(x.l_cs, y.l_cs) > 1e-9
    skip = cs_priced ? Symbol[] : [:feedstock_corn_cs, :feedstock_soy_cs]
    paired = Set(g for p in SOY_NONSOY_PAIRS for g in p)

    g = 0.0
    # Floor the denominator so the relative gap does not blow up near zero.
    rel!(a, b, floor) = (g = max(g, abs(a - b) / max(abs(a), abs(b), floor)))

    # MCP returns a JuMP DenseAxisArray and the planner a Dict, so iterate keys, not axes.
    idx(c) = c isa AbstractDict ? keys(c) : axes(c, 1)

    for k in idx(x.q)
        k in paired && continue
        rel!(x.q[k], y.q[k], 1e-3)
    end
    for (a, b) in SOY_NONSOY_PAIRS
        rel!(x.q[a] + x.q[b], y.q[a] + y.q[b], 1e-3)
    end
    for k in idx(x.x)
        rel!(x.x[k], y.x[k], 1e-3)
    end
    for k in idx(x.p_f)
        k in skip && continue
        rel!(x.p_f[k], y.p_f[k], 1e-3)
    end
    rel!(x.l_n, y.l_n, 1e-6)
    rel!(x.l_cs, y.l_cs, 1e-6)
    rel!(x.emissions.total, y.emissions.total, 1e-6)
    return g
end

println("\n--- Design equivalence check (pairs that must return the same solution) ---")
for c in DESIGN_EQUIVALENCES
    gap = max_solution_gap(all_case_results[c.a].results[c.policy],
        all_case_results[c.b].results[c.policy])
    # Allow the gap left by the two bisections stopping at slightly different
    # stringencies (~1e-6 relative). Genuinely different solutions differ by ~1e-2,
    # so 1e-4 is a strict enough cut.
    ok = gap < 1e-4
    @printf("  %-10s %s == %s : max relative gap %.2e  %s\n", c.policy, c.a, c.b, gap,
        ok ? "equivalent, reported as $(c.shown)" : "!! no longer equivalent, revisit the table columns")
    ok || @warn "$(c.policy) $(c.a) vs $(c.b) are no longer equivalent (gap = $(gap))."
end

# Table headers carry the tag only: nine columns of full labels overflow the page. The
# tag-to-label key goes in the note.
const SCEN_POLICY = filter(s -> s.case !== :base, SCEN)   # status quo dropped


sol_of(s) = Scenarios.is_sq(s) ? SQ : all_case_results[s.case].results[s.policy]
welf_of(s) = all_case_results[s.case].welfare.welfare_summary[s.policy]
aac_of(s) = all_case_results[s.case].welfare.aac_results[s.policy]

function stringency_of(s)
    Scenarios.is_sq(s) && return 0.0
    cfg = all_case_results[s.case].policy_configs[s.policy]
    s.policy === :carbontax ? cfg.t :
    s.policy === :rfs ? cfg.θ_avi :
    s.policy === :lcfs ? cfg.σ : cfg.p
end

abatement_mt(s) = (SQ_EMISSIONS - sol_of(s).emissions.total) * 1000   # M ton CO2e

const SAF_TYPES = [
    (:saf_atj_conv, "Conv. ATJ"),
    (:saf_atj_cs, "CS ATJ"),
    (:saf_hefa_conv, "Conv. HEFA"),
    (:saf_hefa_cs, "CS HEFA"),
    (:saf_hefa_nonsoy, "Non-soy HEFA"),
]

saf_q(s, g) = max(0.0, sol_of(s).q[g])
saf_total(s) = sum(saf_q(s, g) for (g, _) in SAF_TYPES)

# Percent change. Values at the 1e-13 % level are solver noise, squashed to zero so
# numbers like -1.5e-14 never reach the CSV (the carbon tax column is the usual case:
# it taxes aviation fuel only, so crop, land and road fuel prices literally do not move).
pct(new, base) = abs(base) < 1e-10 ? NaN :
                 let r = (new - base) / base * 100
    abs(r) < 1e-9 ? 0.0 : r
end

# Scenario key, shared by the table notes and the figure captions
scen_legend() = join(["$(s.tag) $(Scenarios.one_line(s))" for s in SCEN], "; ")

println("\n--- Scenarios ---")
for s in SCEN
    @printf("  %-4s %-40s stringency = %10.5f   abatement = %7.3f Mt\n",
        s.tag, Scenarios.one_line(s), stringency_of(s), abatement_mt(s))
end

# =================================================================================
# 3. Shared LaTeX helpers
# =================================================================================
#
# Squash values like -1e-14 to zero first so they do not print as "-0.00".
fmt_cell(x, d, signed) =
    (ismissing(x) || (x isa Number && isnan(x))) ? "--" :
    begin
        v = abs(x) < 1e-10 ? 0.0 : x
        s = @sprintf("%.*f", d, v)
        (signed && v > 0) ? "+" * s : s
    end

fmt_commas(x) = begin
    s = string(round(Int, x))
    neg = startswith(s, "-")
    neg && (s = s[2:end])
    parts = String[]
    while !isempty(s)
        if lastindex(s) > 3
            pushfirst!(parts, s[(end-2):end])
            s = s[1:(end-3)]
        else
            pushfirst!(parts, s)
            s = ""
        end
    end
    (neg ? "-" : "") * join(parts, ",")
end

esc_tex(x) = replace(string(x), "\$" => "\\\$", "%" => "\\%", "_" => "\\_")

# latex_table(df; caption, label, note, digits, signed, hlines)
#
# Column 1 of `df` holds metric names, the rest are scenarios. `digits` and `signed` are
# functions of the metric name giving decimal places and whether to show a sign. Metrics
# listed in `hlines` get a rule above them.
function latex_table(df; caption, label, note, digits, signed, hlines=String[], bolds=String[])
    cols = names(df)[2:end]
    lines = String[
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{$(caption)}",
        "\\label{$(label)}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{l"*"c"^length(cols)*"}",
        "\\hline\\hline",
        " & "*join(cols, " & ")*" \\\\",
        "\\hline",
    ]
    for i in 1:nrow(df)
        m = string(df[i, 1])
        m in hlines && push!(lines, "\\hline")
        vals = [fmt_cell(df[i, c], digits(m), signed(m)) for c in cols]
        bold = m in bolds
        cells = bold ? ["\\textbf{$(v)}" for v in vals] : vals
        name = bold ? "\\textbf{$(esc_tex(m))}" : esc_tex(m)
        push!(lines, name * " & " * join(cells, " & ") * " \\\\")
    end
    append!(lines, [
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{adjustbox}",
        "\\begin{minipage}{0.95\\textwidth}",
        "\\footnotesize",
        "\\vspace{2mm}",
        "\\textit{Note:} " * note,
        "\\end{minipage}",
        "\\end{table}",
    ])
    return join(lines, "\n")
end

function save_tex(latex_str, output_dir, filename)
    path = joinpath(output_dir, filename)
    open(io -> write(io, latex_str), path, "w")
    println("✓ Saved: $path")
    return path
end

function save_csv(df, output_dir, filename)
    path = joinpath(output_dir, filename)
    CSV.write(path, df)
    println("✓ Saved: $path")
    return path
end

# Anchor sentence shared by all three tables
const ANCHOR_NOTE =
    "Every policy column is evaluated at the stringency that delivers the SAME GHG abatement, " *
    "namely $(round(TARGET_REDUCTION_MT, digits=2)) Mt CO\\textsubscript{2}e, the abatement of the " *
    "benchmark volumetric mandate (climate-smart practices not credited, no CI threshold) calibrated to " *
    "$(round(Units.gal_to_L(bench_info.total_saf), digits=2)) " *
    "$(Units.METRIC ? "billion liters" : "billion gallons") of SAF. Columns are " * scen_legend() * ". " *
    "``CI threshold'' refers to the 50\\% carbon-intensity reduction requirement for SAF eligibility."

# =================================================================================
# 4. Table A: aviation market outcomes and SAF mix
# =================================================================================

# Row labels are constants because `digits_A`, `signed_A` and the `hlines` argument all
# match on them; a literal typed twice would drift the moment the unit changed.
# Air travel is passenger-km in metric, road is vehicle-km. See main/units.jl for why
# both are spelled out rather than abbreviated to RPK / vkm.
const A_TRAVEL_Q = Units.METRIC ? "Passenger-km quantity change (%)" : "RPM quantity change (%)"
const A_TRAVEL_P = Units.METRIC ? "Passenger-km price change (%)" : "RPM price change (%)"
const A_VOL = Units.METRIC ? "B liters" : "B gallon"
const A_JET = "Jet fuel ($(A_VOL))"
const A_SAF = "Total SAF ($(A_VOL))"
const A_ABATE = "GHG abatement (Mt CO2e)"
# Emissions, the SCC, the carbon tax and the AAC are all ALREADY per metric ton (delta is
# tonnes of CO2e per gallon), so nothing in those rows converts -- only the spelling.
const C_TON = Units.METRIC ? "tonne" : "ton"

const A_ROWS = [
    "Policy stringency",
    A_ABATE,
    A_TRAVEL_Q,
    A_TRAVEL_P,
    A_JET,
    "Jet fuel change (%)",
    A_SAF,
    [lbl for (_, lbl) in SAF_TYPES]...,
]

function build_table_A()
    df = DataFrame(Metric=A_ROWS)
    for s in SCEN
        sol = sol_of(s)
        col = [
            stringency_of(s),
            abatement_mt(s),
            Scenarios.is_sq(s) ? NaN : pct(sol.x[:avi], SQ.x[:avi]),
            Scenarios.is_sq(s) ? NaN : pct(sol.p_c[:avi], SQ.p_c[:avi]),
            Units.gal_to_L(sol.q[:jet_fuel]),
            Scenarios.is_sq(s) ? NaN : pct(sol.q[:jet_fuel], SQ.q[:jet_fuel]),
            Units.gal_to_L(saf_total(s)),
            [Units.gal_to_L(saf_q(s, g)) for (g, _) in SAF_TYPES]...,
        ]
        df[!, Symbol(s.tag)] = col
    end
    return df
end

digits_A(m) = occursin("(%)", m) ? 2 :
              m == "Policy stringency" ? 4 :
              m == A_ABATE ? 2 : 3
signed_A(m) = occursin("(%)", m)

df_A = build_table_A()
println("\n--- Table A: aviation market outcomes ---")
show(df_A, allrows=true, allcols=true)
println()

tex_A = latex_table(df_A;
    caption="Aviation market outcomes at a common GHG abatement",
    label="tab:aviation_3B",
    note=ANCHOR_NOTE * " Policy stringency is in the unit of each instrument: \\\$/$(C_TON) CO\\textsubscript{2}e " *
         "for the carbon tax, the aviation mandate share for the volumetric mandate, the required CI reduction " *
         "for the CI standard, and \\\$/$(C_TON) CO\\textsubscript{2}e for the IRA tax credit. Quantity and price changes are " *
         "relative to the " *
         "status quo; SAF pathway rows are levels in $(Units.METRIC ? "billion liters" : "billion gallons").",
    digits=digits_A, signed=signed_A,
    hlines=[A_TRAVEL_Q, A_JET, A_SAF, "Conv. ATJ"])

save_csv(df_A, TABLE_DIR, "table_A_aviation_3B.csv")
save_tex(tex_A, TABLE_DIR, "table_A_aviation_3B.tex")

# =================================================================================
# 5. Table B: road transport and food spillovers
# =================================================================================

const DIESEL_SUBFUELS = [
    (:rd_soy, "Soy RD share (%)"),
    (:rd_nonsoy, "Non-soy RD share (%)"),
    (:biodiesel_soy, "Soy BD share (%)"),
    (:biodiesel_nonsoy, "Non-soy BD share (%)"),
]

# Every cell in Table B is a percentage change or an energy-equivalent share, so nothing
# here converts: only the road-travel unit in the row names changes (VMT -> vehicle-km).
const B_TRAVEL = Units.METRIC ? "vehicle-km" : "VMT"

function build_table_B()
    r, beta = params.coeff.r, params.coeff.beta
    share(part, total) = abs(total) < 1e-10 ? NaN : part / total * 100

    rows = ["Gasoline $(B_TRAVEL) quantity change (%)", "Gasoline share (%)", "Ethanol share (%)",
        "Diesel $(B_TRAVEL) quantity change (%)", "Diesel share (%)",
        [lbl for (_, lbl) in DIESEL_SUBFUELS]...,
        "Corn for food quantity change (%)", "Soybean oil for food quantity change (%)",
        "Soybean meal for food quantity change (%)"]

    df = DataFrame(Metric=rows)
    for s in SCEN
        sol = sol_of(s)
        gas_mi = r[:gasoline] * sol.q[:gasoline]
        eth_mi = r[:gasoline] * beta[(:ethanol, :gasoline)] * sol.q[:ethanol]
        die_mi = r[:diesel] * sol.q[:diesel]
        sub_mi = Dict(g => r[:diesel] * beta[(:rd, :diesel)] * sol.q[g] for (g, _) in DIESEL_SUBFUELS[1:2])
        merge!(sub_mi, Dict(g => r[:diesel] * beta[(:biodiesel, :diesel)] * sol.q[g] for (g, _) in DIESEL_SUBFUELS[3:4]))
        gas_tot = gas_mi + eth_mi
        die_tot = die_mi + sum(values(sub_mi))

        chg(new, base) = Scenarios.is_sq(s) ? NaN : pct(new, base)

        df[!, Symbol(s.tag)] = [
            chg(sol.x[:gas], SQ.x[:gas]), share(gas_mi, gas_tot), share(eth_mi, gas_tot),
            chg(sol.x[:die], SQ.x[:die]), share(die_mi, die_tot),
            [share(sub_mi[g], die_tot) for (g, _) in DIESEL_SUBFUELS]...,
            chg(sol.x[:corn], SQ.x[:corn]), chg(sol.x[:soyoil], SQ.x[:soyoil]),
            chg(sol.x[:soymeal], SQ.x[:soymeal]),
        ]
    end

    # Drop the RD/BD sub-rows that no policy ever uses
    sub_labels = [lbl for (_, lbl) in DIESEL_SUBFUELS]
    tags = [Symbol(s.tag) for s in SCEN_POLICY]
    keep = trues(nrow(df))
    for (i, m) in enumerate(df.Metric)
        m in sub_labels || continue
        vals = [df[i, t] for t in tags]
        all(v -> isnan(v) || abs(v) < 1e-6, vals) && (keep[i] = false)
    end
    return df[keep, :]
end

digits_B(m) = occursin("change", m) ? 2 : 3
signed_B(m) = occursin("change", m)

df_B = build_table_B()
println("\n--- Table B: road transportation and food outcomes ---")
show(df_B, allrows=true, allcols=true)
println()

tex_B = latex_table(df_B;
    caption="Road transportation and food outcomes at a common GHG abatement",
    label="tab:spillover_3B",
    note=ANCHOR_NOTE * " $(B_TRAVEL) and food quantity changes are relative to the status quo, so the status " *
         "quo column is blank for those rows. Fuel shares are computed on an energy-equivalent " *
         "($(Units.METRIC ? "kilometre" : "mile")) " *
         "basis within each mode. RD and BD subtype rows appear only if adopted under at least one policy.",
    digits=digits_B, signed=signed_B,
    hlines=["Diesel $(B_TRAVEL) quantity change (%)", "Corn for food quantity change (%)"])

save_csv(df_B, TABLE_DIR, "table_B_spillover_3B.csv")
save_tex(tex_B, TABLE_DIR, "table_B_spillover_3B.tex")

# 6. Table C: welfare decomposition and average abatement cost
# =================================================================================

#   (d) RFS crediting CS is effectively the same solution as (c), so the column would
#       duplicate it (same rule as Figure 2).

const SECTORS_C = [:avi, :gas, :die, :corn, :soyoil, :soymeal]
const SECTOR_LABELS_C = Dict(:avi => "Aviation", :gas => "Gasoline", :die => "Diesel",
    :corn => "Corn", :soyoil => "Soyoil", :soymeal => "Soymeal")

const SCEN_WELF = filter(s -> !Scenarios.is_sq(s) && s.tag != "(d)", SCEN)

# Row order and names. Totals (A to D) come first, with their components indented
# below. Writing (A+B+C) into the Private/Social rows makes the arithmetic readable
# from the table alone.
const C_TOTAL_CS = "(A) Total consumer surplus"
const C_TOTAL_PS = "(B) Producer surplus"
const C_GR = "(C) Govt revenue"
const C_ENV = "(D) Environmental benefit"
const C_PRIV = "Private surplus (A+B+C)"
const C_SOC = "Social welfare (A+B+C+D)"
const C_AAC_P = "AAC private (\$/$(C_TON) CO2e)"
const C_AAC_S = "AAC social (\$/$(C_TON) CO2e)"

# Rows printed in bold (totals) and rows that get a rule above them
const C_BOLD_ROWS = [C_TOTAL_CS, C_TOTAL_PS, C_GR, C_ENV]
const C_RULE_ABOVE = ["CS: Aviation", C_TOTAL_PS, "PS: Land", C_PRIV, C_AAC_P]

function build_table_C()
    rows = [C_TOTAL_CS]
    append!(rows, ["CS: $(SECTOR_LABELS_C[s])" for s in SECTORS_C])
    append!(rows, [C_TOTAL_PS, "PS: Land", "PS: Non-soy feedstock", "PS: Fossil refining",
        C_GR, C_ENV, C_PRIV, C_SOC, C_AAC_P, C_AAC_S])

    df = DataFrame(Metric=rows)
    for s in SCEN_WELF
        w, a = welf_of(s), aac_of(s)
        col = [w.cs_change]
        append!(col, [w.cs_by_sector[g] for g in SECTORS_C])
        append!(col, [w.ps_total_change, w.ps_land_change, w.ps_nonsoy_change, w.ps_fossil_change,
            w.gr_change, w.env_benefit, w.private_surplus, w.social_welfare,
            a.aac_private, a.aac_social])
        df[!, Symbol(s.tag)] = col
    end
    return df
end

digits_C(m) = occursin("AAC", m) ? 1 : 2
signed_C(m) = !occursin("AAC", m)

df_C = build_table_C()
println("\n--- Table C: welfare decomposition ---")
show(df_C, allrows=true, allcols=true)
println()

tex_C = latex_table(df_C;
    caption="Welfare decomposition at a common GHG abatement (billion \\\$)",
    label="tab:welfare_3B",
    note=ANCHOR_NOTE * " Column (a), the status quo, is omitted because every entry would be zero by " *
         "construction, and column (d) is omitted because it coincides with column (c). All entries " *
         "are changes relative to the status quo, in billion \\\$. Private surplus is the sum of " *
         "consumer surplus (CS), producer surplus (PS), and government revenue; social welfare is " *
         "private surplus plus the environmental benefit, valued at a social cost of carbon of " *
         "\\\$$(SCC)/$(C_TON) CO\\textsubscript{2}e. The average abatement cost (AAC) is defined as " *
         "\$-\\Delta\$welfare\$/\\Delta\$emissions; a negative AAC indicates that abatement raises " *
         "welfare even without valuing the environmental benefit, while a positive AAC indicates a " *
         "welfare cost per ton abated. Because every column is evaluated at the same GHG reduction " *
         "by construction, the environmental benefit is nearly identical across columns, and " *
         "differences in social welfare therefore stem almost entirely from how each policy " *
         "distributes surplus between consumers, producers, and the government.",
    digits=digits_C, signed=signed_C,
    hlines=C_RULE_ABOVE, bolds=C_BOLD_ROWS)

save_csv(df_C, TABLE_DIR, "table_C_welfare_3B.csv")

# Metadata read by main/table_C_to_docx.py, which moves Table C into Word. Retyping the
# abatement, SAF volume, units and column names on the Python side means that after a
# model change only the table body updates while the caption and footnotes keep the old
# numbers (the docx really was stuck at 40.45 Mt). Export once here; Python only reads.
save_csv(DataFrame(key=["target_reduction_mt", "bench_saf", "scc", "vol_unit", "ton_word"],
        value=[string(round(TARGET_REDUCTION_MT, digits=2)),
            string(round(Units.gal_to_L(bench_info.total_saf), digits=2)),
            string(SCC),
            Units.METRIC ? "billion liters" : "billion gallons",
            Units.METRIC ? "tonne" : "ton"]),
    TABLE_DIR, "table_C_welfare_3B_meta.csv")

# Column headers are exported here for the same reason (SCEN short is the source).
save_csv(DataFrame(tag=[s.tag for s in SCEN_WELF],
        short=[replace(s.short, "\n" => "|") for s in SCEN_WELF]),
    TABLE_DIR, "table_C_welfare_3B_headers.csv")
save_tex(tex_C, TABLE_DIR, "table_C_welfare_3B.tex")

# =================================================================================
# 7. Figure 1: aviation demand / jet fuel / SAF by scenario (three panels, left to right)
# =================================================================================
#
# left:   aviation passenger demand, RPM (B miles)
# middle: fossil jet fuel production (single bar)
# right:  SAF production (stacked by pathway)
# All three share the same y axis (scenarios), so how much of a given abatement comes
# from cutting demand versus switching fuel is readable in one row.

# Colors match the existing figures (the SAF mix figure in extract_results.jl section 19).
# Jet fuel is new there, so it takes the gray50 used for land rent in the spillover
# figure. RPM is demand rather than fuel, so it is a lighter gray.
const RPM_COLOR = :gray75
const JET_COLOR = :gray50
const SAF_PATHWAYS = [
    (:saf_atj_conv, "Conv ATJ-SAF", :blue),
    (:saf_atj_cs, "CS ATJ-SAF", :red),
    (:saf_hefa_conv, "Conv HEFA-SAF", :green),
    (:saf_hefa_cs, "CS HEFA-SAF", :orange),
    (:saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple),
]

# Ticks used to be hard-coded (0:400:1200 for RPM, 0:5:20 for jet fuel), which the
# conversion invalidates. Derive a round step from the panel maximum instead, so the same
# code gives sensible ticks in either unit system.
function nice_ticks(vmax; target=4)
    vmax <= 0 && return 0:1:1
    raw = vmax / target
    mag = 10.0^floor(log10(raw))
    step = raw / mag <= 1.5 ? mag : raw / mag <= 3.5 ? 2mag : raw / mag <= 7.5 ? 5mag : 10mag
    return 0:step:(ceil(vmax/step)*step)
end

function plot_jet_saf()
    n = length(SCEN)
    ypos = collect(n:-1:1)                       # (a) on top
    ylabels = ["$(s.tag) $(s.label)" for s in SCEN]
    bar_h = 0.62

    # Air travel is miles in the model, jet fuel and SAF are gallons; all three convert.
    rpm = [Units.mile_to_km(sol_of(s).x[:avi]) for s in SCEN]
    jet = [Units.gal_to_L(sol_of(s).q[:jet_fuel]) for s in SCEN]
    rpm_max = maximum(rpm)
    jet_max = maximum(jet)
    saf_max = maximum(Units.gal_to_L(saf_total(s)) for s in SCEN)

    fig = Figure(size=(950, 55 * n + 65), fontsize=12)

    ax0 = Axis(fig[1, 1];
        xlabel=Units.METRIC ? "Air travel (B passenger-km)" : "Air travel (B RPM)",
        yticks=(ypos, ylabels), yticklabelsize=13, xticklabelsize=13,
        ygridvisible=false, topspinevisible=false, rightspinevisible=false,
        xticks=nice_ticks(rpm_max),
        limits=((0, rpm_max * 1.20), (0.4, n + 0.6)))

    ax1 = Axis(fig[1, 2];
        xlabel=Units.METRIC ? "Jet fuel production (B liters)" : "Jet fuel production (B gal)",
        yticks=(ypos, fill("", n)),
        yticklabelsvisible=false, yticksvisible=false, xticklabelsize=13,
        ygridvisible=false, topspinevisible=false, rightspinevisible=false,
        xticks=nice_ticks(jet_max),
        limits=((0, jet_max * 1.24), (0.4, n + 0.6)))

    ax2 = Axis(fig[1, 3];
        xlabel=Units.METRIC ? "SAF production (B liters)" : "SAF production (B gal)",
        yticks=(ypos, fill("", n)),
        yticklabelsvisible=false, yticksvisible=false, xticklabelsize=13,
        ygridvisible=false, topspinevisible=false, rightspinevisible=false,
        xticks=nice_ticks(saf_max),
        limits=((0, saf_max * 1.22), (0.4, n + 0.6)))

    for (i, s) in enumerate(SCEN)
        y = ypos[i]

        # left: aviation passenger demand (RPM)
        poly!(ax0, Rect2f(0, y - bar_h / 2, rpm[i], bar_h);
            color=RPM_COLOR, strokecolor=:white, strokewidth=1.0)
        text!(ax0, rpm[i] + rpm_max * 0.015, y;
            text=fmt_commas(rpm[i]), align=(:left, :center), fontsize=13)

        # middle: fossil jet fuel
        poly!(ax1, Rect2f(0, y - bar_h / 2, jet[i], bar_h);
            color=JET_COLOR, strokecolor=:white, strokewidth=1.0)
        text!(ax1, jet[i] + jet_max * 0.015, y;
            text=@sprintf("%.2f", jet[i]), align=(:left, :center), fontsize=13)

        # right: SAF stacked by pathway
        left = 0.0
        for (g, _, col) in SAF_PATHWAYS
            v = Units.gal_to_L(saf_q(s, g))
            v > 1e-10 || continue
            poly!(ax2, Rect2f(left, y - bar_h / 2, v, bar_h);
                color=col, strokecolor=:white, strokewidth=1.0)
            left += v
        end
        text!(ax2, left + saf_max * 0.02, y;
            text=@sprintf("%.2f", left), align=(:left, :center), fontsize=13)
    end

    # Give the three panels the same plot width (y labels are handled by protrusion)
    for c in 1:3
        colsize!(fig.layout, c, Auto(1.0))
    end

    # Offset used to center the title and legend on the FIGURE, not on the cells.
    # Scenario labels sit in the axis left protrusion rather than in the column width, so
    # cells 1:3 are pushed right by that much. Measure it and widen the cell by the two
    # protrusions (negative padding). The layout has to be resolved once to read it.
    Makie.update_state_before_display!(fig)
    lp = ax0.layoutobservables.protrusions[].left
    rp = ax2.layoutobservables.protrusions[].right
    full = Outside(-lp, -rp, 0, 0)

    Label(fig[0, 1:3],
        "Air travel, jet fuel, and SAF at the same GHG abatement " *
        "($(round(TARGET_REDUCTION_MT, digits=1)) Mt CO₂e)";
        fontsize=14, font=:bold, padding=(0, 0, 2, 4), alignmode=full)

    # Two legend rows: the top holds the two items of the left and middle panels, the
    # bottom the five SAF pathways. nbanks=2 fills column-first and cannot split 2/5, so
    # stack two Legends and draw one box around both to separate them from the body.
    # Conv HEFA is zero in every scenario, but "in the model and not chosen" is itself a
    # result, so it stays in the legend.
    leg_kw = (orientation=:horizontal, framevisible=false, nbanks=1,
        labelsize=13, patchsize=(12, 12), colgap=8, tellwidth=true,
        padding=(12, 12, 3, 3))

    gl = GridLayout(fig[2, 1:3]; tellwidth=false, halign=:center, alignmode=full)

    Legend(gl[1, 1],
        [PolyElement(color=RPM_COLOR, strokecolor=:white),
            PolyElement(color=JET_COLOR, strokecolor=:white)],
        [Units.METRIC ? "Air travel (passenger-km)" : "Air travel (RPM)", "Fossil jet fuel"]; leg_kw...)

    Legend(gl[2, 1],
        [PolyElement(color=c, strokecolor=:white) for (_, _, c) in SAF_PATHWAYS],
        [lab for (_, lab, _) in SAF_PATHWAYS]; leg_kw...)

    rowgap!(gl, 1, 0)
    Box(gl[1:2, 1]; color=:transparent, strokecolor=:gray65, strokewidth=0.8)

    rowgap!(fig.layout, 2, 12)   # between axes and legend box (row 1 is the title)

    return fig
end

fig_fuel = plot_jet_saf()
save(joinpath(FIGURE_DIR, "fig_jet_saf_by_scenario.png"), fig_fuel; px_per_unit=3)
save(joinpath(FIGURE_DIR, "fig_jet_saf_by_scenario.pdf"), fig_fuel)
println("✓ Saved: ", joinpath(FIGURE_DIR, "fig_jet_saf_by_scenario.png"))

# =================================================================================
# 8. Figure 2: price spillovers (land rent, crops, VMT)
# =================================================================================
#
# Percent change from the status quo, so (a) status quo drops out of the rows.
# (d) RFS crediting CS also drops: without a threshold the CS pathway is not chosen, so
# it is effectively the same solution as (c) and would just draw two bars on top of each
# other.
#
# The corn and soybean panels place conventional and climate-smart side by side for each
# scenario. Where CS land is zero the CS price is not an interior value, so only the conv
# bar is drawn, centered on the scenario position.

const SCEN_SPILL = filter(s -> s.tag != "(d)", SCEN_POLICY)

has_cs_land(s) = sol_of(s).l_cs > 1e-9

# Climate-smart bars are hatched in the same color, as in the earlier Plots figures
hatch(col) = Makie.LinePattern(direction=[Vec2f(1, 1), Vec2f(1, -1)], width=1.2,
    tilesize=(11, 11), linecolor=col, backgroundcolor=:white)

function plot_price_spillovers()
    xs = collect(1:length(SCEN_SPILL))
    labels = ["$(s.tag) $(s.short)" for s in SCEN_SPILL]

    land = [pct(sol_of(s).duals.r_land, SQ.duals.r_land) for s in SCEN_SPILL]
    gas = [pct(sol_of(s).p_c[:gas], SQ.p_c[:gas]) for s in SCEN_SPILL]
    die = [pct(sol_of(s).p_c[:die], SQ.p_c[:die]) for s in SCEN_SPILL]

    corn_conv = [pct(sol_of(s).p_f[:feedstock_corn_n], SQ.p_f[:feedstock_corn_n]) for s in SCEN_SPILL]
    soy_conv = [pct(sol_of(s).p_f[:feedstock_soy_n], SQ.p_f[:feedstock_soy_n]) for s in SCEN_SPILL]
    corn_cs = [has_cs_land(s) ? pct(sol_of(s).p_f[:feedstock_corn_cs], SQ.p_f[:feedstock_corn_n]) : NaN
               for s in SCEN_SPILL]
    soy_cs = [has_cs_land(s) ? pct(sol_of(s).p_f[:feedstock_soy_cs], SQ.p_f[:feedstock_soy_n]) : NaN
              for s in SCEN_SPILL]

    fig = Figure(size=(1500, 700), fontsize=14)

    # Each panel scales to its own data. Soybean oil (IRA, no CS) is +350%, so a shared
    # axis would flatten every other panel beyond reading.
    function panel(pos, title, series...; ylabel="")
        vals = filter(!isnan, collect(Iterators.flatten(series)))
        hi, lo = maximum(vals), minimum(vals)
        ymax = hi > 0 ? hi * 1.18 : 1.0
        ymin = lo < 0 ? lo * 1.25 : -ymax * 0.06
        ax = Axis(fig[pos...]; title=title, titlesize=17,
            # 7 bars share a ~450px panel, so each tick has about 64px. The widest line
            # in `short` is 8 characters ("standard", "(c) Vol."), ~50px at size 12 --
            # comfortable. Raising this much above 13 brings the overlap back.
            xticks=(xs, labels), xticklabelsize=12,
            ylabel=ylabel, ylabelsize=14,
            xgridvisible=false, topspinevisible=false, rightspinevisible=false,
            limits=((0.4, length(xs) + 0.6), (ymin, ymax)))
        hlines!(ax, [0.0]; color=(:black, 0.35), linewidth=0.8)
        return ax, ymax
    end

    # One bar. A zero value draws nothing, so print "0" instead.
    PAIR_W = 0.30      # bar width in the paired panels (0.6 for a pair, 0.4 between scenarios)
    SOLO_W = 0.50      # single-bar panels

    # A change of 1e-9 PERCENT is one part in 1e11 -- solver noise, not a result. The old
    # cutoff sat right in the middle of that noise (land rent came out at 1.5e-9 and corn
    # at 2.0e-9, above it; soy oil at 8.2e-10 and the two road prices at ~1e-11, below),
    # so the same economic non-effect was drawn as a bar in two panels and as "0" in the
    # other three. Worse, a bar of height 1e-9 is zero pixels tall but still carries its
    # 0.8px WHITE stroke, which painted over the grey zero line and left it looking broken
    # under the carbon tax.
    #
    # 1e-6 sits five orders of magnitude above the noise and five below the smallest real
    # effect in this figure (diesel under the volumetric mandate, 0.25%), so it separates
    # the two cleanly.
    ZERO_TOL = 1e-6      # local to plot_price_spillovers, so no `const`

    function bar!(ax, xcenter, v, w, color, ymax; stroke=:white)
        if abs(v) < ZERO_TOL
            text!(ax, xcenter, ymax * 0.02; text="0", align=(:center, :bottom), fontsize=13)
        else
            # Second guard: any bar thinner than the stroke itself would still nick the
            # zero line. Below roughly one pixel of height, draw the fill without a stroke.
            sw = abs(v) > 0.005 * ymax ? 0.8 : 0.0
            poly!(ax, Rect2f(xcenter - w / 2, min(0.0, v), w, abs(v));
                color=color, strokecolor=stroke, strokewidth=sw)
            # When a panel holds a much larger value the axis stretches and small bars
            # become unreadable, so label those (to tell them apart from zero).
            if abs(v) < 0.025 * ymax
                text!(ax, xcenter, max(v, 0.0) + ymax * 0.015;
                    text=abs(v) < 1 ? @sprintf("%.2f", v) : @sprintf("%.1f", v),
                    align=(:center, :bottom), fontsize=11, color=:gray25)
            end
        end
    end

    draw_solo!(ax, vals, color, ymax) =
        for (x, v) in zip(xs, vals)
            bar!(ax, x, v, SOLO_W, color, ymax)
        end

    # Draw conv and cs as a pair. Conv always sits on the left (moving it makes the bars
    # ambiguous). Where CS land is zero, so the CS price is not interior, an x marks the
    # slot as not applicable.
    draw_pair!(ax, conv, cs, color, ymax) =
        for (i, x) in enumerate(xs)
            bar!(ax, x - PAIR_W / 2, conv[i], PAIR_W, color, ymax)
            if isnan(cs[i])
                text!(ax, x + PAIR_W / 2, ymax * 0.02;
                    text="×", align=(:center, :bottom), fontsize=17, color=:gray40)
            else
                bar!(ax, x + PAIR_W / 2, cs[i], PAIR_W, hatch(color), ymax; stroke=color)
            end
        end

    ax_land, m = panel((1, 1), "Land rent", land; ylabel="% change from the status quo")
    draw_solo!(ax_land, land, :gray50, m)

    ax_corn, m = panel((1, 2), "Corn price", corn_conv, corn_cs)
    draw_pair!(ax_corn, corn_conv, corn_cs, :goldenrod, m)

    ax_soy, m = panel((1, 3), "Soybean oil price", soy_conv, soy_cs)
    draw_pair!(ax_soy, soy_conv, soy_cs, :seagreen, m)

    ax_gas, m = panel((2, 1), "Gasoline $(B_TRAVEL) price", gas; ylabel="% change from the status quo")
    draw_solo!(ax_gas, gas, RGBf(0.87, 0.79, 0.60), m)

    ax_die, m = panel((2, 2), "Diesel $(B_TRAVEL) price", die)
    draw_solo!(ax_die, die, :saddlebrown, m)

    # The conv/CS distinction applies only to the corn and soybean oil panels, so the
    # legend goes horizontally at the top of the empty bottom-right cell, just below them.
    Legend(fig[2, 3],
        [PolyElement(color=:gray50),
            PolyElement(color=hatch(:gray50), strokecolor=:gray50, strokewidth=0.8)],
        ["Conventional feedstock", "Climate-smart feedstock"];
        orientation=:horizontal, nbanks=1,
        framevisible=true, framecolor=:gray65, framewidth=0.8,
        labelsize=13, patchsize=(14, 14), colgap=10, padding=(12, 12, 6, 6),
        tellwidth=false, tellheight=false, halign=:center, valign=:top)

    return fig
end

fig_prices = plot_price_spillovers()
save(joinpath(FIGURE_DIR, "fig_spillover_prices.png"), fig_prices; px_per_unit=3)
println("✓ Saved: ", joinpath(FIGURE_DIR, "fig_spillover_prices.png"))


# =================================================================================
# 9. Figure 3: land use by scenario (conventional / climate-smart, stacked)
# =================================================================================
#
# Moved here from fig_land_use.jl. It shared the anchor, the scenario axis and the color
# convention with the two figures above, so a separate file bought nothing. The real risk
# was the scenario definition living in two places, where fixing one silently desynced
# the other.
#
#   fig_land_use.png / .pdf          levels (M ha)
#   fig_land_use_change.png / .pdf   change from the status quo
#
# (d) RFS crediting CS is dropped from this figure only: without a threshold the CS
# pathway is not chosen, so it duplicates (c) and draws two bars on top of each other.
# It stays in the tables.
const SCEN_LAND = filter(s -> s.tag != "(d)", SCEN)

# The model solves in B acres (L0 = 0.11768 = 117.68 M acres). Multiply by 1000 for
# M acres, then hand off to units.jl. With Units.METRIC off these stay M acres.
land_conv(s) = Units.acre_to_ha(sol_of(s).l_n * 1000)
land_cs(s) = Units.acre_to_ha(sol_of(s).l_cs * 1000)

# Scenarios with no CS land have nothing to stack on top. Solver residuals at
# 1e-9 B acre (about one acre) count as zero.
const CS_TOL = 1e-6      # M ha

# =================================================================================
# Figures
# =================================================================================

# Dark green. Climate-smart is the same color hatched (as elsewhere in this file).
const LAND_COLOR = RGBf(0.11, 0.35, 0.20)

# Net-change diamond. It has to read clearly on top of the green bars, so dark red.
const NET_COLOR = RGBf(0.62, 0.08, 0.11)

hatch(col) = Makie.LinePattern(direction=[Vec2f(1, 1), Vec2f(1, -1)], width=1.2,
    tilesize=(11, 11), linecolor=col, backgroundcolor=:white)

function plot_land_use()
    xs = collect(1:length(SCEN_LAND))
    conv = [land_conv(s) for s in SCEN_LAND]
    cs = [land_cs(s) for s in SCEN_LAND]
    tot = conv .+ cs

    println("\n", rpad("tag", 6), rpad("conventional", 15), rpad("climate-smart", 16), "total")
    for (i, s) in enumerate(SCEN_LAND)
        println(rpad(s.tag, 6), rpad(@sprintf("%.3f", conv[i]), 15),
            rpad(@sprintf("%.3f", cs[i]), 16), @sprintf("%.3f", tot[i]))
    end

    ymax = maximum(tot) * 1.10

    fig = Figure(size=(1000, 560), fontsize=14)
    ax = Axis(fig[1, 1];
        ylabel=Units.METRIC ? "Million hectares" : "Million acres",
        ylabelsize=15,
        xticks=(xs, ["$(s.tag) $(s.short)" for s in SCEN_LAND]), xticklabelsize=12,
        xgridvisible=false, topspinevisible=false, rightspinevisible=false,
        limits=((0.4, length(xs) + 0.6), (0, ymax)))

    W = 0.58
    for (i, x) in enumerate(xs)
        # conventional: solid, from zero
        poly!(ax, Rect2f(x - W / 2, 0.0, W, conv[i]);
            color=LAND_COLOR, strokecolor=LAND_COLOR, strokewidth=0.8)
        # climate-smart: hatched, stacked on conventional
        if cs[i] > CS_TOL
            poly!(ax, Rect2f(x - W / 2, conv[i], W, cs[i]);
                color=hatch(LAND_COLOR), strokecolor=LAND_COLOR, strokewidth=0.8)
        end
        # total above the bar
        text!(ax, x, tot[i] + ymax * 0.012; text=@sprintf("%.2f", tot[i]),
            align=(:center, :bottom), fontsize=12.5)
    end

    Legend(fig[2, 1],
        [PolyElement(color=LAND_COLOR),
            PolyElement(color=hatch(LAND_COLOR), strokecolor=LAND_COLOR, strokewidth=0.8)],
        ["Conventional land", "Climate-smart land"];
        orientation=:horizontal, nbanks=1,
        framevisible=true, framecolor=:gray65, framewidth=0.8,
        labelsize=13, patchsize=(14, 14), colgap=10, padding=(12, 12, 6, 6),
        tellwidth=false, tellheight=true, halign=:center)

    rowgap!(fig.layout, 8)
    return fig
end

# =================================================================================
# Change figure: delta from the status quo
# =================================================================================
#
# In the level figure all eight bars share 48.7 M ha and look nearly identical. Removing
# that common part leaves the result: under designs that credit CS, CS land does not sit
# on top of conventional land, it displaces it. The compositional change is +-3 to 5.6
# M ha while the net change stays at or below +1.4 M ha.
#
# Delta M ha rather than percent because status quo CS land is zero, so the percent
# change of the CS component is undefined. Only conventional and total could be shown in
# percent, and CS, the point of the figure, could not.
#
# (a) status quo is zero throughout by construction, so it is dropped. The base level is
# printed under the axis.

const SCEN_LAND_CHG = filter(!Scenarios.is_sq, SCEN_LAND)

# 1e-4 M ha = 100 ha, four orders below the smallest meaningful change here
# (0.83 M ha) and above solver residuals, so it separates the two cleanly.
const ZERO_TOL = 1e-4      # M ha

function plot_land_use_change()
    xs = collect(1:length(SCEN_LAND_CHG))
    sq_conv, sq_tot = land_conv(SCEN_LAND[1]), land_conv(SCEN_LAND[1]) + land_cs(SCEN_LAND[1])

    d_conv = [land_conv(s) - sq_conv for s in SCEN_LAND_CHG]
    d_cs = [land_cs(s) for s in SCEN_LAND_CHG]   # status quo CS = 0
    d_tot = [land_conv(s) + land_cs(s) - sq_tot for s in SCEN_LAND_CHG]

    println("\n", rpad("tag", 6), rpad("Δ conventional", 17), rpad("Δ climate-smart", 18), "Δ net")
    for (i, s) in enumerate(SCEN_LAND_CHG)
        println(rpad(s.tag, 6), rpad(@sprintf("%+.4f", d_conv[i]), 17),
            rpad(@sprintf("%+.4f", d_cs[i]), 18), @sprintf("%+.4f", d_tot[i]))
    end

    hi = maximum(vcat(d_cs, d_tot, 0.0))
    lo = minimum(vcat(d_conv, 0.0))
    pad = (hi - lo) * 0.16
    ymax, ymin = hi + pad, lo - pad * 0.7

    fig = Figure(size=(1000, 555), fontsize=14)
    ax = Axis(fig[1, 1];
        ylabel="Change from the status quo (" *
               (Units.METRIC ? "million hectares)" : "million acres)"),
        ylabelsize=15,
        xticks=(xs, ["$(s.tag) $(s.short)" for s in SCEN_LAND_CHG]), xticklabelsize=12,
        xgridvisible=false, topspinevisible=false, rightspinevisible=false,
        limits=((0.4, length(xs) + 0.6), (ymin, ymax)))
    hlines!(ax, [0.0]; color=(:black, 0.45), linewidth=0.9)

    W = 0.46
    for (i, x) in enumerate(xs)
        # conventional: up or down from zero. Negative under designs that credit CS.
        # A zero-height bar is invisible, so print "0" to show it is zero, not missing.
        # Skip it when the net-change diamond already carries a value at the same spot.
        if abs(d_conv[i]) < ZERO_TOL
            if abs(d_tot[i]) >= ZERO_TOL
                text!(ax, x, ymax * 0.015; text="0", align=(:center, :bottom), fontsize=13)
            end
        else
            poly!(ax, Rect2f(x - W / 2, min(0.0, d_conv[i]), W, abs(d_conv[i]));
                color=LAND_COLOR, strokecolor=LAND_COLOR, strokewidth=0.8)
        end
        # climate-smart: always positive (the status quo has no CS land)
        if d_cs[i] > ZERO_TOL
            poly!(ax, Rect2f(x - W / 2, 0.0, W, d_cs[i]);
                color=hatch(LAND_COLOR), strokecolor=LAND_COLOR, strokewidth=0.8)
        end
    end

    # Net change (the sum of the two components). Drawn as a diamond so it is not
    # confused with a bar. The white stroke keeps (c)(f)(h) legible where the diamond
    # sits on a dark green bar.
    scatter!(ax, xs, d_tot; marker=:diamond, markersize=13,
        color=NET_COLOR, strokecolor=:white, strokewidth=1.2)
    # The number goes directly above the diamond in the same color. Set off to the right,
    # as before, it was unclear whether it labeled the diamond or the bar.
    # In (e)(g)(i) the diamond sits inside the hatching, so a white glow lifts the text.
    dy = (ymax - ymin) * 0.028
    for (i, x) in enumerate(xs)
        text!(ax, x, d_tot[i] + dy; text=@sprintf("%+.2f", d_tot[i]),
            align=(:center, :bottom), fontsize=11.5, color=NET_COLOR,
            font=:bold, glowcolor=(:white, 1.0), glowwidth=3.5)
    end

    Legend(fig[2, 1],
        [PolyElement(color=LAND_COLOR),
            PolyElement(color=hatch(LAND_COLOR), strokecolor=LAND_COLOR, strokewidth=0.8),
            MarkerElement(marker=:diamond, markersize=13, color=NET_COLOR,
                strokecolor=:white, strokewidth=1.2)],
        ["Conventional land", "Climate-smart land", "Net change"];
        orientation=:horizontal, nbanks=1,
        framevisible=true, framecolor=:gray65, framewidth=0.8,
        labelsize=13, patchsize=(14, 14), colgap=10, padding=(12, 12, 6, 6),
        tellwidth=false, tellheight=true, halign=:center)

    rowgap!(fig.layout, 8)
    return fig
end

fig = plot_land_use()
save(joinpath(FIGURE_DIR, "fig_land_use.png"), fig; px_per_unit=3)
save(joinpath(FIGURE_DIR, "fig_land_use.pdf"), fig)
println("\n✓ Saved: ", joinpath(FIGURE_DIR, "fig_land_use.png"))

fig_chg = plot_land_use_change()
save(joinpath(FIGURE_DIR, "fig_land_use_change.png"), fig_chg; px_per_unit=3)
save(joinpath(FIGURE_DIR, "fig_land_use_change.pdf"), fig_chg)
println("✓ Saved: ", joinpath(FIGURE_DIR, "fig_land_use_change.png"))

println("\n" * "="^80)
println("Done. Wrote 3 tables (csv + tex) and 4 figures.")
println("  Anchor: common abatement $(round(TARGET_REDUCTION_MT, digits=3)) Mt CO2e")
println("="^80)
