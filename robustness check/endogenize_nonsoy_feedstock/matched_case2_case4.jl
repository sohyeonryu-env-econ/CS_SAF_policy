# matched_case2_case4.jl  (endogenized non-soy supply)
#
# Purpose: with non-soy endogenized as an upward-sloping CES supply rather than a hard
#          capacity, compare case2 (no CS crediting + 50% CI threshold) against case4
#          (CS crediting + 50% CI threshold) at EQUAL GHG abatement.
#
# The main model's tables fix total SAF at 3B gal. But the RFS mandate counts eligible SAF
# only, so case2, with fewer eligible pathways, needs a far tighter policy to reach 3B, and
# the two cases end up abating very different amounts. Here the abatement is fixed instead.
#
# Procedure:
#   Step 0. Solve the status quo.
#   Step 1. Tune the case2 RFS to total SAF = 3B and take its abatement as the target.
#   Step 2. Tune the case2 LCFS/IRA and the case4 RFS/LCFS/IRA to that same abatement.
#   Step 3. Decompose welfare and write the tables and LaTeX.
#
# To run:  julia matched_case2_case4.jl     or  include("matched_case2_case4.jl")

include("model_endo_nonsoy.jl")
include("analysis_endo_nonsoy.jl")

using .ModelEndoNonsoy
using .AnalysisEndoNonsoy
using JuMP, DataFrames, Printf, CSV

import .ModelEndoNonsoy: params, build_unified_model, extract_solution
import .AnalysisEndoNonsoy: calculate_emissions_detail, calculate_implicit_taxes,
    calculate_cs_changes, calculate_ps_land_changes, calculate_ps_nonsoy_changes,
    calculate_gr_changes, calculate_environmental_benefit,
    calculate_total_welfare, SQ_CONFIG

include(joinpath(@__DIR__, "..", "..", "main", "paths.jl"))
using .Paths
const OUT = Paths.variant("endo_nonsoy")
const OUT_DIR = OUT.tables

const SCC = 190.0
const SAF_GOODS_M = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
const POLS_M = [:rfs, :lcfs, :taxcredit]
const POL_LABEL_M = Dict(:rfs => "RFS", :lcfs => "LCFS", :taxcredit => "IRA")
const SECTORS_M = [:avi, :gas, :die, :corn, :soyoil, :soymeal]
const SECTOR_LABEL_M = Dict(:avi => "Aviation", :gas => "Gasoline", :die => "Diesel",
    :corn => "Corn", :soyoil => "Soyoil", :soymeal => "Soymeal")
const CASE_DEF_M = Dict(
    :case2 => (recognize_cs=false, use_ci_threshold=true, label="No CS, 50\\% threshold"),
    :case4 => (recognize_cs=true, use_ci_threshold=true, label="CS, 50\\% threshold"),
)
# case2 has fewer eligible pathways, so the same abatement needs a very tight policy. Keep the range wide.
const RANGE_M = Dict(:rfs => (0.0, 3.0), :lcfs => (0.0, 1.0), :taxcredit => (0.0, 2000.0))

mkcfg(policy, v, cd) = (
    t=0.0,
    θ_avi=policy === :rfs ? v : 0.0,
    σ=policy === :lcfs ? v : 0.0,
    p=policy === :taxcredit ? v : 0.0,
    carbon_tax_scope=:aviation,
    use_ci_threshold=cd.use_ci_threshold,
    recognize_cs=cd.recognize_cs,
)

# solve_at(policy, v, cd): solve the model at a given stringency and return it with the
# fields the welfare calculation needs. Returns `nothing` if it does not solve.
function solve_at(policy::Symbol, v, cd)
    cfg = mkcfg(policy, v, cd)
    m = build_unified_model(params, cfg)
    optimize!(m)
    is_solved_and_feasible(m) || return nothing
    sol = extract_solution(m, policy)
    return merge(sol, (emissions=calculate_emissions_detail(sol, params),
        implicit_taxes=calculate_implicit_taxes(sol, params, cfg)))
end

total_saf(sol) = sum(sol.q[g] for g in SAF_GOODS_M)

# bisect(cd, policy, f, target; tol, maxiter): bisect for the stringency at which
# `f(sol)` equals `target`. `f` must be increasing in stringency (both total SAF and
# abatement are). If the target is not reached inside the range it reports
# `converged=false`, so a wrong value is never used silently.
function bisect(cd, policy::Symbol, f, target; tol=1e-4, maxiter=200)
    low, high = RANGE_M[policy]
    best = nothing
    for _ in 1:maxiter
        (high - low) < 1e-7 && break
        mid = (low + high) / 2
        sol = solve_at(policy, mid, cd)
        if isnothing(sol)
            high = mid
            continue
        end
        val = f(sol)
        best = (value=mid, sol=sol, val=val)
        abs(val - target) < tol && break
        val < target ? (low = mid) : (high = mid)
    end
    isnothing(best) && error("$policy: nothing solved inside the range.")
    return merge(best, (converged=abs(best.val - target) < 1e-3,))
end

# =============================================================================
# Step 0-2: solve
# =============================================================================

println("="^100)
println("ENDOGENIZED NON-SOY SUPPLY: case2 vs case4 at EQUAL GHG ABATEMENT")
println("="^100)

msq = build_unified_model(params, SQ_CONFIG)
optimize!(msq)
is_solved_and_feasible(msq) || error("the status quo did not solve.")
sq = extract_solution(msq, :statusquo)
sq = merge(sq, (emissions=calculate_emissions_detail(sq, params),
    implicit_taxes=calculate_implicit_taxes(sq, params, SQ_CONFIG)))
sq_em = sq.emissions.total
@printf("\nStatus quo emissions = %.4f B ton CO2e   non-soy = %.2f B lb @ %.3f \$/lb\n",
    sq_em, sq.q_feedstock.nonsoy, sq.p_f[:feedstock_nonsoy])

# Step 1: anchor the case2 RFS at total SAF = 3B and set the target abatement
println("\n-- Step 1: anchor the case2 RFS at total SAF = 3.0 B gal --")
anchor = bisect(CASE_DEF_M[:case2], :rfs, total_saf, 3.0)
target_red = sq_em - anchor.sol.emissions.total
@printf("  theta = %.5f   total SAF = %.4f B gal   target abatement = %.2f Mt CO2e%s\n",
    anchor.value, anchor.val, target_red * 1000,
    anchor.converged ? "" : "   ** short of 3B, out of range **")

# Step 2: align every policy in both cases on the target abatement
println("\n-- Step 2: align all policies in both cases on the same abatement --")
red_of(sol) = sq_em - sol.emissions.total
S = Dict{Tuple{Symbol,Symbol},Any}()
for ck in (:case2, :case4), pt in POLS_M
    r = (ck === :case2 && pt === :rfs) ?
        merge(anchor, (val=target_red, converged=true)) :
        bisect(CASE_DEF_M[ck], pt, red_of, target_red)
    S[(ck, pt)] = r
    @printf("  %-6s %-10s stringency = %10.5f   abatement %8.2f Mt%s\n",
        ck, pt, r.value, red_of(r.sol) * 1000,
        r.converged ? "" : "   ** short of target, may be out of range **")
end

# =============================================================================
# Step 3: welfare decomposition
# =============================================================================

function welfare_of(ck)
    D = Dict{Symbol,Any}(:statusquo => sq)
    for pt in POLS_M
        D[pt] = S[(ck, pt)].sol
    end
    cs = calculate_cs_changes(D, sq, params; scenarios=POLS_M)
    psl = calculate_ps_land_changes(D, sq, params; scenarios=POLS_M)
    psn = calculate_ps_nonsoy_changes(D, sq, params; scenarios=POLS_M)
    gr = calculate_gr_changes(D; scenarios=POLS_M)
    env = calculate_environmental_benefit(D, sq, SCC; scenarios=POLS_M)
    return calculate_total_welfare(cs, psl, gr, env; ps_nonsoy_changes=psn, scenarios=POLS_M)
end

W = Dict(ck => welfare_of(ck) for ck in (:case2, :case4))

rows = ["Policy stringency", "Total SAF (B gallon)", "GHG abatement (Mt CO2e)",
    "Non-soy use (B lb)", "Non-soy price (\$/lb)", "Total consumer surplus"]
append!(rows, ["CS: $(SECTOR_LABEL_M[s])" for s in SECTORS_M])
append!(rows, ["PS: Land", "PS: Non-soy feedstock", "Producer surplus",
    "Govt revenue", "Environmental benefit", "Private surplus", "Social welfare",
    "AAC private (\$/ton CO2e)", "AAC social (\$/ton CO2e)"])

function pack(ck, pt)
    r = S[(ck, pt)]
    sol = r.sol
    w = W[ck][pt]
    red = red_of(sol)
    v = Float64[r.value, total_saf(sol), red * 1000,
        sol.q_feedstock.nonsoy, sol.p_f[:feedstock_nonsoy], w.cs_change]
    append!(v, [w.cs_by_sector[s] for s in SECTORS_M])
    append!(v, [w.ps_land_change, w.ps_nonsoy_change, w.ps_total_change,
        w.gr_change, w.env_benefit, w.private_surplus, w.social_welfare,
        -w.private_surplus / red, -w.social_welfare / red])
    return v
end

df = DataFrame(Metric=rows)
for pt in POLS_M
    a, b = pack(:case2, pt), pack(:case4, pt)
    df[!, Symbol("$(POL_LABEL_M[pt]) (case2)")] = a
    df[!, Symbol("$(POL_LABEL_M[pt]) (case4)")] = b
    df[!, Symbol("$(POL_LABEL_M[pt]) diff")] = b .- a
end

println()
show(df, allrows=true, allcols=true)
println()

CSV.write(joinpath(OUT_DIR, "table_matched_endog_case2_vs_case4.csv"), df)
println("✓ Saved: ", joinpath(OUT_DIR, "table_matched_endog_case2_vs_case4.csv"))

# =============================================================================
# Step 4: LaTeX
# =============================================================================

fmt_M(x) = isnan(x) ? "--" :
           abs(x) < 5e-3 ? "0.00" :
           (x > 0 ? @sprintf("+%.2f", x) : @sprintf("%.2f", x))

function to_latex_M(df)
    esc(s) = replace(s, "\$" => "\\\$")
    n = 3 * length(POLS_M)
    L = String[
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Robustness with endogenous non-soy feedstock supply: welfare " *
        "decomposition at equal GHG abatement, No CS vs.\\ CS (both with the 50\\% CI " *
        "threshold; Billion \\\$)}",
        "\\label{tab:welfare_matched_endog_case2_case4}",
        "\\begin{tabular}{l" * "c"^n * "}",
        "\\hline\\hline",
        " & " * join(["\\multicolumn{3}{c}{$(POL_LABEL_M[p])}" for p in POLS_M], " & ") * " \\\\",
        join(["\\cmidrule(lr){$(2+3*(i-1))-$(4+3*(i-1))}" for i in 1:length(POLS_M)], " "),
        " & " * join(repeat(["(a)", "(b)", "\$\\Delta\$"], length(POLS_M)), " & ") * " \\\\",
        "\\hline",
    ]
    for (i, m) in enumerate(df.Metric)
        m in ("Total consumer surplus", "CS: Aviation", "PS: Land", "Private surplus",
            "AAC private (\$/ton CO2e)") && push!(L, "\\hline")
        push!(L, esc(m) * " & " * join([fmt_M(df[i, c]) for c in names(df)[2:end]], " & ") * " \\\\")
    end
    append!(L, [
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{minipage}{0.95\\textwidth}",
        "\\footnotesize",
        "\\vspace{2mm}",
        "\\textit{Note:} This table repeats the main welfare comparison under the robustness " *
        "specification in which non-soy feedstock is supplied along an upward-sloping CES " *
        "curve rather than at a fixed price up to a hard capacity. Column (a) is " *
        "$(CASE_DEF_M[:case2].label); column (b) is $(CASE_DEF_M[:case4].label); " *
        "\$\\Delta\$ is (b) minus (a). All welfare values are changes relative to the status " *
        "quo, in billion \\\$. The two cases are compared at the SAME GHG abatement: the " *
        "target is the abatement delivered by the RFS stringency that induces a total SAF " *
        "volume of 3 billion gallons under (a), and every other policy in both columns is " *
        "re-solved to deliver that same abatement. Differences in welfare therefore isolate " *
        "the cost of achieving a given abatement rather than confounding it with a difference " *
        "in how much is abated. Stringency units are the mandate ratio for the RFS, the " *
        "required CI reduction share for the LCFS, and \\\$/gallon for IRA. Social welfare " *
        "is private surplus plus environmental benefit at a social cost of carbon of " *
        "\\\$$(Int(SCC))/ton CO\\textsubscript{2}e.",
        "\\end{minipage}",
        "\\end{table}",
    ])
    return join(L, "\n")
end

tex = to_latex_M(df)
open(joinpath(OUT_DIR, "table_matched_endog_case2_vs_case4.tex"), "w") do io
    write(io, tex)
end
println("✓ Saved: ", joinpath(OUT_DIR, "table_matched_endog_case2_vs_case4.tex"))
println()
println(tex)

# =============================================================================
# Step 5: summary against the hard-capacity results
# =============================================================================

println("\n" * "="^100)
println("Summary: does endogenizing change the conclusion?")
println("="^100)
@printf("%-8s %14s %14s %14s %14s\n", "policy", "dSocial", "dAAC_priv", "PS_ns case2", "PS_ns case4")
for pt in POLS_M
    wa, wb = W[:case2][pt], W[:case4][pt]
    ra, rb = red_of(S[(:case2, pt)].sol), red_of(S[(:case4, pt)].sol)
    @printf("%-8s %+14.2f %+14.1f %14.2f %14.2f\n", POL_LABEL_M[pt],
        wb.social_welfare - wa.social_welfare,
        (-wb.private_surplus / rb) - (-wa.private_surplus / ra),
        wa.ps_nonsoy_change, wb.ps_nonsoy_change)
end
println("\ndSocial > 0 means crediting CS raises welfare under endogenized supply.")
println("Hard-capacity reference: RFS -5.10, LCFS -4.01, IRA -0.02")
println("PS_ns hard-capacity reference: case2 46.45/46.45/162.42, case4 9.74/14.77/28.73")
