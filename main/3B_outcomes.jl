# table_4_3_market_outcomes.jl
#
# Section 4.3 "SAF adoption and aviation market outcomes"에 들어갈
# Table B: 3B-equivalent anchor에서의 정책별 시장 결과 비교표를 생성한다.
#
# 전제: run_model.jl 을 먼저 실행해서
#   /output/results/results_target.jld2 (3B 앵커, use_ci_threshold 적용된 실제 정책 설계)
# 가 이미 저장되어 있어야 한다. 그 안의 equivalent_policies와 results_target을 그대로 쓴다.
#
# 사용법:
include("SAFModel.jl");
include("analysis.jl")
const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"
const FIGURE_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/figures"
import .SAFAnalysis: calculate_emissions_detail, calculate_implicit_taxes,
    calculate_cs_changes, calculate_ps_land_changes,
    calculate_gr_changes, calculate_environmental_benefit,
    calculate_total_welfare
import .SAFModel: params

using JLD2, DataFrames, Printf

# =================================================================================
# 1. 표에 쓸 정책 순서와 라벨 (스크린샷 표의 열 순서: Carbon tax, RFS, LCFS, IRA)
# =================================================================================

const TABLE_POLICY_ORDER = [:carbontax, :rfs, :lcfs, :taxcredit]
const TABLE_POLICY_LABELS = Dict(
    :carbontax => "Carbon tax",
    :rfs => "RFS",
    :lcfs => "LCFS",
    :taxcredit => "IRA",
)

# SAF 유형 5종 (스크린샷 표 하단 행 순서와 동일)
const SAF_TYPE_ORDER = [
    (:saf_atj_conv, "Conv. ATJ"),
    (:saf_atj_cs, "CS ATJ"),
    (:saf_hefa_conv, "Conv. HEFA"),
    (:saf_hefa_cs, "CS HEFA"),
    (:saf_hefa_nonsoy, "Non soy HEFA"),
]

# =================================================================================
# 2. 앵커 데이터 로드
# =================================================================================

"""
    load_3B_anchor(output_dir)
 
results_target.jld2 에서 status quo와 3B-equivalent 정책 해를 불러온다.
run_model.jl 의 1번 블록(base scenarios)에서 status quo를,
2번 블록(equivalent policies, target_saf=3.0)에서 각 정책의 3B 앵커 해를 가져온다.
"""
function load_3B_anchor(output_dir)
    # status quo는 results_base.jld2 에 들어 있음
    @load joinpath(output_dir, "results_base.jld2") results_base
    sq = results_base[:statusquo]

    # 3B 앵커 정책 해 (target_saf == 3.0 → suffix "" 사용)
    @load joinpath(output_dir, "results_target.jld2") results_target policy_configs_target equivalent_policies

    return sq, results_target, equivalent_policies
end

# =================================================================================
# 3. 표 본체 계산
# =================================================================================

"""
    pct_change(new, base)
 
status quo 대비 % 변화. base가 0에 가까우면 NaN을 반환해 계산 오류를 막는다.
"""
pct_change(new, base) = abs(base) < 1e-10 ? NaN : (new - base) / base * 100

"""
    build_table_4_3(output_dir; digits_pct=1, digits_qty=3)
 
스크린샷의 표 구조(정책 4열 × 지표 9행)를 DataFrame으로 만든다.
행 순서:
  1. RPM quantity change (%)
  2. RPM price change (%)
  3. Jet fuel quantity change (%)
  4. Total SAF quantity (B gallon)
  5-9. SAF 유형별 물량 (Conv ATJ, CS ATJ, Conv HEFA, CS HEFA, Non soy HEFA)
"""
function build_table_4_3(output_dir; digits_pct=1, digits_qty=3)
    sq, results_target, equivalent_policies = load_3B_anchor(output_dir)

    # status quo 기준값
    sq_rpm_qty = sq.x[:avi]
    sq_rpm_price = sq.p_c[:avi]
    sq_jet_qty = sq.q[:jet_fuel]

    # 결과를 담을 딕셔너리: row_label => Dict(policy => value)
    rows = OrderedRowsInit()

    for pt in TABLE_POLICY_ORDER
        sol = results_target[pt]

        rpm_qty_chg = pct_change(sol.x[:avi], sq_rpm_qty)
        rpm_price_chg = pct_change(sol.p_c[:avi], sq_rpm_price)
        jet_qty_chg = pct_change(sol.q[:jet_fuel], sq_jet_qty)

        saf_vals = Dict(g => sol.q[g] for (g, _) in SAF_TYPE_ORDER)
        total_saf = sum(values(saf_vals))

        push_row!(rows, "RPM quantity change (%)", pt, rpm_qty_chg)
        push_row!(rows, "RPM price change (%)", pt, rpm_price_chg)
        push_row!(rows, "Jet fuel quantity change (%)", pt, jet_qty_chg)
        push_row!(rows, "Total SAF quantity (B gallon)", pt, total_saf)
        for (g, label) in SAF_TYPE_ORDER
            push_row!(rows, label, pt, saf_vals[g])
        end
    end

    # DataFrame으로 변환 (행 = 지표, 열 = 정책)
    row_order = [
        "RPM quantity change (%)",
        "RPM price change (%)",
        "Jet fuel quantity change (%)",
        "Total SAF quantity (B gallon)",
        [label for (_, label) in SAF_TYPE_ORDER]...,
    ]

    df = DataFrame(Metric=row_order)
    for pt in TABLE_POLICY_ORDER
        col = [rows[(m, pt)] for m in row_order]
        df[!, Symbol(TABLE_POLICY_LABELS[pt])] = col
    end

    return df, to_latex(df, row_order, digits_pct, digits_qty)
end

# 간단한 순서 보존용 컨테이너 (Dict 대신 사용, 삽입 순서 문제 방지)
OrderedRowsInit() = Dict{Tuple{String,Symbol},Float64}()
function push_row!(rows, metric::String, policy::Symbol, val)
    rows[(metric, policy)] = val
end

# =================================================================================
# 4. LaTeX 표 문자열 생성 (논문에 바로 붙여넣기용)
# =================================================================================

function fmt_val(row_name, x, digits_pct, digits_qty)
    isnan(x) && return "--"
    is_pct = occursin("(%)", row_name)
    d = is_pct ? digits_pct : digits_qty
    s = @sprintf("%.*f", d, x)
    # 퍼센트 행은 부호 명시 (증가/감소 방향을 표에서 바로 읽도록)
    if is_pct && x > 0
        s = "+" * s
    end
    return s
end

function to_latex(df, row_order, digits_pct, digits_qty)
    n_policy_cols = DataFrames.ncol(df) - 1  # Metric 열 제외
    header = join(names(df)[2:end], " & ")
    lines = String[]
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "\\centering")
    push!(lines, "\\caption{Aviation market outcomes at the 3B-gallon-equivalent anchor (relative to status quo)}")
    push!(lines, "\\label{tab:market_outcomes_3B}")
    push!(lines, "\\begin{tabular}{l" * "c"^n_policy_cols * "}")
    push!(lines, "\\hline\\hline")
    push!(lines, " & " * header * " \\\\")
    push!(lines, "\\hline")

    for (i, m) in enumerate(row_order)
        # SAF 유형 세부 행 앞에 소구분선
        if m == "Conv. ATJ"
            push!(lines, "\\hline")
            push!(lines, "\\multicolumn{$(n_policy_cols+1)}{l}{\\textit{SAF composition (B gallon)}} \\\\")
        end
        vals = [fmt_val(m, df[i, Symbol(TABLE_POLICY_LABELS[pt])], digits_pct, digits_qty)
                for pt in TABLE_POLICY_ORDER]
        push!(lines, m * " & " * join(vals, " & ") * " \\\\")
    end

    push!(lines, "\\hline\\hline")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\begin{minipage}{0.95\\textwidth}")
    push!(lines, "\\footnotesize")
    push!(lines, "\\vspace{2mm}")
    push!(lines, "\\textit{Note:} All values are evaluated at each policy's stringency level " *
                 "that induces a total SAF volume of 3 billion gallons (using the CI-threshold " *
                 "policy design for RFS and IRA, as in the main analysis). Percentage changes are " *
                 "relative to the status quo (no SAF policy).")
    push!(lines, "\\end{minipage}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# =================================================================================
# 5. 실행 예시 (파일 하단, 직접 실행 시)
# =================================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"
    df, latex_str = build_table_4_3(OUTPUT_DIR)
    println(df)
    println()
    println(latex_str)
end

include("table_4_3_market_outcomes.jl")
df, latex_str = build_table_4_3(OUTPUT_DIR)
println(latex_str)   # 논문에 붙여넣을 LaTeX
using CSV
CSV.write(joinpath(OUTPUT_DIR, "table_SAF_3B_outcomes.csv"), df)

# ---------------------------------------------------------------------------
# Section 4.4 Table: Road transportation & food composition at the 3B anchor
# (4.3 코드에서 이미 로드한 sq, results_target 재사용)
# ---------------------------------------------------------------------------

using CSV

const SCENARIOS_44 = [(:statusquo, "Status quo"), (:carbontax, "Carbon tax"),
    (:rfs, "RFS"), (:lcfs, "LCFS"), (:taxcredit, "Tax credit")]

function build_table_4_4(sq, results_target; digits=1)
    r, beta = params.coeff.r, params.coeff.beta
    sols = merge(Dict(:statusquo => sq), results_target)

    share(part, total) = abs(total) < 1e-10 ? NaN : part / total * 100
    chg(new, base, sc) = sc == :statusquo || abs(base) < 1e-10 ? NaN : (new - base) / base * 100

    # RD/BD를 soy/non-soy로 분리 (mile 등가 기준 share)
    diesel_subfuels = [
        (:rd_soy, "Soy RD share (%)"),
        (:rd_nonsoy, "Non-soy RD share (%)"),
        (:biodiesel_soy, "Soy BD share (%)"),
        (:biodiesel_nonsoy, "Non-soy BD share (%)"),
    ]

    rows = ["Gasoline VMT quantity change (%)", "Gasoline share (%)", "Ethanol share (%)",
        "Diesel VMT quantity change (%)", "Diesel share (%)",
        [lbl for (_, lbl) in diesel_subfuels]...,
        "Corn for food quantity change (%)", "Soybean oil for food quantity change (%)",
        "Soybean meal for food quantity change (%)"]

    df = DataFrame(Metric=rows)
    for (sc, label) in SCENARIOS_44
        sol = sols[sc]
        gas_mi = r[:gasoline] * sol.q[:gasoline]
        eth_mi = r[:gasoline] * beta[(:ethanol, :gasoline)] * sol.q[:ethanol]
        die_mi = r[:diesel] * sol.q[:diesel]
        sub_mi = Dict(g => r[:diesel] * beta[(:rd, :diesel)] * sol.q[g] for (g, _) in diesel_subfuels[1:2])
        merge!(sub_mi, Dict(g => r[:diesel] * beta[(:biodiesel, :diesel)] * sol.q[g] for (g, _) in diesel_subfuels[3:4]))
        gas_tot = gas_mi + eth_mi
        die_tot = die_mi + sum(values(sub_mi))

        col = [
            chg(sol.x[:gas], sq.x[:gas], sc), share(gas_mi, gas_tot), share(eth_mi, gas_tot),
            chg(sol.x[:die], sq.x[:die], sc), share(die_mi, die_tot),
            [share(sub_mi[g], die_tot) for (g, _) in diesel_subfuels]...,
            chg(sol.x[:corn], sq.x[:corn], sc), chg(sol.x[:soyoil], sq.x[:soyoil], sc),
            chg(sol.x[:soymeal], sq.x[:soymeal], sc),
        ]
        df[!, Symbol(label)] = col
    end

    # 모든 정책 열(status quo 제외)에서 값이 0에 가깝거나 NaN인 RD/BD 세부 행은 제거
    diesel_sub_labels = [lbl for (_, lbl) in diesel_subfuels]
    policy_labels = [label for (sc, label) in SCENARIOS_44 if sc != :statusquo]
    keep = trues(nrow(df))
    for (i, m) in enumerate(df.Metric)
        m in diesel_sub_labels || continue
        vals = [df[i, Symbol(l)] for l in policy_labels]
        all(v -> isnan(v) || abs(v) < 1e-6, vals) && (keep[i] = false)
    end
    return df[keep, :]
end

# CSV 저장 -------------------------------------------------------------------

function save_table_4_4_csv(df, output_dir; filename="table_spillovers_3B_outcomes.csv")
    path = joinpath(output_dir, filename)
    CSV.write(path, df)
    println("✓ Saved: $path")
    return path
end

# LaTeX 변환 -------------------------------------------------------------------
# quantity change 행: 소수점 2자리, share 행: 소수점 3자리

function fmt_44(row_name, x)
    isnan(x) && return "--"
    digits = occursin("change", row_name) ? 2 : 3
    s = @sprintf("%.*f", digits, x)
    occursin("change", row_name) && x > 0 && (s = "+" * s)
    return s
end

function to_latex_4_4(df)
    row_order = df.Metric
    policy_cols = names(df)[2:end]
    n_cols = length(policy_cols)

    lines = String[
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Road transportation and food outcomes at the 3B-gallon-equivalent anchor}",
        "\\label{tab:road_spillovers_3B}",
        "\\begin{tabular}{l"*"c"^n_cols*"}",
        "\\hline\\hline",
        " & "*join(policy_cols, " & ")*" \\\\",
        "\\hline",
    ]

    for (i, m) in enumerate(row_order)
        m == "Diesel VMT quantity change (%)" && push!(lines, "\\hline")
        m == "Corn for food quantity change (%)" && push!(lines, "\\hline")
        vals = [fmt_44(m, df[i, c]) for c in policy_cols]
        push!(lines, m * " & " * join(vals, " & ") * " \\\\")
    end

    append!(lines, [
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{minipage}{0.95\\textwidth}",
        "\\footnotesize",
        "\\vspace{2mm}",
        "\\textit{Note:} Policy columns are evaluated at the stringency level that induces " *
        "a total SAF volume of 3 billion gallons (CI-threshold policy design for RFS and " *
        "tax credit, as in the main analysis). VMT and food quantity changes are relative " *
        "to the status quo; the status quo column is blank for these rows since the change " *
        "is by definition zero. Fuel shares are computed on an energy-equivalent (mile) " *
        "basis within each mode. RD and BD subtype rows are shown only if adopted under " *
        "at least one policy.",
        "\\end{minipage}",
        "\\end{table}",
    ])

    return join(lines, "\n")
end

# 사용:
df_44 = build_table_4_4(sq, results_target)
save_table_4_4_csv(df_44, OUTPUT_DIR)
println(to_latex_4_4(df_44))


# ---------------------------------------------------------------------------
# Section 4.4 Figure: Price % change from status quo at the 3B anchor
# (sq, results_target 재사용 — 4.3/4.4 표 코드 뒤에 이어붙임)
# ---------------------------------------------------------------------------

using Plots

const POL4 = [(:carbontax, "CT"), (:rfs, "RFS"), (:lcfs, "LCFS"), (:taxcredit, "TC")]

pct44(new, base) = (new - base) / base * 100

function plot_spillover_bars(sq, results_target)
    # conv 기준 % 변화 (5패널 공통)
    land_chg = [pct44(results_target[pt].duals.r_land, sq.duals.r_land) for (pt, _) in POL4]
    # VMT는 물량(x)이 아니라 소비자 가격(p_c)의 변화율
    gas_chg = [pct44(results_target[pt].p_c[:gas], sq.p_c[:gas]) for (pt, _) in POL4]
    die_chg = [pct44(results_target[pt].p_c[:die], sq.p_c[:die]) for (pt, _) in POL4]

    # corn/soyoil: conv는 항상 존재, cs는 CT에서만 없음 (NaN → 빈 자리)
    corn_conv = [pct44(results_target[pt].p_f[:feedstock_corn_n], sq.p_f[:feedstock_corn_n]) for (pt, _) in POL4]
    corn_cs = [pt == :carbontax ? NaN :
               pct44(results_target[pt].p_f[:feedstock_corn_cs], sq.p_f[:feedstock_corn_n]) for (pt, _) in POL4]
    soy_conv = [pct44(results_target[pt].p_f[:feedstock_soy_n], sq.p_f[:feedstock_soy_n]) for (pt, _) in POL4]
    soy_cs = [pt == :carbontax ? NaN :
              pct44(results_target[pt].p_f[:feedstock_soy_cs], sq.p_f[:feedstock_soy_n]) for (pt, _) in POL4]

    labels = [l for (_, l) in POL4]
    x = 1:4

    # 빈 막대(0 또는 NaN) 자리에 "0" 텍스트를 x축 바로 위에 표시
    function annotate_zeros!(p, vals, xs, ymax; dx=0.0)
        for (i, v) in enumerate(vals)
            (isnan(v) || abs(v) < 1e-9) && annotate!(p, xs[i] + dx, ymax * 0.05, text("0", :black, :center, 13))
        end
    end

    common(color; show_ylabel) = (xticks=(x, labels), legend=false, titlefontsize=20, grid=true,
        guidefontsize=17, tickfontsize=16, left_margin=show_ylabel ? 10Plots.mm : 4Plots.mm,
        bottom_margin=5Plots.mm, top_margin=4Plots.mm,
        ylabel=show_ylabel ? "% change from status quo" : "", color=color, linewidth=0.5)

    # 단일 막대 패널
    function single_panel(vals, title, ymax, color; show_ylabel=false)
        p = bar(x, vals, title=title, ylims=(0, ymax); common(color; show_ylabel=show_ylabel)...)
        annotate_zeros!(p, vals, x, ymax)
        return p
    end

    # conv/cs 쌍 막대 패널 (corn, soyoil) — cs는 빗금, 같은 색상 계열
    function paired_panel(conv_vals, cs_vals, title, ymax, color; show_ylabel=false)
        p = bar(x .- 0.19, conv_vals, bar_width=0.38, title=title, ylims=(0, ymax);
            common(color; show_ylabel=show_ylabel)...)
        bar!(p, x .+ 0.19, cs_vals, bar_width=0.38, color=color, fillstyle=:x, linewidth=1.0)
        annotate_zeros!(p, conv_vals, x, ymax; dx=-0.19)
        annotate_zeros!(p, cs_vals, x, ymax; dx=0.19)
        return p
    end

    TOP_YMAX = 85
    BOT_YMAX = 8

    p_rent = single_panel(land_chg, "Land rent", TOP_YMAX, :gray50; show_ylabel=true)
    p_corn = paired_panel(corn_conv, corn_cs, "Corn", TOP_YMAX, :goldenrod)
    p_soy = paired_panel(soy_conv, soy_cs, "Soyoil", TOP_YMAX, :seagreen)
    p_gas = single_panel(gas_chg, "Gasoline VMT price", BOT_YMAX, RGB(0.87, 0.79, 0.60); show_ylabel=true)
    p_die = single_panel(die_chg, "Diesel VMT price", BOT_YMAX, :saddlebrown)

    p_leg = plot(legend=:top, legendcolumns=2, grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1), framestyle=:none, legendfontsize=17)
    bar!(p_leg, [NaN], [NaN], label="Conventional", color=:gray50)
    bar!(p_leg, [NaN], [NaN], label="Climate-smart", color=:gray50, fillstyle=:x)

    # 빈 스페이서 subplot으로 윗줄-아랫줄 사이 간격을 확보 (margin 조작 대신)
    p_spacer = plot(framestyle=:none, legend=false, grid=false,
        xlims=(0, 1), ylims=(0, 1), background_color=:white)

    # 3행 3열: 1행=윗줄 패널, 2행=스페이서(얇게), 3행=아랫줄 패널+범례
    return plot(
        p_rent, p_corn, p_soy,
        p_spacer, p_spacer, p_spacer,
        p_gas, p_die, p_leg,
        layout=grid(3, 3, heights=[0.48, 0.05, 0.47]),
        size=(1700, 950))
end


spillover = plot_spillover_bars(sq, results_target)
savefig(spillover, joinpath(FIGURE_DIR, "spillover.png"))



# ---------------------------------------------------------------------------
# Table C: Welfare decomposition at the 3B-gallon-equivalent anchor
# (sq, results_target 재사용 — analysis.jl 의 calculate_* 함수들을 그대로 사용)
# ---------------------------------------------------------------------------

using CSV, JLD2

const SCEN_C = [(:carbontax, "Carbon tax"), (:rfs, "RFS"), (:lcfs, "LCFS"), (:taxcredit, "Tax credit")]
const SECTORS_C = [:avi, :gas, :die, :corn, :soyoil, :soymeal]
const SECTOR_LABELS_C = Dict(:avi => "Aviation", :gas => "Gasoline", :die => "Diesel",
    :corn => "Corn", :soyoil => "Soyoil", :soymeal => "Soymeal")

"""
    build_table_C_from_welfare(welfare_target)
 
run_model.jl 에서 이미 계산해 저장한 welfare_target (display_comparison_tables 반환값)을
그대로 표로 변환한다. 재계산 없이 welfare_target.welfare_summary[pt].cs_by_sector[s],
welfare_target.aac_results[pt] 를 사용.
 
AAC(private/social)는 status quo 대비 변화율이 아니라 3B 지점에서의 수준값
(\$/ton CO2e)이므로 status quo 열 없이 정책 4개 열만 채운다.
"""
function build_table_C_from_welfare(welfare_target)
    ws = welfare_target.welfare_summary
    aac = welfare_target.aac_results

    rows = ["Total consumer surplus"]
    append!(rows, ["CS: $(SECTOR_LABELS_C[s])" for s in SECTORS_C])
    push!(rows, "Producer surplus")
    push!(rows, "Govt revenue")
    push!(rows, "Environmental benefit")
    push!(rows, "Private surplus")
    push!(rows, "Social welfare")
    push!(rows, "AAC private (\$/ton CO2e)")
    push!(rows, "AAC social (\$/ton CO2e)")

    df = DataFrame(Metric=rows)
    for (pt, label) in SCEN_C
        w = ws[pt]
        a = aac[pt]
        col = [w.cs_change]
        append!(col, [w.cs_by_sector[s] for s in SECTORS_C])
        append!(col, [w.ps_land_change, w.gr_change, w.env_benefit,
            w.private_surplus, w.social_welfare,
            a.aac_private, a.aac_social])
        df[!, Symbol(label)] = col
    end
    return df
end

"""
    build_table_C(sq, results_target, policy_configs_target, params; scc=190.0)
 
welfare_target 저장값이 없을 때, calculate_* 함수들을 직접 호출해 재계산하는 대안.
 
주의: calculate_gr_changes → calculate_gov_revenue_change 는 sol.implicit_taxes 를,
calculate_environmental_benefit 은 sol.emissions 를 참조한다. extract_solution 의
순수 반환값에는 이 두 필드가 없으므로, 여기서 먼저 계산해 merge 로 붙여준다
(extended_grid.jl 의 run_extended_analysis 가 하는 것과 동일한 enrich 단계).
 
policy_configs_target: run_model.jl 의 policy_configs_target (정책별 config NamedTuple)
"""
function build_table_C(sq, results_target, policy_configs_target, params; scc=190.0)
    # status quo enrich (implicit_taxes는 정책이 없으므로 전부 0)
    sq_enriched = merge(sq, (
        implicit_taxes=calculate_implicit_taxes(sq, params, SAFAnalysis.SQ_CONFIG),
        emissions=calculate_emissions_detail(sq, params)
    ))

    enriched = Dict{Symbol,Any}(:statusquo => sq_enriched)
    for (pt, _) in SCEN_C
        sol = results_target[pt]
        config = policy_configs_target[pt]
        enriched[pt] = merge(sol, (
            implicit_taxes=calculate_implicit_taxes(sol, params, config),
            emissions=calculate_emissions_detail(sol, params)
        ))
    end

    cs = calculate_cs_changes(enriched, sq_enriched, params)
    ps = calculate_ps_land_changes(enriched, sq_enriched, params)
    gr = calculate_gr_changes(enriched)
    env = calculate_environmental_benefit(enriched, sq_enriched, scc)
    welf = calculate_total_welfare(cs, ps, gr, env)

    return build_table_C_from_welfare((welfare_summary=welf,))
end

# CSV 저장 -------------------------------------------------------------------

function save_table_C_csv(df, output_dir; filename="table_C_welfare_3B.csv")
    path = joinpath(output_dir, filename)
    CSV.write(path, df)
    println("✓ Saved: $path")
    return path
end

# LaTeX 변환 -------------------------------------------------------------------
# 모든 값은 Billion $ 단위, 소수점 2자리

function fmt_C(x)
    isnan(x) && return "--"
    s = @sprintf("%.2f", x)
    x > 0 && (s = "+" * s)
    return s
end

function to_latex_C(df; scc=190)
    row_order = df.Metric
    policy_cols = names(df)[2:end]
    n_cols = length(policy_cols)

    lines = String[
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Welfare decomposition at the 3B-gallon-equivalent anchor (Billion \\\$)}",
        "\\label{tab:welfare_3B}",
        "\\begin{tabular}{l"*"c"^n_cols*"}",
        "\\hline\\hline",
        " & "*join(policy_cols, " & ")*" \\\\",
        "\\hline",
    ]

    for (i, m) in enumerate(row_order)
        m == "CS: Aviation" && push!(lines, "\\hline")
        m == "Producer surplus" && push!(lines, "\\hline")
        m == "Private surplus" && push!(lines, "\\hline")
        m == "AAC private (\$/ton CO2e)" && push!(lines, "\\hline")
        vals = [fmt_C(df[i, c]) for c in policy_cols]
        push!(lines, m * " & " * join(vals, " & ") * " \\\\")
    end

    append!(lines, [
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{minipage}{0.95\\textwidth}",
        "\\footnotesize",
        "\\vspace{2mm}",
        "\\textit{Note:} All values are changes relative to the status quo (no SAF policy), " *
        "evaluated at each policy's stringency level that induces a total SAF volume of " *
        "3 billion gallons (CI-threshold policy design for RFS and tax credit, as in the " *
        "main analysis). Private surplus is the sum of consumer surplus (CS), producer " *
        "surplus (PS), and government revenue; social welfare is private surplus plus " *
        "environmental benefit, valued at a social cost of carbon of \\\$$(scc)/ton CO\\textsubscript{2}e.",
        "\\end{minipage}",
        "\\end{table}",
    ])

    return join(lines, "\n")
end


# 사용:
# 방법 A (권장, 재계산 없음 — run_model.jl 이 이미 welfare_target 을 저장해뒀다면):
@load joinpath(OUTPUT_DIR, "results_target.jld2") welfare_target
df_C = build_table_C_from_welfare(welfare_target)
println(to_latex_C(df_C; scc=190))
