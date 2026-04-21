# diagnose_carbontax_kink.jl
# Carbon tax MAC curve의 5B 선 근처 꺾임 진단
# extended_grid.jl 실행 후 results_extended_analysis, mac_extended가 메모리에 있다고 가정

cd(@__DIR__)

using DataFrames
using Printf

# =================================================================================
# 1. Carbon tax 시나리오를 abatement 순으로 정렬해서 5B 근방 추출
# =================================================================================

solutions = results_extended_analysis.solutions
welfare_summary = results_extended_analysis.welfare_summary
mac_data = mac_extended[:carbontax]

statusquo_em = solutions[:statusquo].emissions.total

# 5B에 해당하는 abatement 값
abatement_5B = statusquo_em - ep_5B[:carbontax === :carbontax ? :carbontax : :carbontax].actual_emission
# ep_5B는 extended_grid.jl에서 로드된 변수
abatement_5B_val = statusquo_em - ep_5B[:carbontax].actual_emission

println("Status quo total emissions: $(@sprintf("%.6f", statusquo_em)) billion ton CO2e")
println("5B abatement target:        $(@sprintf("%.6f", abatement_5B_val)) billion ton CO2e")
println("(emissions at 5B target:    $(@sprintf("%.6f", statusquo_em - abatement_5B_val)))")

# =================================================================================
# 2. MAC data를 DataFrame으로 변환
# =================================================================================

rows = []
for d in mac_data
    abatement = statusquo_em - d.emission
    t_val = parse(Int, split(String(d.scenario), "_")[2])  # carbon tax 값 복원

    push!(rows, (
        scenario=String(d.scenario),
        t=Float64(t_val),
        emission=d.emission,
        abatement=abatement,
        private_surplus=d.private_surplus,
        social_welfare=d.social_welfare,
        Δemission=d.Δemission,
        Δprivate_surplus=d.Δprivate_surplus,
        Δsocial_welfare=d.Δsocial_welfare,
        mac_private=d.mac_private,
        mac_social=d.mac_social,
    ))
end

df_mac = DataFrame(rows)
sort!(df_mac, :abatement)

# =================================================================================
# 3. 5B 근방 ±0.02 billion ton 구간 필터링
# =================================================================================

window = 0.02
df_near = filter(r -> abs(r.abatement - abatement_5B_val) <= window, df_mac)

println("\n" * "="^110)
println("Carbon Tax MAC near 5B line  (abatement window: $(abatement_5B_val - window) ~ $(abatement_5B_val + window))")
println("="^110)
println(@sprintf("%-30s %8s %10s %10s %12s %12s %12s %12s %12s",
    "Scenario", "t(\$/t)", "emission", "abatement",
    "priv_surp", "soc_welf", "mac_priv", "mac_soc", "Δemission"))
println("-"^110)

for r in eachrow(df_near)
    marker = abs(r.abatement - abatement_5B_val) < 0.005 ? " ◀ 5B" : ""
    println(@sprintf("%-30s %8.0f %10.5f %10.5f %12.4f %12.4f %12.2f %12.2f %12.6f%s",
        r.scenario, r.t, r.emission, r.abatement,
        r.private_surplus, r.social_welfare,
        r.mac_private, r.mac_social, r.Δemission, marker))
end

# =================================================================================
# 4. 꺾임 감지: MAC 변화가 큰 구간 찾기
# =================================================================================

println("\n" * "="^110)
println("Points with large MAC jumps (|ΔMAC_private| > 50)")
println("="^110)
println(@sprintf("%-30s %8s %10s %12s %12s %12s",
    "Scenario", "t(\$/t)", "abatement", "mac_priv", "ΔMAC_priv", "mac_soc"))
println("-"^110)

prev_mac = df_mac.mac_private[1]
for r in eachrow(df_mac)
    delta_mac = r.mac_private - prev_mac
    if abs(delta_mac) > 50
        println(@sprintf("%-30s %8.0f %10.5f %12.2f %12.2f %12.2f",
            r.scenario, r.t, r.abatement,
            r.mac_private, delta_mac, r.mac_social))
    end
    prev_mac = r.mac_private
end

# =================================================================================
# 5. 5B 근방 시나리오의 fuel quantities, prices, duals 출력
# =================================================================================

println("\n" * "="^110)
println("Fuel quantities & duals near 5B (abatement window ±0.015)")
println("="^110)

SAF_FUELS = [:saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy, :saf_atj_conv, :saf_atj_cs]

df_detail = filter(r -> abs(r.abatement - abatement_5B_val) <= 0.015, df_mac)

for r in eachrow(df_detail)
    scen = Symbol(r.scenario)
    sol = solutions[scen]
    marker = abs(r.abatement - abatement_5B_val) < 0.005 ? "  ◀◀ 5B" : ""

    println("\n--- $(r.scenario)  t=$(r.t)  abatement=$(@sprintf("%.5f", r.abatement))$marker ---")
    println("  Emissions  : total=$(@sprintf("%.5f", sol.emissions.total))  avi=$(@sprintf("%.5f", sol.emissions.aviation))  road=$(@sprintf("%.5f", sol.emissions.road))")
    println("  MAC private: $(@sprintf("%.2f", r.mac_private))   MAC social: $(@sprintf("%.2f", r.mac_social))")
    println("  Surplus    : private=$(@sprintf("%.4f", r.private_surplus))  social=$(@sprintf("%.4f", r.social_welfare))")
    println("  Prices     : p_avi=$(@sprintf("%.4f", sol.p_c[:avi]))  p_gas=$(@sprintf("%.4f", sol.p_c[:gas]))  p_die=$(@sprintf("%.4f", sol.p_c[:die]))")
    println("  r_land     : $(@sprintf("%.4f", sol.duals.r_land))")
    println("  λ_rfs      : $(@sprintf("%.4f", sol.duals.λ_rfs))")

    print("  SAF quantities: ")
    for fuel in SAF_FUELS
        @printf("  %s=%.4f", fuel, sol.q[fuel])
    end
    println()

    print("  Road biofuels: ")
    for fuel in [:ethanol, :rd_soy, :rd_nonsoy, :biodiesel_soy, :biodiesel_nonsoy]
        @printf("  %s=%.4f", fuel, sol.q[fuel])
    end
    println()
end

# =================================================================================
# 6. Δ welfare components 분해 (5B 근방)
# =================================================================================

println("\n" * "="^110)
println("Welfare component decomposition near 5B")
println("="^110)
println(@sprintf("%-30s %8s %12s %12s %12s %12s %12s",
    "Scenario", "t(\$/t)", "Δsoc_welf", "Δemissions", "MAC_social", "SAF_nonsoy", "SAF_atj_conv"))
println("-"^110)

for r in eachrow(df_detail)
    scen = Symbol(r.scenario)
    if !haskey(welfare_summary, scen)
        continue
    end
    sol = solutions[scen]
    saf_nonsoy_qty = sol.q[:saf_hefa_nonsoy]
    saf_atj_conv_qty = sol.q[:saf_atj_conv]
    println(@sprintf("%-30s %8.0f %12.4f %12.6f %12.2f %12.4f %12.4f",
        r.scenario, r.t,
        r.Δsocial_welfare, r.Δemission, r.mac_social,
        saf_nonsoy_qty, saf_atj_conv_qty))
end

# =================================================================================
# 7. Carbon tax 370~380 범위 상세 정보
# =================================================================================

println("\n" * "="^110)
println("Carbon Tax 370~380 range - detailed view")
println("="^110)
println(@sprintf("%-30s %8s %12s %12s %12s %12s %12s",
    "Scenario", "t(\$/t)", "Δsoc_welf", "Δemissions", "MAC_social", "SAF_nonsoy", "SAF_atj_conv"))
println("-"^110)

df_tax_range = filter(r -> 370 <= r.t <= 380, df_mac);
for r in eachrow(df_tax_range)
    marker = ""
    if r.t ≈ 374
        marker = " ◀ Non soy HEFA adopted"
    elseif r.t ≈ 377
        marker = " ◀ Conv ATJ SAF adopted"
    end
    scen = Symbol(r.scenario)
    sol = solutions[scen]
    saf_nonsoy_qty = sol.q[:saf_hefa_nonsoy]
    saf_atj_conv_qty = sol.q[:saf_atj_conv]
    println(@sprintf("%-30s %8.0f %12.4f %12.6f %12.2f %12.4f %12.4f%s",
        r.scenario, r.t,
        r.Δsocial_welfare, r.Δemission, r.mac_social,
        saf_nonsoy_qty, saf_atj_conv_qty, marker))
end

println("\n✓ Diagnosis complete.")

# plot_carbontax_kink.jl
# diagnose_carbontax_kink.jl 실행 후 df_mac, abatement_5B_val이 메모리에 있다고 가정

using Plots
using Printf

# =================================================================================
# 0. 꺾임 지점 설정 (t=374, 377)
# =================================================================================

# t=374와 t=377 근처의 인덱스 찾기
kink_indices = Int[]
for (idx, row) in enumerate(eachrow(df_mac))
    if abs(row.t - 374) < 0.1 || abs(row.t - 377) < 0.1
        push!(kink_indices, idx)
    end
end

if isempty(kink_indices)
    @warn "kink 지점 미발견"
    push!(kink_indices, div(nrow(df_mac), 2))
end

println("지정된 kink 지점 $(length(kink_indices))개")
for ki in kink_indices
    println(@sprintf("  index=%d  abatement=%.5f  mac_social=%.2f  t=%.0f",
        ki, df_mac.abatement[ki], df_mac.mac_social[ki], df_mac.t[ki]))
end

# =================================================================================
# 1. 통합 그래프 — 모든 kink를 한 그래프에 표시
# =================================================================================

WINDOW = 0.04

# 모든 kink 주변 데이터 수집
all_kinks_data = []
for kink_i in kink_indices
    kink_ab = df_mac.abatement[kink_i]
    kink_t = df_mac.t[kink_i]
    df_win = filter(r -> abs(r.abatement - kink_ab) <= WINDOW, df_mac)
    if nrow(df_win) >= 3
        push!(all_kinks_data, (kink_i=kink_i, kink_ab=kink_ab, kink_t=kink_t, df_win=df_win))
    end
end

if !isempty(all_kinks_data)
    # 첫 번째 kink 데이터를 기본으로 사용
    d = all_kinks_data[1]
    x = d.df_win.abatement
    y1 = d.df_win.mac_social
    y2 = d.df_win.Δemission
    y3 = d.df_win.Δsocial_welfare

    p1 = plot(x, y1;
        label="MAC social",
        color=:steelblue,
        linewidth=2.5,
        marker=:circle,
        markersize=2,
        ylabel=raw"$/tCO2e",
        title="MAC Social",
        titlefontsize=14,
        guidefontsize=12,
        legend=:bottomleft,
        legendfontsize=12,
        grid=true,
        gridstyle=:dash,
        gridcolor=:lightgray,
        xlims=(0.08, 0.1),
    )
    # 모든 kink를 표시
    for d in all_kinks_data
        vline!(p1, [d.kink_ab]; linestyle=:dash, color=:red, lw=1.5, alpha=0.8, label="")
    end

    p2 = plot(x, y2;
        label="Δemissions / step",
        color=:darkorange,
        linewidth=2.5,
        marker=:diamond,
        markersize=2,
        ylabel="bn tCO2e",
        title="Δ Emissions per step",
        titlefontsize=14,
        guidefontsize=12,
        legend=:bottomleft,
        legendfontsize=12,
        grid=true,
        gridstyle=:dash,
        gridcolor=:lightgray,
        xlims=(0.08, 0.1),
    )
    for d in all_kinks_data
        vline!(p2, [d.kink_ab]; linestyle=:dash, color=:red, lw=1.5, alpha=0.8, label="")
    end
    hline!(p2, [0.0]; linestyle=:solid, color=:gray, lw=1.0, alpha=0.5, label="zero")

    p3 = plot(x, y3;
        label="Δ social welfare / step",
        color=:mediumseagreen,
        linewidth=2.5,
        marker=:utriangle,
        markersize=2,
        ylabel=raw"bn $",
        title="Δ Social Welfare per step",
        titlefontsize=14,
        guidefontsize=12,
        legend=:bottomleft,
        legendfontsize=12,
        grid=true,
        gridstyle=:dash,
        gridcolor=:lightgray,
        xlims=(0.08, 0.1),
    )
    for d in all_kinks_data
        vline!(p3, [d.kink_ab]; linestyle=:dash, color=:red, lw=1.5, alpha=0.8, label="")
    end
    hline!(p3, [0.0]; linestyle=:solid, color=:gray, lw=1.0, alpha=0.5, label="zero")

    # ═════════════════════════════════════════════════════════════════════════════
    # p4: Stacked jet fuel and SAF quantity
    # ═════════════════════════════════════════════════════════════════════════════

    SAF_FUELS = [:saf_hefa_conv, :saf_atj_cs, :saf_atj_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    SAF_COLORS = [:green, :red, :blue, :orange, :purple]
    SAF_LABELS = ["Conventional HEFA-SAF", "Climate-Smart ATJ-SAF", "Conventional ATJ-SAF", "Climate-Smart HEFA-SAF", "Non-soy HEFA-SAF"]

    # Extract fuel quantities for each scenario (ordered by abatement)
    jet_fuel_vals = Float64[]
    t_vals = Float64[]  # t 값 저장
    saf_totals = Array{Float64,2}(undef, length(SAF_FUELS), nrow(d.df_win))  # Each SAF type separately

    for (idx, r) in enumerate(eachrow(d.df_win))
        scen = Symbol(r.scenario)
        sol = solutions[scen]

        push!(jet_fuel_vals, sol.q[:jet_fuel])
        push!(t_vals, r.t)  # t 값 저장

        # Store each SAF type
        for (saf_idx, fuel) in enumerate(SAF_FUELS)
            saf_totals[saf_idx, idx] = sol.q[fuel]
        end
    end

    p4 = plot(
        xlabel="Abatement (bn tCO2e)",
        ylabel="Quantity (B gallons)",
        title="Stacked Jet Fuel and SAF Quantity",
        titlefontsize=14,
        guidefontsize=12,
        legend=:bottomleft,
        legendfontsize=12,
        grid=true,
        gridstyle=:dash,
        gridcolor=:lightgray,
        xlims=(0.08, 0.1),
        ylims=(10, 15),
    )

    # Plot jet fuel (bottom)
    plot!(p4, x, jet_fuel_vals,
        fillrange=0,
        fillalpha=0.7,
        fillcolor=:black,
        linewidth=2.0,
        color=:black,
        label="Jet Fuel",
    )

    # Plot stacked SAF (each type separately) - only if non-zero
    cumsum_saf = copy(jet_fuel_vals)  # Start stacking from jet fuel level
    final_cumsum = zeros(length(x))
    saf_markers_x = []
    saf_markers_y = []

    for (saf_idx, fuel) in enumerate(SAF_FUELS)
        saf_vals = vec(saf_totals[saf_idx, :])
        # Only plot if any value is non-zero
        if any(saf_vals .> 1e-6)
            new_cumsum = cumsum_saf .+ saf_vals
            plot!(p4, x, new_cumsum,
                fillrange=cumsum_saf,
                fillalpha=0.7,
                fillcolor=SAF_COLORS[saf_idx],
                linewidth=2.0,
                color=SAF_COLORS[saf_idx],
                label=SAF_LABELS[saf_idx],
            )

            # 각 SAF 층의 위에 marker 추가 (처음으로 도입되는 지점만)
            for (j, val) in enumerate(saf_vals)
                if val .> 1e-6
                    # 첫 지점이거나 이전 지점에서 0이었던 경우만 marker 추가 (처음 도입되는 순간)
                    if j == 1 || saf_vals[j-1] <= 1e-6
                        push!(saf_markers_x, x[j])
                        push!(saf_markers_y, new_cumsum[j])
                    end
                end
            end

            cumsum_saf = new_cumsum
        end
    end

    # 모든 SAF 층의 marker 추가
    if !isempty(saf_markers_x)
        plot!(p4, saf_markers_x, saf_markers_y,
            color=:black,
            marker=:circle,
            markersize=3,
            markerstrokewidth=0.5,
            markerstrokecolor=:black,
            linewidth=0,
            label="",
        )
    end

    # 모든 kink 표시
    for d in all_kinks_data
        vline!(p4, [d.kink_ab]; linestyle=:dash, color=:red, lw=1.5, alpha=0.8, label="")
    end

    # Kink 라벨
    kink_labels = join([@sprintf("t=%.0f", d.kink_t) for d in all_kinks_data], ", ")

    fig = plot(p1, p2, p3, p4;
        layout=(4, 1),
        size=(1000, 1280),
        left_margin=12Plots.mm,
        right_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
        top_margin=4Plots.mm,
        plot_title=@sprintf("Carbon Tax MAC Kink Analysis  (kinks @ %s)", kink_labels),
        plot_titlefontsize=16,
    )

    display(fig)
end

# =================================================================================
# plot_zpc.jl
# diagnose_carbontax_kink.jl 실행 후 df_mac, solutions, abatement_5B_val 이 메모리에 있다고 가정
# =================================================================================

begin
    # 파라미터 로드
    alpha_nonsoy = params.coeff.alpha[:saf_hefa_nonsoy]
    alpha_atj_conv = params.coeff.alpha[:saf_atj_conv]
    nonsoy_fp = params.coeff.nonsoy_feedstock_price
    delta = params.coeff.delta
    fuel_cost = params.supply.fuel
    r_jet = params.coeff.r[:jet_fuel]
    beta_saf = params.coeff.beta[(:saf, :jet_fuel)]

    # process_mc 재계산 함수
    function calc_process_mc_hefa(sol)
        total = sol.q[:saf_hefa_conv] + sol.q[:saf_hefa_cs] + sol.q[:saf_hefa_nonsoy] +
                sol.q[:rd_soy] + sol.q[:rd_nonsoy]
        fc = fuel_cost[:saf_hefa_shared]
        return fc.c0 + fc.c1 * total + fc.c2 * max(0.0, total - fc.v)^2
    end

    function calc_process_mc_atj(sol)
        total = sol.q[:saf_atj_conv] + sol.q[:saf_atj_cs]
        fc = fuel_cost[:saf_atj_shared]
        return fc.c0 + fc.c1 * total + fc.c2 * max(0.0, total - fc.v)^2
    end

    function calc_policy_adj(sol, t, g)
        ct = t * delta[g]
        λ_ns = sol.duals.λ_nonsoy_capacity
        ns_adj = (g == :saf_hefa_nonsoy) ? λ_ns * alpha_nonsoy : 0.0
        return ct + ns_adj
    end

    function calc_price_per_unit_saf(sol)
        return r_jet * beta_saf * sol.p_c[:avi]
    end

    # x축 범위 설정 (전체 carbontax 시나리오 사용)
    WINDOW = 0.04
    kink_ab = abatement_5B_val   # 또는 원하는 중심값으로 변경
    df_win = filter(r -> abs(r.abatement - kink_ab) <= WINDOW, df_mac)
    sort!(df_win, :abatement)

    # ZPC 계산
    zpc_x = Float64[]
    zpc_nonsoy = Float64[]
    zpc_atj = Float64[]

    for r in eachrow(df_win)
        scen = Symbol(r.scenario)
        sol = solutions[scen]
        t = r.t
        p_corn = sol.p_f[:feedstock_corn_n]

        mc_hefa = calc_process_mc_hefa(sol)
        mc_atj = calc_process_mc_atj(sol)
        ppu = calc_price_per_unit_saf(sol)

        zpc_ns = mc_hefa + alpha_nonsoy * nonsoy_fp +
                 calc_policy_adj(sol, t, :saf_hefa_nonsoy) - ppu

        zpc_at = mc_atj + (alpha_atj_conv - 0.159) * p_corn +
                 calc_policy_adj(sol, t, :saf_atj_conv) - ppu

        push!(zpc_x, r.abatement)
        push!(zpc_nonsoy, zpc_ns)
        push!(zpc_atj, zpc_at)
    end

    # kink 지점
    kink_indices = Int[]
    for (idx, row) in enumerate(eachrow(df_win))
        if abs(row.t - 374) < 0.1 || abs(row.t - 377) < 0.1
            push!(kink_indices, idx)
        end
    end
    kink_abs = [df_win.abatement[ki] for ki in kink_indices]
    kink_labels = join([@sprintf("t=%.0f", df_win.t[ki]) for ki in kink_indices], ", ")

    # =================================================================================
    # 그래프
    # =================================================================================

    p_nonsoy = plot(zpc_x, zpc_nonsoy;
        label="Non-soy HEFA SAF",
        color=:purple,
        linewidth=2.5,
        marker=:circle,
        markersize=2,
        xlabel="Abatement (bn tCO2e)",
        ylabel=raw"$/gallon",
        title="Zero Profit Condition — Non-soy HEFA SAF",
        titlefontsize=14,
        guidefontsize=12,
        legend=:topright,
        legendfontsize=11,
        grid=true,
        gridstyle=:dash,
        gridcolor=:lightgray,
        xlims=(0.08, 0.1),
        ylims=(-0.05, 0.1),
    )
    hline!(p_nonsoy, [0.0]; linestyle=:solid, color=:black, lw=1.5, alpha=0.7, label="ZPC = 0")
    for kab in kink_abs
        vline!(p_nonsoy, [kab]; linestyle=:dash, color=:red, lw=1.5, alpha=0.8, label="")
    end

    p_atj = plot(zpc_x, zpc_atj;
        label="Conv ATJ SAF",
        color=:blue,
        linewidth=2.5,
        marker=:diamond,
        markersize=2,
        xlabel="Abatement (bn tCO2e)",
        ylabel=raw"$/gallon",
        title="Zero Profit Condition — Conv ATJ SAF",
        titlefontsize=14,
        guidefontsize=12,
        legend=:topright,
        legendfontsize=11,
        grid=true,
        gridstyle=:dash,
        gridcolor=:lightgray,
        xlims=(0.08, 0.1),
        ylims=(-0.05, 0.1),
    )
    hline!(p_atj, [0.0]; linestyle=:solid, color=:black, lw=1.5, alpha=0.7, label="ZPC = 0")
    for kab in kink_abs
        vline!(p_atj, [kab]; linestyle=:dash, color=:red, lw=1.5, alpha=0.8, label="")
    end

    fig_zpc = plot(p_nonsoy, p_atj;
        layout=(2, 1),
        size=(1000, 640),
        left_margin=12Plots.mm,
        right_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
        top_margin=4Plots.mm,
        plot_title=@sprintf("Zero Profit Condition  (kinks @ %s)", kink_labels),
        plot_titlefontsize=16,
    )

    display(fig_zpc)
end




# =================================================================================
# diagnose_all_kinks.jl
# diagnose_carbontax_kink.jl (+ plot_zpc.jl) 실행 후 이어서 실행
# 이미 메모리에 있는 변수: solutions, welfare_summary, mac_extended,
#   statusquo_em, df_mac (carbontax용), EXTENDED_POLICY_MATRIX, params, ep_3B, ep_5B
# =================================================================================

using DataFrames
using Printf
using Plots
using Statistics

# =================================================================================
# 0. 공통 유틸리티 (기존 변수와 충돌 방지를 위해 접두어 사용)
# =================================================================================

const SAF_FUELS_ALL = [:saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy, :saf_atj_conv, :saf_atj_cs]
const ROAD_BIOFUELS_ALL = [:ethanol, :rd_soy, :rd_nonsoy, :biodiesel_soy, :biodiesel_nonsoy]

const POLICY_META = Dict(
    :carbontax => (label="Carbon Tax", unit="\$/ton CO₂e", field=:t),
    :rfs       => (label="RFS Aviation", unit="θ_avi", field=:θ_avi),
    :lcfs      => (label="LCFS", unit="σ", field=:σ),
    :taxcredit => (label="Tax Credit", unit="\$/gallon", field=:p)
)

function get_pval(scenario_name::Symbol, policy_type::Symbol)
    config = EXTENDED_POLICY_MATRIX[scenario_name]
    return getfield(config, POLICY_META[policy_type].field)
end

# =================================================================================
# 1. MAC → DataFrame 변환 (모든 정책 공통)
# =================================================================================

function mac_to_df(mac_data, policy_type)
    rows = []
    for d in mac_data
        ab = statusquo_em - d.emission
        pv = get_pval(d.scenario, policy_type)
        sol = solutions[d.scenario]
        push!(rows, (
            scenario      = String(d.scenario),
            policy_value  = pv,
            emission      = d.emission,
            abatement     = ab,
            Δemission     = d.Δemission,
            mac_private   = d.mac_private,
            mac_social    = d.mac_social,
            # SAF
            saf_hefa_conv   = sol.q[:saf_hefa_conv],
            saf_hefa_cs     = sol.q[:saf_hefa_cs],
            saf_hefa_nonsoy = sol.q[:saf_hefa_nonsoy],
            saf_atj_conv    = sol.q[:saf_atj_conv],
            saf_atj_cs      = sol.q[:saf_atj_cs],
            jet_fuel        = sol.q[:jet_fuel],
            # Road
            ethanol         = sol.q[:ethanol],
            rd_soy          = sol.q[:rd_soy],
            rd_nonsoy       = sol.q[:rd_nonsoy],
            biodiesel_soy   = sol.q[:biodiesel_soy],
            biodiesel_nonsoy= sol.q[:biodiesel_nonsoy],
            # Prices
            p_avi   = sol.p_c[:avi],
            p_gas   = sol.p_c[:gas],
            p_die   = sol.p_c[:die],
            p_corn_n  = sol.p_f[:feedstock_corn_n],
            p_corn_cs = sol.p_f[:feedstock_corn_cs],
            p_soy_n   = sol.p_f[:feedstock_soy_n],
            p_soy_cs  = sol.p_f[:feedstock_soy_cs],
            # Duals
            λ_rfs           = sol.duals.λ_rfs,
            λ_rfs_avi       = sol.duals.λ_rfs_avi,
            λ_lcfs          = sol.duals.λ_lcfs,
            λ_blendwall_eth = sol.duals.λ_blendwall_ethanol,
            λ_blendwall_bd  = sol.duals.λ_blendwall_biodiesel,
            λ_nonsoy_cap    = sol.duals.λ_nonsoy_capacity,
            r_land          = sol.duals.r_land,
            # Emissions
            em_avi  = sol.emissions.aviation,
            em_road = sol.emissions.road,
            em_food = sol.emissions.food,
        ))
    end
    return sort!(DataFrame(rows), :abatement)
end


# =================================================================================
# 2. Kink / 기술전환 / Dual 전환 감지 함수
# =================================================================================

"""Δ²MAC이 큰 지점 감지"""
function detect_mac_kinks(df; mac_col=:mac_social, threshold_q=0.93, min_gap=2)
    n = nrow(df)
    n < 5 && return Int[]
    Δ = diff(df[!, mac_col])
    Δ² = abs.(diff(Δ))
    thr = quantile(Δ², threshold_q)
    cands = findall(x -> x > thr, Δ²) .+ 1
    isempty(cands) && return Int[]
    out = [cands[1]]
    for i in 2:length(cands)
        if cands[i] - out[end] >= min_gap
            push!(out, cands[i])
        elseif Δ²[cands[i]-1] > Δ²[out[end]-1]
            out[end] = cands[i]
        end
    end
    return out
end

"""SAF 기술 도입(entry) / 퇴출(exit) 감지"""
function detect_tech_transitions(df; thr=1e-4)
    trans = []
    for fuel in SAF_FUELS_ALL
        vals = df[!, fuel]
        for i in 2:length(vals)
            if vals[i-1] <= thr && vals[i] > thr
                push!(trans, (idx=i, fuel=fuel, type=:entry,
                    pval=df.policy_value[i], ab=df.abatement[i], qty=vals[i]))
            elseif vals[i-1] > thr && vals[i] <= thr
                push!(trans, (idx=i, fuel=fuel, type=:exit,
                    pval=df.policy_value[i], ab=df.abatement[i], qty=vals[i-1]))
            end
        end
    end
    sort!(trans, by=x -> x.ab)
end

"""Dual variable binding ↔ unbinding 전환 감지"""
function detect_dual_trans(df; thr=1e-4)
    dcols = [:λ_rfs, :λ_rfs_avi, :λ_lcfs, :λ_blendwall_eth, :λ_blendwall_bd, :λ_nonsoy_cap]
    trans = []
    for dc in dcols
        vals = df[!, dc]
        for i in 2:length(vals)
            if vals[i-1] <= thr && vals[i] > thr
                push!(trans, (idx=i, dual=dc, type=:binding,
                    pval=df.policy_value[i], ab=df.abatement[i], val=vals[i]))
            elseif vals[i-1] > thr && vals[i] <= thr
                push!(trans, (idx=i, dual=dc, type=:unbinding,
                    pval=df.policy_value[i], ab=df.abatement[i], val=vals[i-1]))
            end
        end
    end
    sort!(trans, by=x -> x.ab)
end


# =================================================================================
# 3. 4개 정책 DataFrame 생성
# =================================================================================

println("\n" * "="^120)
println("  MAC Kink Diagnosis — All Policies")
println("="^120)

df_all = Dict(
    :carbontax  => mac_to_df(mac_extended[:carbontax], :carbontax),
    :rfs        => mac_to_df(mac_extended[:rfs], :rfs),
    :lcfs       => mac_to_df(mac_extended[:lcfs], :lcfs),
    :taxcredit  => mac_to_df(mac_extended[:taxcredit][2:end], :taxcredit)
)

# 모든 정책의 이벤트 수집
events_all = Dict()

# =================================================================================
# 4. 정책별 kink 감지 + 출력
# =================================================================================

for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
    df = df_all[pt]
    m = POLICY_META[pt]
    println("\n" * "▓"^120)
    println("  $(m.label)  ($(m.unit))")
    println("▓"^120)

    ki_priv = detect_mac_kinks(df; mac_col=:mac_private)
    ki_soc  = detect_mac_kinks(df; mac_col=:mac_social)
    ki_all  = sort(unique(vcat(ki_priv, ki_soc)))
    tt = detect_tech_transitions(df)
    dt = detect_dual_trans(df)

    events_all[pt] = (kinks=ki_all, tech=tt, duals=dt, df=df)

    # ── 4a. MAC kinks ──
    println("\n  [A] MAC Kinks  (Private $(length(ki_priv)), Social $(length(ki_soc)), Union $(length(ki_all)))")
    if !isempty(ki_all)
        println(@sprintf("  %6s  %12s  %12s  %12s  %12s  %12s",
            "Idx", "PolicyVal", "Abatement", "MAC_priv", "MAC_soc", "Δem"))
        println("  " * "-"^78)
        for k in ki_all
            println(@sprintf("  %6d  %12.4f  %12.6f  %12.2f  %12.2f  %12.6f",
                k, df.policy_value[k], df.abatement[k],
                df.mac_private[k], df.mac_social[k], df.Δemission[k]))
        end
    end

    # ── 4b. 기술 전환 ──
    println("\n  [B] Technology Transitions")
    if isempty(tt)
        println("    (없음)")
    else
        println(@sprintf("  %-22s  %-7s  %12s  %12s  %12s",
            "Fuel", "Type", "PolicyVal", "Abatement", "Quantity"))
        println("  " * "-"^70)
        for t in tt
            println(@sprintf("  %-22s  %-7s  %12.4f  %12.6f  %12.6f",
                t.fuel, t.type, t.pval, t.ab, t.qty))
        end
    end

    # ── 4c. Dual 전환 ──
    println("\n  [C] Dual Variable Transitions")
    if isempty(dt)
        println("    (없음)")
    else
        println(@sprintf("  %-24s  %-10s  %12s  %12s  %12s",
            "Dual", "Type", "PolicyVal", "Abatement", "Value"))
        println("  " * "-"^75)
        for t in dt
            println(@sprintf("  %-24s  %-10s  %12.4f  %12.6f  %12.6f",
                t.dual, t.type, t.pval, t.ab, t.val))
        end
    end

    # ── 4d. 이벤트 근방 상세 ──
    event_idxs = sort(unique(vcat(ki_all, [t.idx for t in tt], [t.idx for t in dt])))
    println("\n  [D] Detail near event points")

    for ei in event_idxs
        # 이벤트 설명
        tags = String[]
        for t in tt;  t.idx == ei && push!(tags, "$(t.fuel) $(t.type)"); end
        for t in dt;  t.idx == ei && push!(tags, "$(t.dual) $(t.type)"); end
        ei in ki_all && push!(tags, "MAC kink")
        tag_str = isempty(tags) ? "" : " ← " * join(tags, ", ")

        r1 = max(1, ei - 2)
        r2 = min(nrow(df), ei + 2)

        println("\n  ─── idx=$(ei)  $(m.unit)=$(@sprintf("%.4f", df.policy_value[ei]))$(tag_str) ───")
        println(@sprintf("  %5s  %10s  %10s  %10s  %10s  %10s │ %8s %8s %8s %8s %8s │ %8s %8s %8s",
            "idx", "pol_val", "abatement", "mac_prv", "mac_soc", "Δem",
            "hf_cv", "hf_cs", "hf_ns", "at_cv", "at_cs",
            "λ_rfs_a", "λ_lcfs", "λ_ns"))
        println("  " * "-"^145)
        for i in r1:r2
            mk = i == ei ? " ◀" : ""
            println(@sprintf("  %5d  %10.4f  %10.6f  %10.2f  %10.2f  %10.6f │ %8.4f %8.4f %8.4f %8.4f %8.4f │ %8.4f %8.4f %8.4f%s",
                i, df.policy_value[i], df.abatement[i],
                df.mac_private[i], df.mac_social[i], df.Δemission[i],
                df.saf_hefa_conv[i], df.saf_hefa_cs[i], df.saf_hefa_nonsoy[i],
                df.saf_atj_conv[i], df.saf_atj_cs[i],
                df.λ_rfs_avi[i], df.λ_lcfs[i], df.λ_nonsoy_cap[i], mk))
        end
    end

    # ── 4e. 큰 MAC 점프 (|ΔMAC_priv| > 20) ──
    println("\n  [E] Large MAC jumps (|ΔMAC_private| > 20)")
    println(@sprintf("  %-30s  %10s  %12s  %12s  %12s  %12s",
        "Scenario", m.unit, "abatement", "mac_priv", "ΔMAC_priv", "mac_soc"))
    println("  " * "-"^100)
    prev = df.mac_private[1]
    for r in eachrow(df)
        Δ = r.mac_private - prev
        if abs(Δ) > 20
            println(@sprintf("  %-30s  %10.4f  %12.6f  %12.2f  %12.2f  %12.2f",
                r.scenario, r.policy_value, r.abatement,
                r.mac_private, Δ, r.mac_social))
        end
        prev = r.mac_private
    end
end

# =================================================================================
# 5. 정책 간 비교
# =================================================================================

println("\n\n" * "█"^120)
println("  CROSS-POLICY COMPARISON")
println("█"^120)

# ── 5a. 기술 도입 순서 ──
println("\n  [A] SAF Technology Adoption Order")
println("  " * "-"^100)
println(@sprintf("  %-14s  %-22s  %12s  %12s  %8s",
    "Policy", "Fuel (entry)", "PolicyVal", "Abatement", "% of SQ"))
println("  " * "-"^100)
for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
    entries = filter(t -> t.type == :entry, events_all[pt].tech)
    if isempty(entries)
        println(@sprintf("  %-14s  (no entry detected)", POLICY_META[pt].label))
    else
        for (i, t) in enumerate(entries)
            pct = t.ab / statusquo_em * 100
            println(@sprintf("  %-14s  %-22s  %12.4f  %12.6f  %7.2f%%",
                i == 1 ? POLICY_META[pt].label : "", t.fuel, t.pval, t.ab, pct))
        end
    end
end

# ── 5b. 기술별 도입 MAC 수준 ──
println("\n  [B] Private MAC at Technology Entry")
println(@sprintf("  %-22s  %12s  %12s  %12s  %12s",
    "Fuel", "CarbonTax", "RFS", "LCFS", "TaxCredit"))
println("  " * "-"^75)
for fuel in SAF_FUELS_ALL
    vals = String[]
    for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
        es = filter(t -> t.type == :entry && t.fuel == fuel, events_all[pt].tech)
        if !isempty(es)
            push!(vals, @sprintf("%.1f", events_all[pt].df.mac_private[es[1].idx]))
        else
            push!(vals, "n/a")
        end
    end
    println(@sprintf("  %-22s  %12s  %12s  %12s  %12s", fuel, vals...))
end

# ── 5c. Kink 원인 분류 ──
println("\n  [C] Kink Cause Classification")
println("  " * "-"^120)
println(@sprintf("  %-14s  %12s  %12s  %-60s",
    "Policy", "PolicyVal", "Abatement", "Cause(s)"))
println("  " * "-"^120)
for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
    ev = events_all[pt]
    df = ev.df
    emap = Dict{Int, Vector{String}}()
    for t in ev.tech
        push!(get!(emap, t.idx, String[]), "SAF $(t.type) $(t.fuel)")
    end
    for t in ev.duals
        push!(get!(emap, t.idx, String[]), "$(t.dual) $(t.type)")
    end
    for k in ev.kinks
        cs = get!(emap, k, String[])
        isempty(cs) && push!(cs, "MAC slope change only")
    end
    for idx in sort(collect(keys(emap)))
        println(@sprintf("  %-14s  %12.4f  %12.6f  %s",
            POLICY_META[pt].label, df.policy_value[idx], df.abatement[idx],
            join(emap[idx], " + ")))
    end
end

# ── 5d. Δemission step size at kinks ──
println("\n  [D] Δemission Step Size at Kink Points")
println("  " * "-"^100)
for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
    ev = events_all[pt]
    df = ev.df
    isempty(ev.kinks) && (println(@sprintf("  %-14s  no kinks", POLICY_META[pt].label)); continue)
    println(@sprintf("\n  %-14s:", POLICY_META[pt].label))
    for ki in ev.kinks
        Δ_at     = df.Δemission[ki]
        Δ_before = ki > 1 ? df.Δemission[ki-1] : NaN
        Δ_after  = ki < nrow(df) ? df.Δemission[ki+1] : NaN
        ratio    = isnan(Δ_before) || abs(Δ_before) < 1e-10 ? NaN : abs(Δ_at / Δ_before)
        println(@sprintf("    pval=%.4f  ab=%.6f  Δem_before=%.6f  Δem_at=%.6f  Δem_after=%.6f  ratio=%.2fx",
            df.policy_value[ki], df.abatement[ki], Δ_before, Δ_at, Δ_after, ratio))
    end
end

# =================================================================================
# 6. Kink 패턴 요약
# =================================================================================

println("\n\n" * "="^120)
println("  KINK PATTERN SUMMARY")
println("="^120)
println("""
  관찰된 kink 발생 원인 유형

  Type 1. 기술 도입 (Technology Entry)
     새 SAF 기술이 zero-profit condition을 충족하여 생산 시작.
     MAC 기울기가 완만해짐 (더 싼 감축 수단 추가).

  Type 2. 기술 퇴출 (Technology Exit)
     특정 SAF 기술의 경제성 상실로 생산 중단.

  Type 3. 제약 조건 전환 (Constraint Transition)
     Blend wall, 비대두 용량 제약, RFS 의무 등의 binding ↔ non-binding 전환.
     Dual variable 0 ↔ 양수 전환.

  Type 4. 복합 이벤트 (Compound Event)
     기술 도입과 제약 전환이 동시 발생.
     예: 비대두 HEFA SAF 도입 시 비대두 용량 제약이 즉시 binding.
""")

# =================================================================================
# 7. RFS / LCFS Zero Profit Condition 진단
# =================================================================================

begin
    _alpha_ns   = params.coeff.alpha[:saf_hefa_nonsoy]
    _alpha_atj  = params.coeff.alpha[:saf_atj_conv]
    _alpha_hefa_conv = params.coeff.alpha[:saf_hefa_conv]
    _alpha_hefa_cs   = params.coeff.alpha[:saf_hefa_cs]
    _nonsoy_fp  = params.coeff.nonsoy_feedstock_price
    _hefa_prem  = params.coeff.hefa_saf_premium
    _δ = params.coeff.delta
    _fc = params.supply.fuel
    _r_j = params.coeff.r[:jet_fuel]
    _β_saf = params.coeff.beta[(:saf, :jet_fuel)]

    _mc_hefa(sol) = begin
        tot = sol.q[:saf_hefa_conv] + sol.q[:saf_hefa_cs] + sol.q[:saf_hefa_nonsoy] +
              sol.q[:rd_soy] + sol.q[:rd_nonsoy]
        f = _fc[:saf_hefa_shared]
        f.c0 + f.c1 * tot + f.c2 * max(0.0, tot - f.v)^2
    end
    _mc_atj(sol) = begin
        tot = sol.q[:saf_atj_conv] + sol.q[:saf_atj_cs]
        f = _fc[:saf_atj_shared]
        f.c0 + f.c1 * tot + f.c2 * max(0.0, tot - f.v)^2
    end
    _ppu_saf(sol) = _r_j * _β_saf * sol.p_c[:avi]

    for pt in [:rfs, :lcfs]
        ev = events_all[pt]
        df_pt = ev.df
        m = POLICY_META[pt]

        entry_idxs = [t.idx for t in ev.tech if t.type == :entry]
        isempty(entry_idxs) && continue

        zpc_x = Float64[]
        zpc_nonsoy = Float64[]
        zpc_atj_conv = Float64[]
        zpc_hefa_conv = Float64[]
        zpc_hefa_cs = Float64[]
        zpc_atj_cs = Float64[]

        for r in eachrow(df_pt)
            scen = Symbol(r.scenario)
            sol = solutions[scen]
            ppu = _ppu_saf(sol)
            mch = _mc_hefa(sol)
            mca = _mc_atj(sol)
            p_corn_n = sol.p_f[:feedstock_corn_n]
            p_corn_cs = sol.p_f[:feedstock_corn_cs]
            p_soy_n = sol.p_f[:feedstock_soy_n]
            p_soy_cs = sol.p_f[:feedstock_soy_cs]

            config = EXTENDED_POLICY_MATRIX[scen]
            λ_ns = sol.duals.λ_nonsoy_capacity

            if pt == :rfs
                λ_ra = sol.duals.λ_rfs_avi
                pa_nonsoy    = λ_ns * _alpha_ns + (_δ[:saf_hefa_nonsoy] <= 0.5 * _δ[:jet_fuel] ? -λ_ra * 1.6 : 0.0)
                pa_atj_conv  = (_δ[:saf_atj_conv] <= 0.5 * _δ[:jet_fuel] ? -λ_ra * 1.6 : 0.0)
                pa_atj_cs    = (_δ[:saf_atj_cs] <= 0.5 * _δ[:jet_fuel] ? -λ_ra * 1.6 : 0.0)
                pa_hefa_conv = (_δ[:saf_hefa_conv] <= 0.5 * _δ[:jet_fuel] ? -λ_ra * 1.6 : 0.0)
                pa_hefa_cs   = (_δ[:saf_hefa_cs] <= 0.5 * _δ[:jet_fuel] ? -λ_ra * 1.6 : 0.0)
            elseif pt == :lcfs
                λ_lc = sol.duals.λ_lcfs
                σ_val = config.σ
                ci_jet = _δ[:jet_fuel]
                pa_nonsoy    = λ_ns * _alpha_ns + λ_lc * (-((1 - σ_val) * ci_jet - _δ[:saf_hefa_nonsoy]))
                pa_atj_conv  = λ_lc * (-((1 - σ_val) * ci_jet - _δ[:saf_atj_conv]))
                pa_atj_cs    = λ_lc * (-((1 - σ_val) * ci_jet - _δ[:saf_atj_cs]))
                pa_hefa_conv = λ_lc * (-((1 - σ_val) * ci_jet - _δ[:saf_hefa_conv]))
                pa_hefa_cs   = λ_lc * (-((1 - σ_val) * ci_jet - _δ[:saf_hefa_cs]))
            end

            push!(zpc_x, r.abatement)
            push!(zpc_nonsoy,    mch + _alpha_ns * _nonsoy_fp + pa_nonsoy - ppu)
            push!(zpc_atj_conv,  mca + (_alpha_atj - 0.159) * p_corn_n + pa_atj_conv - ppu)
            push!(zpc_atj_cs,    mca + params.coeff.alpha[:saf_atj_cs] * p_corn_cs + pa_atj_cs - 0.159 * p_corn_n - ppu)
            push!(zpc_hefa_conv, mch + _alpha_hefa_conv * p_soy_n + pa_hefa_conv - ppu)
            push!(zpc_hefa_cs,   mch + _alpha_hefa_cs * p_soy_cs + pa_hefa_cs - ppu)
        end

        kink_abs = [df_pt.abatement[ei] for ei in entry_idxs]
        kink_lbl = join([@sprintf("%.4f", df_pt.policy_value[ei]) for ei in entry_idxs], ", ")

        ckw = (xlabel="Abatement (bn tCO₂e)", ylabel="\$/gallon",
            guidefontsize=12, legendfontsize=10, grid=true,
            gridstyle=:dash, gridcolor=:lightgray, titlefontsize=14)

        p1 = plot(zpc_x, zpc_nonsoy; label="Non-soy HEFA", color=:purple,
            lw=2.5, marker=:circle, ms=1.5, title="ZPC Non-soy HEFA ($(m.label))", ckw...)
        hline!(p1, [0]; color=:black, lw=1.5, alpha=0.7, label="ZPC=0")
        for ab in kink_abs; vline!(p1, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label=""); end

        p2 = plot(zpc_x, zpc_atj_conv; label="Conv ATJ", color=:blue,
            lw=2.5, marker=:diamond, ms=1.5, title="ZPC Conv ATJ ($(m.label))", ckw...)
        hline!(p2, [0]; color=:black, lw=1.5, alpha=0.7, label="ZPC=0")
        for ab in kink_abs; vline!(p2, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label=""); end

        p3 = plot(zpc_x, zpc_hefa_conv; label="Conv HEFA", color=:green,
            lw=2.5, marker=:square, ms=1.5, title="ZPC Conv HEFA ($(m.label))", ckw...)
        hline!(p3, [0]; color=:black, lw=1.5, alpha=0.7, label="ZPC=0")
        for ab in kink_abs; vline!(p3, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label=""); end

        p4 = plot(zpc_x, zpc_atj_cs; label="CS ATJ", color=:red,
            lw=2.5, marker=:utriangle, ms=1.5, title="ZPC CS ATJ ($(m.label))", ckw...)
        hline!(p4, [0]; color=:black, lw=1.5, alpha=0.7, label="ZPC=0")
        for ab in kink_abs; vline!(p4, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label=""); end

        p5 = plot(zpc_x, zpc_hefa_cs; label="CS HEFA", color=:orange,
            lw=2.5, marker=:star5, ms=1.5, title="ZPC CS HEFA ($(m.label))", ckw...)
        hline!(p5, [0]; color=:black, lw=1.5, alpha=0.7, label="ZPC=0")
        for ab in kink_abs; vline!(p5, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label=""); end

        fig_zpc = plot(p1, p2, p3, p4, p5;
            layout=(5, 1), size=(1100, 1600),
            left_margin=14Plots.mm, right_margin=8Plots.mm,
            bottom_margin=6Plots.mm, top_margin=4Plots.mm,
            plot_title="$(m.label) Zero Profit Conditions (entries @ $(m.unit)=$(kink_lbl))",
            plot_titlefontsize=16)
        display(fig_zpc)
    end
end

# =================================================================================
# 8. 4정책 Kink 진단 패널 (MAC + SAF + Dual + Δem)
# =================================================================================

const SAF_PANEL_COLORS = Dict(:saf_hefa_conv => :green, :saf_hefa_cs => :orange,
    :saf_hefa_nonsoy => :purple, :saf_atj_conv => :blue, :saf_atj_cs => :red)

function plot_kink_panel(pt)
    ev = events_all[pt]
    df = ev.df
    m = POLICY_META[pt]
    x = df.abatement
    xl = extrema(x)

    entry_abs = [t.ab for t in ev.tech if t.type == :entry]
    entry_labels = [String(t.fuel) for t in ev.tech if t.type == :entry]
    mac_yl = pt == :taxcredit ? (-300, 2200) : (-300, 800)

    # p1: MAC
    p1 = plot(x, df.mac_private; label="Private", color=:steelblue, lw=2.5,
        ylabel="\$/tCO₂e", title="$(m.label) MAC", titlefontsize=14,
        legend=:topleft, legendfontsize=9, grid=true, xlims=xl, ylims=mac_yl,
        guidefontsize=11, tickfontsize=9, left_margin=12Plots.mm)
    plot!(p1, x, df.mac_social; label="Social", color=:red, lw=2.5)
    hline!(p1, [0]; color=:gray, linestyle=:dot, lw=1, label="")
    for (ab, lbl) in zip(entry_abs, entry_labels)
        vline!(p1, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label="")
        annotate!(p1, ab, mac_yl[2]*0.9, text(lbl, :red, :center, 7, rotation=90))
    end

    # p2: SAF stacked
    p2 = plot(ylabel="B gal", title="SAF Quantities", titlefontsize=14,
        legend=:topleft, legendfontsize=8, grid=true, xlims=xl,
        guidefontsize=11, tickfontsize=9, left_margin=12Plots.mm)
    plot!(p2, x, df.jet_fuel; fillrange=0, fillalpha=0.5, fillcolor=:lightgray,
        lw=1.5, color=:black, label="Jet Fuel")
    cum = copy(df.jet_fuel)
    for fuel in SAF_FUELS_ALL
        vals = df[!, fuel]
        any(vals .> 1e-6) || continue
        nc = cum .+ vals
        plot!(p2, x, nc; fillrange=cum, fillalpha=0.7, fillcolor=SAF_PANEL_COLORS[fuel],
            lw=1.5, color=SAF_PANEL_COLORS[fuel], label=String(fuel))
        cum = nc
    end
    for ab in entry_abs
        vline!(p2, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label="")
    end

    # p3: Duals
    p3 = plot(ylabel="Dual", title="Dual Variables", titlefontsize=14,
        legend=:topleft, legendfontsize=8, grid=true, xlims=xl,
        guidefontsize=11, tickfontsize=9, left_margin=12Plots.mm)
    plot!(p3, x, df.λ_rfs_avi; label="λ_rfs_avi", color=:steelblue, lw=2)
    plot!(p3, x, df.λ_lcfs; label="λ_lcfs", color=:green, lw=2)
    plot!(p3, x, df.λ_nonsoy_cap; label="λ_nonsoy", color=:purple, lw=2)
    plot!(p3, x, df.λ_rfs; label="λ_rfs", color=:orange, lw=2, linestyle=:dash)
    for ab in entry_abs
        vline!(p3, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label="")
    end

    # p4: Δemission
    p4 = plot(x, df.Δemission; label="Δem/step", color=:darkorange,
        lw=2, marker=:circle, ms=1.5,
        ylabel="bn tCO₂e", xlabel="Abatement (bn tCO₂e)",
        title="Δ Emission / Step", titlefontsize=14,
        legend=:topleft, legendfontsize=9, grid=true, xlims=xl,
        guidefontsize=11, tickfontsize=9, left_margin=12Plots.mm, bottom_margin=10Plots.mm)
    hline!(p4, [0]; color=:gray, linestyle=:dot, lw=1, label="")
    for ab in entry_abs
        vline!(p4, [ab]; color=:red, linestyle=:dash, lw=1.5, alpha=0.7, label="")
    end

    return plot(p1, p2, p3, p4; layout=(4, 1), size=(1200, 1400),
        left_margin=10Plots.mm, right_margin=8Plots.mm,
        plot_title="$(m.label) Kink Diagnosis", plot_titlefontsize=18)
end

for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
    display(plot_kink_panel(pt))
end

# =================================================================================
# 9. 기술 도입 순서 비교 (scatter, abatement 축)
# =================================================================================

begin
    _fuel_mk = Dict(:saf_hefa_conv => :circle, :saf_hefa_cs => :diamond,
        :saf_hefa_nonsoy => :utriangle, :saf_atj_conv => :square, :saf_atj_cs => :star5)
    _ypos = Dict(:carbontax => 4, :rfs => 3, :lcfs => 2, :taxcredit => 1)
    _labeled = Set{Symbol}()

    p_cmp = plot(ylabel="Policy", xlabel="Abatement (bn tCO₂e)",
        title="SAF Technology Entry Points Across Policies",
        titlefontsize=16, guidefontsize=13, tickfontsize=11,
        legend=:outerright, legendfontsize=10, size=(1400, 500),
        left_margin=15Plots.mm, bottom_margin=12Plots.mm, right_margin=25Plots.mm)

    for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
        for t in events_all[pt].tech
            t.type != :entry && continue
            lbl = t.fuel in _labeled ? "" : String(t.fuel)
            push!(_labeled, t.fuel)
            scatter!(p_cmp, [t.ab], [_ypos[pt]]; label=lbl,
                color=SAF_PANEL_COLORS[t.fuel], marker=_fuel_mk[t.fuel], ms=10)
            annotate!(p_cmp, t.ab, _ypos[pt]+0.25,
                text(@sprintf("%.3f", t.pval), SAF_PANEL_COLORS[t.fuel], :center, 8))
        end
    end
    yticks!(p_cmp, [1,2,3,4], ["Tax Credit","LCFS","RFS","Carbon Tax"])
    display(p_cmp)
end

println("\n✓ All kink diagnosis complete.")

sol_sq = solutions[:statusquo]
for fuel in SAF_FUELS_ALL
    @printf("  %-22s  %.6f\n", fuel, sol_sq.q[fuel])
end

sol_first = solutions[:lcfs_0]  # σ=0.0003에 해당하는 시나리오
for fuel in SAF_FUELS_ALL
    @printf("  %-22s  %.6f\n", fuel, sol_first.q[fuel])
end

# =================================================================================
# 10. Status Quo → 첫 grid point 기술 전환 확인
# diagnose_all_kinks.jl 맨 끝에 붙이기
# =================================================================================

println("\n" * "="^120)
println("  Status Quo SAF Quantities")
println("="^120)

sol_sq = solutions[:statusquo]
println(@sprintf("  %-22s  %12s", "Fuel", "Quantity"))
println("  " * "-"^36)
for fuel in SAF_FUELS_ALL
    println(@sprintf("  %-22s  %12.6f", fuel, sol_sq.q[fuel]))
end
println(@sprintf("  %-22s  %12.6f", :jet_fuel, sol_sq.q[:jet_fuel]))

# 각 정책의 첫 grid point와 비교
println("\n" * "="^120)
println("  Status Quo → First Grid Point Comparison (SAF entry at policy activation)")
println("="^120)

first_scenarios = Dict(
    :carbontax => :carbontax_0,
    :rfs       => :rfs_1,
    :lcfs      => :lcfs_0,
    :taxcredit => :taxcredit_5
)

# 정확한 첫 시나리오 찾기 (df_all에서)
for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
    df = df_all[pt]
    first_scen = Symbol(df.scenario[1])
    m = POLICY_META[pt]

    println("\n  $(m.label)  first grid point = $(first_scen)  ($(m.unit) = $(@sprintf("%.4f", df.policy_value[1])))")
    println(@sprintf("  %-22s  %12s  %12s  %10s", "Fuel", "StatusQuo", "FirstGrid", "Transition"))
    println("  " * "-"^62)

    sol_first = solutions[first_scen]
    for fuel in SAF_FUELS_ALL
        q_sq = sol_sq.q[fuel]
        q_first = sol_first.q[fuel]
        trans = ""
        if q_sq <= 1e-4 && q_first > 1e-4
            trans = "◀ ENTRY"
        elseif q_sq > 1e-4 && q_first <= 1e-4
            trans = "◀ EXIT"
        elseif abs(q_first - q_sq) > 1e-4
            trans = @sprintf("Δ = %+.4f", q_first - q_sq)
        end
        println(@sprintf("  %-22s  %12.6f  %12.6f  %10s",
            fuel, q_sq, q_first, trans))
    end
    println(@sprintf("  %-22s  %12.6f  %12.6f  %10s",
        :jet_fuel, sol_sq.q[:jet_fuel], sol_first.q[:jet_fuel],
        @sprintf("Δ = %+.4f", sol_first.q[:jet_fuel] - sol_sq.q[:jet_fuel])))
end

# =================================================================================
# 11. 수정된 기술 전환 감지 (status quo 포함)
# =================================================================================

println("\n" * "="^120)
println("  CORRECTED Technology Transitions (including StatusQuo → First Grid)")
println("="^120)

for pt in [:carbontax, :rfs, :lcfs, :taxcredit]
    df = df_all[pt]
    m = POLICY_META[pt]
    first_scen = Symbol(df.scenario[1])
    sol_first = solutions[first_scen]

    println("\n  $(m.label)")
    println(@sprintf("  %-22s  %-7s  %12s  %12s  %12s",
        "Fuel", "Type", "PolicyVal", "Abatement", "Quantity"))
    println("  " * "-"^70)

    # status quo → first grid point 전환
    for fuel in SAF_FUELS_ALL
        q_sq = sol_sq.q[fuel]
        q_first = sol_first.q[fuel]
        if q_sq <= 1e-4 && q_first > 1e-4
            println(@sprintf("  %-22s  %-7s  %12.4f  %12.6f  %12.6f  (from SQ)",
                fuel, :entry, df.policy_value[1], df.abatement[1], q_first))
        end
    end

    # 이후 grid 간 전환 (기존 detect_tech_transitions 결과)
    for t in events_all[pt].tech
        println(@sprintf("  %-22s  %-7s  %12.4f  %12.6f  %12.6f",
            t.fuel, t.type, t.pval, t.ab, t.qty))
    end
end

println("\n✓ Status quo comparison complete.")

# =================================================================================
# 12. Non-soy 진입 / 용량 포화 전후 연료 수량 변화 확인
# diagnose_all_kinks.jl 끝에 붙이기
# =================================================================================

println("\n" * "█"^120)
println("  Fuel Quantity Changes at Non-soy Entry & Capacity Saturation")
println("█"^120)

ALL_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy,
             :ethanol, :rd_soy, :rd_nonsoy, :biodiesel_soy, :biodiesel_nonsoy, :gasoline, :diesel]

for pt in [:carbontax, :rfs, :lcfs]
    ev = events_all[pt]
    df = ev.df
    m = POLICY_META[pt]

    # non-soy entry 지점
    nonsoy_entry = filter(t -> t.fuel == :saf_hefa_nonsoy && t.type == :entry, ev.tech)
    # nonsoy_cap binding 지점
    cap_binding = filter(t -> t.dual == :λ_nonsoy_cap && t.type == :binding, ev.duals)

    println("\n" * "▓"^120)
    println("  $(m.label)")
    println("▓"^120)

    # ── Non-soy entry ──
    if !isempty(nonsoy_entry)
        ei = nonsoy_entry[1].idx
        r1 = max(1, ei - 2)
        r2 = min(nrow(df), ei + 2)

        println("\n  [Non-soy Entry]  idx=$(ei)  $(m.unit)=$(@sprintf("%.4f", df.policy_value[ei]))")
        println(@sprintf("\n  %10s │ %10s %10s %10s %10s %10s %10s │ %8s %8s │ %10s",
            m.unit, "jet_fuel", "atj_conv", "atj_cs", "hefa_conv", "hefa_cs", "hefa_ns",
            "p_avi", "x_avi", "Δemission"))
        println("  " * "-"^120)

        for i in r1:r2
            scen = Symbol(df.scenario[i])
            sol = solutions[scen]
            mk = i == ei ? " ◀" : ""
            println(@sprintf("  %10.4f │ %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f │ %8.5f %8.2f │ %10.6f%s",
                df.policy_value[i],
                sol.q[:jet_fuel], sol.q[:saf_atj_conv], sol.q[:saf_atj_cs],
                sol.q[:saf_hefa_conv], sol.q[:saf_hefa_cs], sol.q[:saf_hefa_nonsoy],
                sol.p_c[:avi], sol.x[:avi],
                df.Δemission[i], mk))
        end

        # 전후 차이
        if ei > 1
            before = Symbol(df.scenario[ei-1])
            after = Symbol(df.scenario[ei])
            sb = solutions[before]
            sa = solutions[after]
            println("\n  Δ(entry - 1step before)")
            println(@sprintf("    jet_fuel   %+.4f", sa.q[:jet_fuel] - sb.q[:jet_fuel]))
            println(@sprintf("    atj_conv   %+.4f", sa.q[:saf_atj_conv] - sb.q[:saf_atj_conv]))
            println(@sprintf("    atj_cs     %+.4f", sa.q[:saf_atj_cs] - sb.q[:saf_atj_cs]))
            println(@sprintf("    hefa_conv  %+.4f", sa.q[:saf_hefa_conv] - sb.q[:saf_hefa_conv]))
            println(@sprintf("    hefa_cs    %+.4f", sa.q[:saf_hefa_cs] - sb.q[:saf_hefa_cs]))
            println(@sprintf("    hefa_ns    %+.4f", sa.q[:saf_hefa_nonsoy] - sb.q[:saf_hefa_nonsoy]))
            println(@sprintf("    p_avi      %+.6f", sa.p_c[:avi] - sb.p_c[:avi]))
            println(@sprintf("    x_avi      %+.4f", sa.x[:avi] - sb.x[:avi]))
            println(@sprintf("    Δemission  before=%.6f  after=%.6f  change=%+.6f",
                df.Δemission[ei-1], df.Δemission[ei], df.Δemission[ei] - df.Δemission[ei-1]))
        end
    else
        println("\n  [Non-soy Entry]  not detected in grid (may enter at first grid point)")
    end

    # ── Capacity binding ──
    if !isempty(cap_binding)
        ci = cap_binding[1].idx
        r1 = max(1, ci - 2)
        r2 = min(nrow(df), ci + 2)

        println("\n  [Capacity Binding]  idx=$(ci)  $(m.unit)=$(@sprintf("%.4f", df.policy_value[ci]))")
        println(@sprintf("\n  %10s │ %10s %10s %10s %10s %10s %10s │ %8s %8s │ %10s %10s",
            m.unit, "jet_fuel", "atj_conv", "atj_cs", "hefa_conv", "hefa_cs", "hefa_ns",
            "p_avi", "x_avi", "Δemission", "λ_ns_cap"))
        println("  " * "-"^130)

        for i in r1:r2
            scen = Symbol(df.scenario[i])
            sol = solutions[scen]
            mk = i == ci ? " ◀" : ""
            println(@sprintf("  %10.4f │ %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f │ %8.5f %8.2f │ %10.6f %10.4f%s",
                df.policy_value[i],
                sol.q[:jet_fuel], sol.q[:saf_atj_conv], sol.q[:saf_atj_cs],
                sol.q[:saf_hefa_conv], sol.q[:saf_hefa_cs], sol.q[:saf_hefa_nonsoy],
                sol.p_c[:avi], sol.x[:avi],
                df.Δemission[i], df.λ_nonsoy_cap[i], mk))
        end

        # 전후 차이
        if ci > 1
            before = Symbol(df.scenario[ci-1])
            after = Symbol(df.scenario[ci])
            sb = solutions[before]
            sa = solutions[after]
            println("\n  Δ(binding - 1step before)")
            println(@sprintf("    jet_fuel   %+.4f", sa.q[:jet_fuel] - sb.q[:jet_fuel]))
            println(@sprintf("    atj_conv   %+.4f", sa.q[:saf_atj_conv] - sb.q[:saf_atj_conv]))
            println(@sprintf("    atj_cs     %+.4f", sa.q[:saf_atj_cs] - sb.q[:saf_atj_cs]))
            println(@sprintf("    hefa_conv  %+.4f", sa.q[:saf_hefa_conv] - sb.q[:saf_hefa_conv]))
            println(@sprintf("    hefa_cs    %+.4f", sa.q[:saf_hefa_cs] - sb.q[:saf_hefa_cs]))
            println(@sprintf("    hefa_ns    %+.4f", sa.q[:saf_hefa_nonsoy] - sb.q[:saf_hefa_nonsoy]))
            println(@sprintf("    p_avi      %+.6f", sa.p_c[:avi] - sb.p_c[:avi]))
            println(@sprintf("    x_avi      %+.4f", sa.x[:avi] - sb.x[:avi]))
            println(@sprintf("    λ_ns_cap   before=%.4f  after=%.4f",
                sb.duals.λ_nonsoy_capacity, sa.duals.λ_nonsoy_capacity))
            println(@sprintf("    Δemission  before=%.6f  after=%.6f  change=%+.6f",
                df.Δemission[ci-1], df.Δemission[ci], df.Δemission[ci] - df.Δemission[ci-1]))
        end
    else
        println("\n  [Capacity Binding]  not detected")
    end

    # ── 연속 구간 요약: non-soy entry ~ capacity binding 전후 5step씩 ──
    if !isempty(nonsoy_entry) && !isempty(cap_binding)
        ei = nonsoy_entry[1].idx
        ci = cap_binding[1].idx

        println("\n  [Summary: 5 steps before entry → entry → ... → cap binding → 5 steps after]")
        println(@sprintf("  %10s │ %10s %10s %10s │ %8s %8s │ %10s │ %8s",
            m.unit, "jet_fuel", "atj_cs", "hefa_ns",
            "p_avi", "x_avi", "Δemission", "λ_ns_cap"))
        println("  " * "-"^100)

        # 출력할 인덱스 선택
        show_idxs = sort(unique(vcat(
            collect(max(1, ei-3):min(nrow(df), ei+2)),
            collect(max(1, ci-2):min(nrow(df), ci+3))
        )))

        for i in show_idxs
            scen = Symbol(df.scenario[i])
            sol = solutions[scen]
            mk = i == ei ? " ◀ entry" : i == ci ? " ◀ cap" : ""
            println(@sprintf("  %10.4f │ %10.4f %10.4f %10.4f │ %8.5f %8.2f │ %10.6f │ %8.4f%s",
                df.policy_value[i],
                sol.q[:jet_fuel], sol.q[:saf_atj_cs], sol.q[:saf_hefa_nonsoy],
                sol.p_c[:avi], sol.x[:avi],
                df.Δemission[i], df.λ_nonsoy_cap[i], mk))
        end
    end
end

println("\n✓ Fuel quantity verification complete.")