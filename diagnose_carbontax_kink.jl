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