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
println(@sprintf("%-30s %8s %10s %10s %10s %10s %12s %12s",
    "Scenario", "t(\$/t)", "ΔCS", "ΔPS_land", "ΔGovRev", "ΔEnvBen", "Δpriv_surp", "Δsoc_welf"))
println("-"^110)

cs_ch = results_extended_analysis.cs_changes
ps_ch = results_extended_analysis.ps_land_changes
gr_ch = results_extended_analysis.gr_changes
env_ch = results_extended_analysis.env_benefits

for r in eachrow(df_detail)
    scen = Symbol(r.scenario)
    if !haskey(welfare_summary, scen)
        continue
    end
    w = welfare_summary[scen]
    println(@sprintf("%-30s %8.0f %10.4f %10.4f %10.4f %10.4f %12.4f %12.4f",
        r.scenario, r.t,
        w.cs_change, w.ps_land_change, w.gr_change, w.env_benefit,
        w.private_surplus, w.social_welfare))
end

println("\n✓ Diagnosis complete.")