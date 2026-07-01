# equivalent_emissions_analysis.jl

# equivalent_emissions_welfare.jl
# run_model.jl이 저장한 결과만 읽어서 welfare 분석과 비교 표를 생성

cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "SAFModel.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
using .SAFModel
using .SAFAnalysis
import .SAFModel: params
using JLD2
using DataFrames
using Printf

const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"

const TARGET_SAF_VALUES = [3.0, 6.0]
const POLICY_TYPES = [:carbontax, :rfs, :lcfs, :taxcredit]

# =================================================================================
# Status Quo 로드 및 enrich (한 번만)
# =================================================================================

@load joinpath(OUTPUT_DIR, "results_base.jld2") results_base policy_configs_base

sq_sol = results_base[:statusquo]
status_quo = merge(sq_sol, (
    implicit_taxes=calculate_implicit_taxes(sq_sol, params, policy_configs_base.statusquo),
    emissions=calculate_emissions_detail(sq_sol, params)
))

# =================================================================================
# 각 Target 처리
# =================================================================================

for target_saf in TARGET_SAF_VALUES
    println("\n" * "="^130)
    println("PROCESSING TARGET SAF = $(target_saf)B gallons")
    println("="^130)

    suffix = target_saf == 3.0 ? "" : "_$(Int(target_saf))"
    infile = joinpath(OUTPUT_DIR, "results_target$(suffix).jld2")

    # run_model.jl 결과 로드
    @load infile results_target policy_configs_target equivalent_policies target_emissions

    # ── solution enrich (implicit_taxes, emissions 병합) ──
    implicit_taxes_all = calculate_all_implicit_taxes(results_target, params, policy_configs_target)
    enriched = Dict(
        pt => merge(sol, (
            implicit_taxes=implicit_taxes_all[pt],
            emissions=calculate_emissions_detail(sol, params)
        ))
        for (pt, sol) in results_target
    )

    # RFS 실제 SAF, target 배출량 출력
    rfs_sol = enriched[:rfs]
    saf_goods = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    println("RFS actual SAF = ", round(sum(rfs_sol.q[g] for g in saf_goods), digits=4), " B gal")
    println("Target emissions = ", round(target_emissions, digits=4), " B ton CO2e")

    # =================================================================================
    # Welfare 계산
    # =================================================================================

    cs_changes = calculate_cs_changes(enriched, status_quo, params; scenarios=POLICY_TYPES)
    ps_land = calculate_ps_land_changes(enriched, status_quo, params; scenarios=POLICY_TYPES)
    gr_changes = calculate_gr_changes(enriched; scenarios=POLICY_TYPES)
    env_benefits = calculate_environmental_benefit(enriched, status_quo, SCC; scenarios=POLICY_TYPES)
    welfare_sum = calculate_total_welfare(cs_changes, ps_land, gr_changes, env_benefits; scenarios=POLICY_TYPES)
    aac_results = calculate_average_abatement_cost(welfare_sum, enriched, status_quo; scenarios=POLICY_TYPES)

    display_cs_changes(cs_changes; scenarios=POLICY_TYPES,
        title="EQUIVALENT EMISSIONS (Target SAF $(target_saf)B): CONSUMER SURPLUS CHANGES (billion \$)")
    display_ps_land_changes(ps_land; scenarios=POLICY_TYPES,
        title="EQUIVALENT EMISSIONS (Target SAF $(target_saf)B): LAND PS CHANGES (billion \$)")
    display_gr_changes(gr_changes; scenarios=POLICY_TYPES,
        title="EQUIVALENT EMISSIONS (Target SAF $(target_saf)B): GOVERNMENT REVENUE (billion \$)")
    display_environmental_benefits(env_benefits, SCC; scenarios=POLICY_TYPES,
        title="EQUIVALENT EMISSIONS (Target SAF $(target_saf)B): ENV BENEFITS")
    display_welfare_summary(welfare_sum; scenarios=POLICY_TYPES,
        title="EQUIVALENT EMISSIONS (Target SAF $(target_saf)B): WELFARE SUMMARY (billion \$)")
    display_aac_analysis(aac_results; scenarios=POLICY_TYPES,
        title="EQUIVALENT EMISSIONS (Target SAF $(target_saf)B): AVERAGE ABATEMENT COST")

    # =================================================================================
    # 비교 표 생성
    # =================================================================================

    # 정책 파라미터
    param_labels = Dict(
        :carbontax => "Carbon Tax (\$/ton CO2e)",
        :rfs => "RFS Mandate Share",
        :lcfs => "LCFS CI Reduction (σ)",
        :taxcredit => "Tax Credit (\$/gal)"
    )
    policy_params_df = DataFrame(Policy=String[], Parameter=String[], Value=Float64[],
        Actual_Emissions_Bton=Float64[], Diff_from_Target_Bton=Float64[])
    for pt in POLICY_TYPES
        r = equivalent_policies[pt]
        pval = pt == :carbontax ? r.config.t :
               pt == :rfs ? r.config.θ_avi :
               pt == :lcfs ? r.config.σ : r.config.p
        actual_em = enriched[pt].emissions.total
        push!(policy_params_df, (String(pt), param_labels[pt], pval,
            actual_em, actual_em - target_emissions))
    end
    println("\n--- Policy Parameters ---")
    show(policy_params_df, allrows=true)

    # SAF 생산
    saf_production_df = DataFrame(Policy=String[], saf_atj_conv=Float64[], saf_atj_cs=Float64[],
        saf_hefa_conv=Float64[], saf_hefa_cs=Float64[], saf_hefa_nonsoy=Float64[], Total_SAF=Float64[])
    for pt in POLICY_TYPES
        sol = enriched[pt]
        push!(saf_production_df, (String(pt),
            sol.q[:saf_atj_conv], sol.q[:saf_atj_cs], sol.q[:saf_hefa_conv],
            sol.q[:saf_hefa_cs], sol.q[:saf_hefa_nonsoy],
            sum(sol.q[g] for g in saf_goods)))
    end
    println("\n--- SAF Production (billion gallons) ---")
    show(saf_production_df, allrows=true)

    # 배출량
    emissions_df = DataFrame(Policy=String[], Aviation_Bton=Float64[], Road_Bton=Float64[],
        Food_Bton=Float64[], Total_Bton=Float64[], Diff_from_Target_Bton=Float64[])
    for pt in POLICY_TYPES
        em = enriched[pt].emissions
        push!(emissions_df, (String(pt), em.aviation, em.road, em.food, em.total,
            em.total - target_emissions))
    end
    println("\n--- Emissions (billion ton CO2e) ---")
    show(emissions_df, allrows=true)

    # 토지 이용
    land_df = DataFrame(Policy=String[], Conv_Land=Float64[], CS_Land=Float64[],
        Total_Land=Float64[], Land_Rent=Float64[])
    for pt in POLICY_TYPES
        sol = enriched[pt]
        push!(land_df, (String(pt), sol.l_n*1000, sol.l_cs*1000,
            (sol.l_n + sol.l_cs)*1000, sol.duals.r_land))
    end
    println("\n--- Land Use (million acres) ---")
    show(land_df, allrows=true)

    # 후생 비교
    welfare_df = DataFrame(Policy=String[], CS_Change=Float64[], PS_Land_Change=Float64[],
        Gov_Revenue=Float64[], Env_Benefit=Float64[], Social_Welfare=Float64[])
    for pt in POLICY_TYPES
        w = welfare_sum[pt]
        push!(welfare_df, (String(pt), w.cs_change, w.ps_land_change,
            w.gr_change, w.env_benefit, w.social_welfare))
    end
    println("\n--- Welfare Comparison (billion \$) ---")
    show(welfare_df, allrows=true)

    cs_breakdown_df = make_cs_change_table(cs_changes; scenarios=POLICY_TYPES)
    println("\n--- Consumer Surplus Breakdown by Sector (billion \$) ---")
    show(cs_breakdown_df, allrows=true)

    # =================================================================================
    # 저장
    # =================================================================================

    @save joinpath(OUTPUT_DIR, "results_equivalent_emissions_complete$(suffix).jld2") cs_changes ps_land gr_changes env_benefits welfare_sum aac_results target_emissions policy_configs_target equivalent_policies SCC

    comparison_tables = Dict(
        :policy_params => policy_params_df,
        :saf_production => saf_production_df,
        :emissions => emissions_df,
        :welfare => welfare_df,
        :welfare_cs_breakdown => cs_breakdown_df,
        :land_use => land_df
    )
    @save joinpath(OUTPUT_DIR, "results_equivalent_emissions_tables$(suffix).jld2") comparison_tables

    println("\n✓ Saved results_equivalent_emissions_complete$(suffix).jld2")
    println("✓ Saved results_equivalent_emissions_tables$(suffix).jld2")
end


