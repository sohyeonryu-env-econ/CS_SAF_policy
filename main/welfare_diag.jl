# ---------------------------------------------------------------------------
# Diagnostic: why is the LCFS more costly than the RFS at the same abatement?
#
# results_extended_analysis 를 이용해 RFS와 LCFS를 "감축량(abatement)"축에
# 정렬한 뒤, 지정한 감축량 지점들에서 welfare를 구성요소별로 분해해 비교한다.
#
# 핵심 질문: 같은 감축량에서 LCFS의 social welfare 가 RFS보다 낮은 것이
#   (a) 생산자잉여(PS)를 덜 만들기 때문인지,
#   (b) 소비자잉여(CS) 손실이 더 크기 때문인지,
#   (c) 단위 stringency당 감축량(분모) 차이 때문인지
# 를 숫자로 확정한다.
# ---------------------------------------------------------------------------

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
using DataFrames, Printf

const SECTORS_D = [:avi, :gas, :die, :corn, :soyoil, :soymeal]

"""
    policy_abatement_curve(rea, policy_type)

한 정책의 모든 stringency 점을 (감축량, welfare 분해, SAF 구성) 튜플의 벡터로 반환.
감축량 오름차순 정렬. 감축량 단위는 million ton CO2e.
"""
function policy_abatement_curve(rea, policy_type)
    sols = rea.solutions
    ws = rea.welfare_summary
    sq_em = sols[:statusquo].emissions.total

    group = rea.scenario_groups[policy_type]
    rows = NamedTuple[]
    for s in group
        (haskey(ws, s) && !isnothing(sols[s])) || continue
        ab = (sq_em - sols[s].emissions.total) * 1000  # M ton
        w = ws[s]
        push!(rows, (
            scenario=s,
            abatement=ab,
            cs_total=w.cs_change,
            cs_by_sector=w.cs_by_sector,
            ps=w.ps_land_change,
            gr=w.gr_change,
            env=w.env_benefit,
            private=w.private_surplus,
            social=w.social_welfare,
            # SAF 구성
            saf_atj_cs=sols[s].q[:saf_atj_cs],
            saf_hefa_cs=sols[s].q[:saf_hefa_cs],
            saf_nonsoy=sols[s].q[:saf_hefa_nonsoy],
            land_rent=sols[s].duals.r_land,
        ))
    end
    sort!(rows, by=r -> r.abatement)
    return rows
end

"""
    interp_at(curve, target_ab, field)

감축량 곡선에서 target_ab 에 해당하는 field 값을 선형보간해서 반환.
field 가 :cs_by_sector 이면 부문별 Dict 를 보간한 Dict 를 반환.
"""
function interp_at(curve, target_ab, field)
    abs_ = [r.abatement for r in curve]
    (target_ab < minimum(abs_) || target_ab > maximum(abs_)) && return nothing

    # 바로 아래/위 점 찾기
    i = findlast(a -> a <= target_ab, abs_)
    i === nothing && return nothing
    i == length(abs_) && (i -= 1)
    a0, a1 = abs_[i], abs_[i+1]
    frac = a1 ≈ a0 ? 0.0 : (target_ab - a0) / (a1 - a0)

    if field == :cs_by_sector
        d0, d1 = curve[i].cs_by_sector, curve[i+1].cs_by_sector
        return Dict(s => d0[s] + frac * (d1[s] - d0[s]) for s in SECTORS_D)
    else
        v0 = getfield(curve[i], field)
        v1 = getfield(curve[i+1], field)
        return v0 + frac * (v1 - v0)
    end
end

"""
    compare_rfs_lcfs(rea; abatement_points=[35.0, 40.0, 44.0, 50.0])

지정한 감축량 지점들에서 RFS와 LCFS의 welfare 분해를 나란히 비교하는 표를 만든다.
각 지점에서 (RFS, LCFS, 차이=LCFS-RFS)를 보여준다.
"""
function compare_rfs_lcfs(rea; abatement_points=[35.0, 40.0, 44.0, 50.0])
    rfs = policy_abatement_curve(rea, :rfs)
    lcfs = policy_abatement_curve(rea, :lcfs)

    println("RFS abatement range:  $(round(minimum(r.abatement for r in rfs),digits=1)) ~ $(round(maximum(r.abatement for r in rfs),digits=1)) M ton")
    println("LCFS abatement range: $(round(minimum(r.abatement for r in lcfs),digits=1)) ~ $(round(maximum(r.abatement for r in lcfs),digits=1)) M ton")
    println()

    metrics = [
        ("CS: Aviation", :avi), ("CS: Gasoline", :gas), ("CS: Diesel", :die),
        ("CS: Corn", :corn), ("CS: Soyoil", :soyoil), ("CS: Soymeal", :soymeal),
    ]
    scalar_metrics = [
        ("Total CS", :cs_total), ("Producer surplus", :ps), ("Govt revenue", :gr),
        ("Env benefit", :env), ("Private surplus", :private), ("Social welfare", :social),
        ("Land rent", :land_rent),
        ("CS ATJ (B gal)", :saf_atj_cs), ("CS HEFA (B gal)", :saf_hefa_cs),
        ("Non-soy HEFA (B gal)", :saf_nonsoy),
    ]

    results = DataFrame[]
    for ab in abatement_points
        rfs_cs = interp_at(rfs, ab, :cs_by_sector)
        lcfs_cs = interp_at(lcfs, ab, :cs_by_sector)
        (isnothing(rfs_cs) || isnothing(lcfs_cs)) && (println(">> abatement $ab M ton: out of range for one policy, skipped\n"); continue)

        df = DataFrame(Metric=String[], RFS=Float64[], LCFS=Float64[], Diff_LCFS_minus_RFS=Float64[])
        for (label, s) in metrics
            rv, lv = rfs_cs[s], lcfs_cs[s]
            push!(df, (label, rv, lv, lv - rv))
        end
        for (label, f) in scalar_metrics
            rv = interp_at(rfs, ab, f)
            lv = interp_at(lcfs, ab, f)
            push!(df, (label, rv, lv, lv - rv))
        end

        println("="^70)
        println("Abatement = $(ab) M ton CO2e")
        println("="^70)
        show(df, allrows=true, allcols=true)
        println("\n")
        push!(results, df)
    end
    return results
end

# 사용:
@load joinpath(OUTPUT_DIR, "results_extended.jld2") results_extended_analysis
dfs = compare_rfs_lcfs(results_extended_analysis; abatement_points=[35.0, 40.0, 44.0, 50.0])
#
# 해석 가이드:
#  - "Social welfare" 행의 Diff 가 음수면 그 감축량에서 LCFS가 RFS보다 사회적으로 낮음.
#  - 그 음수가 어디서 오는지 위 행들에서 확인:
#      * "Producer surplus" Diff 가 크게 음수 → PS 열세가 원인 (land rent 기반 지주잉여)
#      * "Total CS" Diff 가 양수인데도 Social 이 음수 → 소비자는 LCFS가 유리, PS가 뒤집음
#      * 부문별 CS 행에서 diesel/soyoil 은 LCFS가 더 손실(-), corn/gasoline 은 RFS가 더 손실
#  - "Non-soy HEFA" 와 "CS ATJ" 구성 차이가 PS 격차의 근원인지 확인:
#      LCFS 가 non-soy(무토지) 를 더 쓰고 CS ATJ(유토지) 를 덜 쓰면 land rent·PS 가 작아짐.