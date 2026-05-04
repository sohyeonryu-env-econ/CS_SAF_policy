# sigma_trajectory.jl
# σ_cet별로 r_corn, r_soy, l_corn, l_soy가 어떻게 변하는지 trajectory 분석
cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "SAFModel.jl"))
include(joinpath(@__DIR__, "analysis.jl"))
using .SAFModel
import .SAFModel: params, build_unified_model, extract_solution, is_solved_and_feasible
using JuMP, PATHSolver, DataFrames, Printf, Plots

const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"

# =================================================================================
# 0. 설정
# =================================================================================

# Policy 고정 (RFS θ_avi = 0.15)
policy_rfs = (t=0.0, θ_avi=0.15, σ=0.0, p=0.0, carbon_tax_scope=:aviation)

# σ_cet 범위: 0부터 1.0까지 촘촘하게
sigma_values = vcat(
    0.0,
    0.001, 0.005,
    0.01:0.01:0.09...,
    0.1:0.05:1.0...
)

# =================================================================================
# 1. σ별 실행 및 결과 수집
# =================================================================================

results = []

for σ_val in sigma_values
    @printf("\n-- σ_cet = %.4f --\n", σ_val)

    new_coeff = merge(params.coeff, (σ_cet=Float64(σ_val),))
    new_params = merge(params, (coeff=new_coeff,))

    model = build_unified_model(new_params, policy_rfs)
    optimize!(model)
    status = termination_status(model)

    if is_solved_and_feasible(model)
        println("  ✓ solved")
        sol = extract_solution(model, :rfs)

        # r_corn, r_soy는 CET 모델에서만 존재
        r_corn_val = σ_val > 0 ? value(model[:r_corn]) : NaN
        r_soy_val = σ_val > 0 ? value(model[:r_soy]) : NaN
        r_land_val = sol.duals.r_land

        # l_corn, l_soy: CET에서는 expression으로 존재
        l_corn_n_val = σ_val > 0 ? value(model[:l_corn_n]) : sol.l_n * 0.54
        l_corn_cs_val = σ_val > 0 ? value(model[:l_corn_cs]) : sol.l_cs * 0.54
        l_soy_n_val = σ_val > 0 ? value(model[:l_soy_n]) : sol.l_n * 0.46
        l_soy_cs_val = σ_val > 0 ? value(model[:l_soy_cs]) : sol.l_cs * 0.46

        l_corn_total = l_corn_n_val + l_corn_cs_val
        l_soy_total = l_soy_n_val + l_soy_cs_val
        l_total = sol.l_n + sol.l_cs

        # 실제 share 계산
        corn_share = l_total > 0 ? l_corn_total / l_total : NaN
        soy_share = l_total > 0 ? l_soy_total / l_total : NaN

        push!(results, (
            sigma=Float64(σ_val),
            solved=true,
            status=string(status),
            # land rent
            r_corn=r_corn_val,
            r_soy=r_soy_val,
            r_land=r_land_val,
            r_ratio=(σ_val > 0 && r_soy_val > 0) ? r_corn_val / r_soy_val : NaN,
            # land quantities
            l_n=sol.l_n,
            l_cs=sol.l_cs,
            l_total=l_total,
            l_corn=l_corn_total,
            l_soy=l_soy_total,
            corn_share=corn_share,
            soy_share=soy_share,
            # feedstock prices
            p_corn_n=sol.p_f[:feedstock_corn_n],
            p_soy_n=sol.p_f[:feedstock_soy_n],
            # SAF
            total_saf=sum(sol.q[g] for g in [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]),
        ))
    else
        println("  ✗ failed: $status")
        push!(results, (
            sigma=Float64(σ_val), solved=false, status=string(status),
            r_corn=NaN, r_soy=NaN, r_land=NaN, r_ratio=NaN,
            l_n=NaN, l_cs=NaN, l_total=NaN, l_corn=NaN, l_soy=NaN,
            corn_share=NaN, soy_share=NaN,
            p_corn_n=NaN, p_soy_n=NaN, total_saf=NaN,
        ))
    end
end

df = DataFrame(results)
df_ok = filter(row -> row.solved, df)
df_fail = filter(row -> !row.solved, df)

println("\n===== Trajectory Summary =====")
println(df[:, [:sigma, :solved, :r_corn, :r_soy, :r_land, :r_ratio, :corn_share, :l_total]])

println("\n===== Failed σ values =====")
println(df_fail[:, [:sigma, :status]])

# =================================================================================
# 2. 플롯
# =================================================================================

gr()


# ── Plot 1: r_corn, r_soy, r_land by σ ────────────────────────────────────────
p1 = plot(
    title="Land rent by σ_cet  (RFS θ_avi=0.15)",
    xlabel="σ_cet", ylabel="Land rent (\$/acre)",
    size=(850, 500), background_color=:white, framestyle=:box,
    gridcolor=:lightgrey, legend=:topleft,
    left_margin=10Plots.mm, bottom_margin=8Plots.mm
)
plot!(p1, df_ok.sigma, df_ok.r_corn,
    label="r_corn", color=:goldenrod, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
plot!(p1, df_ok.sigma, df_ok.r_soy,
    label="r_soy", color=:forestgreen, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
plot!(p1, df_ok.sigma, df_ok.r_land,
    label="r_land (CET index)", color=:navy, marker=:diamond, markersize=5,
    linewidth=2.0, linestyle=:dash, markerstrokewidth=0)
# 실패 구간 표시
for row in eachrow(df_fail)
    vline!(p1, [row.sigma], color=:red, alpha=0.3, linewidth=1, label="")
end
if !isempty(df_fail)
    vline!(p1, [df_fail.sigma[1]], color=:red, alpha=0.3, linewidth=1, label="Failed σ")
end

# ── Plot 2: r_corn/r_soy ratio ────────────────────────────────────────────────
p2 = plot(
    title="r_corn / r_soy ratio by σ_cet",
    xlabel="σ_cet", ylabel="r_corn / r_soy",
    size=(850, 400), background_color=:white, framestyle=:box,
    gridcolor=:lightgrey, legend=:topright,
    left_margin=10Plots.mm, bottom_margin=8Plots.mm
)
hline!(p2, [1.0], color=:gray, linestyle=:dash, linewidth=1, label="ratio=1 (equal rents)")
plot!(p2, df_ok.sigma, df_ok.r_ratio,
    label="r_corn/r_soy", color=:crimson, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
for row in eachrow(df_fail)
    vline!(p2, [row.sigma], color=:red, alpha=0.3, linewidth=1, label="")
end

# ── Plot 3: corn/soy land share by σ ─────────────────────────────────────────
p3 = plot(
    title="Corn/Soy land share by σ_cet",
    xlabel="σ_cet", ylabel="Land share",
    size=(850, 400), background_color=:white, framestyle=:box,
    gridcolor=:lightgrey, legend=:right,
    left_margin=10Plots.mm, bottom_margin=8Plots.mm,
    ylims=(0.3, 0.8)
)
hline!(p3, [0.54], color=:goldenrod, linestyle=:dash, linewidth=1, label="ω=0.54 (Leontief)")
hline!(p3, [0.46], color=:forestgreen, linestyle=:dash, linewidth=1, label="1-ω=0.46 (Leontief)")
plot!(p3, df_ok.sigma, df_ok.corn_share,
    label="Corn share (CET)", color=:goldenrod, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
plot!(p3, df_ok.sigma, df_ok.soy_share,
    label="Soy share (CET)", color=:forestgreen, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
for row in eachrow(df_fail)
    vline!(p3, [row.sigma], color=:red, alpha=0.3, linewidth=1, label="")
end

# ── Plot 4: total land & l_n, l_cs by σ ─────────────────────────────────────
p4 = plot(
    title="Land allocation by σ_cet",
    xlabel="σ_cet", ylabel="Land (billion acres)",
    size=(850, 400), background_color=:white, framestyle=:box,
    gridcolor=:lightgrey, legend=:right,
    left_margin=10Plots.mm, bottom_margin=8Plots.mm
)
plot!(p4, df_ok.sigma, df_ok.l_total,
    label="Total land", color=:black, marker=:circle, markersize=5,
    linewidth=2.0, markerstrokewidth=0)
plot!(p4, df_ok.sigma, df_ok.l_n,
    label="Conv (l_n)", color=:steelblue, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
plot!(p4, df_ok.sigma, df_ok.l_cs,
    label="CS (l_cs)", color=:orangered, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
for row in eachrow(df_fail)
    vline!(p4, [row.sigma], color=:red, alpha=0.3, linewidth=1, label="")
end

# ── Plot 5: feedstock prices ─────────────────────────────────────────────────
p5 = plot(
    title="Feedstock prices by σ_cet",
    xlabel="σ_cet", ylabel="Price",
    size=(850, 400), background_color=:white, framestyle=:box,
    gridcolor=:lightgrey, legend=:right,
    left_margin=10Plots.mm, bottom_margin=8Plots.mm
)
plot!(p5, df_ok.sigma, df_ok.p_corn_n,
    label="Conv corn (\$/bu)", color=:goldenrod, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
plot!(p5, df_ok.sigma, df_ok.p_soy_n,
    label="Conv soy (\$/lb)", color=:forestgreen, marker=:circle, markersize=5,
    linewidth=1.5, markerstrokewidth=0)
for row in eachrow(df_fail)
    vline!(p5, [row.sigma], color=:red, alpha=0.3, linewidth=1, label="")
end

# ── 저장 ──────────────────────────────────────────────────────────────────────
savefig(p1, joinpath(OUTPUT_DIR, "traj_land_rent.png"))
savefig(p2, joinpath(OUTPUT_DIR, "traj_rent_ratio.png"))
savefig(p3, joinpath(OUTPUT_DIR, "traj_land_share.png"))
savefig(p4, joinpath(OUTPUT_DIR, "traj_land_alloc.png"))
savefig(p5, joinpath(OUTPUT_DIR, "traj_feedstock_price.png"))

println("\n✓ All plots saved.")
display(plot(p1, p2, p3, p4, p5, layout=(5, 1), size=(900, 2000)))