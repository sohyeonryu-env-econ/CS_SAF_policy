# sigma_sensitivity.jl

using .SAFModel
import .SAFModel: params, build_unified_model, extract_solution, is_solved_and_feasible
using JuMP, PATHSolver, DataFrames, Printf

# RFS mandate = 0.15 고정
policy_rfs = (
    t=0.0, θ_avi=0.15, σ=0.0, p=0.0, carbon_tax_scope=:aviation
)

# σ_cet 값 범위
sigma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

results = []

for σ_val in sigma_values
    println("\n-- Running σ_cet = $σ_val --")

    # σ_cet만 바꾼 새 params 생성
    new_coeff = merge(params.coeff, (σ_cet=σ_val,))
    new_params = merge(params, (coeff=new_coeff,))

    model = build_unified_model(new_params, policy_rfs)
    optimize!(model)

    if is_solved_and_feasible(model)
        println("✓ solved")
        sol = extract_solution(model, :rfs)

        push!(results, (
            sigma=σ_val,
            # SAF quantities
            saf_atj_conv=sol.q[:saf_atj_conv],
            saf_atj_cs=sol.q[:saf_atj_cs],
            saf_hefa_conv=sol.q[:saf_hefa_conv],
            saf_hefa_cs=sol.q[:saf_hefa_cs],
            total_saf=sum(sol.q[g] for g in [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]),
            # feedstock prices
            p_corn_n=sol.p_f[:feedstock_corn_n],
            p_corn_cs=sol.p_f[:feedstock_corn_cs],
            p_soy_n=sol.p_f[:feedstock_soy_n],
            p_soy_cs=sol.p_f[:feedstock_soy_cs],
            # land
            l_n=sol.l_n,
            l_cs=sol.l_cs,
            r_land=sol.duals.r_land
        ))
    else
        println("✗ failed")
        push!(results, (
            sigma=σ_val,
            saf_atj_conv=NaN, saf_atj_cs=NaN,
            saf_hefa_conv=NaN, saf_hefa_cs=NaN, total_saf=NaN,
            p_corn_n=NaN, p_corn_cs=NaN,
            p_soy_n=NaN, p_soy_cs=NaN,
            l_n=NaN, l_cs=NaN, r_land=NaN
        ))
    end
end

# DataFrame으로 정리
df = DataFrame(results)
println("\n===== σ_cet sensitivity: RFS θ_avi = 0.15 =====")
println(df)


## Plot: SAF quantity by type under RFS with varying σ_cet
using Plots

# 유효한 결과만 필터
df_valid = filter(row -> !isnan(row.total_saf), df)

sigma_vals = df_valid.sigma
atj_cs_vals = df_valid.saf_atj_cs
hefa_cs_vals = df_valid.saf_hefa_cs
atj_conv_vals = df_valid.saf_atj_conv
hefa_conv_vals = df_valid.saf_hefa_conv

# x축 위치 (bar width 맞추기)
x = 1:length(sigma_vals)
bar_width = 0.6

begin
    p = plot(
        size=(800, 500),
        xlabel="σ_cet (elasticity of transformation)",
        ylabel="SAF quantity (billion gallons)",
        title="SAF quantity by type under RFS (θ_avi = 0.15)",
        legend=:topleft,
        xticks=(x, string.(sigma_vals)),
        ylims=(0, maximum(df_valid.total_saf) * 1.1),
        leftmargin=10Plots.mm,
        gridcolor=:lightgrey,
        gridlinewidth=0.5,
        background_color=:white,
        framestyle=:box
    )

    # Stacked bar: bottom to top
    # 1. CS ATJ
    bar!(p, x, atj_cs_vals,
        label="CS ATJ-SAF",
        bar_width=bar_width,
        color=:red,
        alpha=0.85,
        bottom=atj_conv_vals .+ hefa_conv_vals)

    # 2. CS HEFA (top)
    bar!(p, x, hefa_cs_vals,
        label="CS HEFA-SAF",
        bar_width=bar_width,
        color=:orange,
        alpha=0.85,
        bottom=atj_conv_vals .+ hefa_conv_vals .+ atj_cs_vals)
end
# 저장
#savefig(p, joinpath(OUTPUT_DIR, "saf_by_sigma_rfs.png"))
#println("✓ Saved saf_by_sigma_rfs.png")
#display(p)

## Plot: feedstock prices by σ_cet

df_valid = filter(row -> !isnan(row.p_corn_n), df)

sigma_vals = df_valid.sigma
p_corn_n = df_valid.p_corn_n
p_corn_cs = df_valid.p_corn_cs
p_soy_n = df_valid.p_soy_n
p_soy_cs = df_valid.p_soy_cs

begin
    # 왼쪽 축 (corn, $/bushel)
    p_left = plot(
        sigma_vals, p_corn_n,
        label="Conv corn (\$/bu)",
        xlabel="σ_cet",
        ylabel="\$/bushel (corn)",
        ylims=(0, 7),
        xlims=(-0.05, 1.05),
        color=:goldenrod,
        marker=:circle,
        markersize=4,
        linewidth=0.0,          # 연결선 없음
        legend=false,
        size=(850, 560),
        title="Feedstock prices by σ_cet — RFS (θ_avi = 0.15)",
        background_color=:white,
        framestyle=:box,
        gridcolor=:lightgrey,
        gridlinewidth=0.5,
        left_margin=10Plots.mm,
        right_margin=10Plots.mm,
        top_margin=5Plots.mm
    )

    plot!(p_left, sigma_vals, p_corn_cs,
        label="CS corn (\$/bu)",
        color=:goldenrod,
        marker=:square,       # * 모양
        markersize=4,
        linewidth=0.0,
        legend=false)

    # 오른쪽 축 (soy, $/lb)
    p_right = twinx(p_left)

    plot!(p_right, sigma_vals, p_soy_n,
        label="Conv soy (\$/lb)",
        ylabel="\$/lb (soy oil)",
        ylims=(0, 1),
        color=:forestgreen,
        marker=:circle,
        markersize=4,
        linewidth=0.0,
        legend=false)

    plot!(p_right, sigma_vals, p_soy_cs,
        label="CS soy (\$/lb)",
        color=:forestgreen,
        marker=:square,
        markersize=4,
        linewidth=0.0,
        legend=false)

    # conv-cs 수직 연결선 추가 (p_left 에 corn, p_right 에 soy)
    for i in 1:length(sigma_vals)
        plot!(p_left, [sigma_vals[i], sigma_vals[i]], [p_corn_n[i], p_corn_cs[i]],
            color=:goldenrod, linewidth=1, alpha=0.5, label=false)
        plot!(p_right, [sigma_vals[i], sigma_vals[i]], [p_soy_n[i], p_soy_cs[i]],
            color=:forestgreen, linewidth=1, alpha=0.5, label=false)
    end

    # legend
    p_legend = plot(
        [NaN], [NaN], label="Conv corn", color=:goldenrod, marker=:circle, markersize=4, linewidth=0, markerstrokewidth=0.5,
        axis=false, border=:none, background_color=:white, legend=:inside, legendcolumns=4,
    )
    plot!(p_legend, [NaN], [NaN], label="CS corn", color=:goldenrod, marker=:square, markersize=4, linewidth=0, markerstrokewidth=0.5)
    plot!(p_legend, [NaN], [NaN], label="Conv soy", color=:forestgreen, marker=:circle, markersize=4, linewidth=0, markerstrokewidth=0.5)
    plot!(p_legend, [NaN], [NaN], label="CS soy", color=:forestgreen, marker=:square, markersize=4, linewidth=0, markerstrokewidth=0.5)

    # 메인 그래프 + legend 패널 수직 결합
    p_final = plot(p_left, p_legend, layout=grid(2, 1, heights=[0.85, 0.15]), size=(850, 580))

end

savefig(p_left, joinpath(OUTPUT_DIR, "feedstock_price_by_sigma_rfs.png"))
println("✓ Saved feedstock_price_by_sigma_rfs.png")
display(p_left)

## Plot: land allocation by σ_cet
sigma_vals = df_valid.sigma
l_n_vals = df_valid.l_n
l_cs_vals = df_valid.l_cs
l_total = l_n_vals .+ l_cs_vals

begin
    p_land = plot(
        sigma_vals, l_total,
        label="Total land",
        xlabel="σ_cet",
        ylabel="Land (billion acres)",
        ylims=(0, maximum(l_total) * 1.2),
        xlims=(-0.05, 1.05),
        color=:brown,
        marker=:circle,
        markersize=6,
        linewidth=1.5,
        markerstrokewidth=0,
        size=(850, 500),
        title="Total land supply by σ_cet — RFS (θ_avi = 0.15)",
        background_color=:white,
        framestyle=:box,
        gridcolor=:lightgrey,
        gridlinewidth=0.5,
        left_margin=10Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=10Plots.mm,
        top_margin=5Plots.mm,
        legend=:topright
    )

    plot!(p_land, sigma_vals, l_n_vals,
        label="Conv land (l_n)",
        color=:steelblue,
        marker=:circle,
        markersize=6,
        linewidth=1.5,
        markerstrokewidth=0)

    plot!(p_land, sigma_vals, l_cs_vals,
        label="CS land (l_cs)",
        color=:orangered,
        marker=:circle,
        markersize=6,
        linewidth=1.5,
        markerstrokewidth=0)

end
savefig(p_land, joinpath(OUTPUT_DIR, "land_by_sigma_rfs.png"))
println("✓ Saved land_by_sigma_rfs.png")
display(p_land)
