# sigma_sensitivity.jl
cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "model_cet_land.jl"))
include(joinpath(@__DIR__, "analysis_cet_land.jl"))

# Output paths. This file only makes figures.
# The definition has to sit at the top of the file: savefig calls above the second
# section (the sigma trajectory) already use these constants.
include(joinpath(@__DIR__, "..", "..", "main", "paths.jl"))
using .Paths
const OUT = Paths.variant("cet_land")
const OUTPUT_DIR = OUT.figures
using .ModelCETLand
import .ModelCETLand: params, build_unified_model, extract_solution, is_solved_and_feasible
using JuMP, PATHSolver, DataFrames, Printf

# Fix at RFS mandate = 0.15
policy_rfs = (
    t=0.0, θ_avi=0.15, σ=0.0, p=0.0, carbon_tax_scope=:aviation
)

# σ_cet range
sigma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

results = []

for σ_val in sigma_values
    println("\n-- Running σ_cet = $σ_val --")

    # new params with updated σ_cet
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

# Collect into a DataFrame
df = DataFrame(results)
println("\n===== σ_cet sensitivity: RFS θ_avi = 0.15 =====")
println(df)


## Plot: SAF quantity by type under RFS with varying σ_cet
using Plots

# filter out only the valid rows (where total_saf is not NaN)
df_valid = filter(row -> !isnan(row.total_saf), df)

sigma_vals = df_valid.sigma
atj_cs_vals = df_valid.saf_atj_cs
hefa_cs_vals = df_valid.saf_hefa_cs
atj_conv_vals = df_valid.saf_atj_conv
hefa_conv_vals = df_valid.saf_hefa_conv

# x axis
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
# Save
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
    # left axis (corn, $/bushel)
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
        linewidth=0.0,
        legend=false,
        size=(850, 560),
        title="Feedstock prices by σ_cet: RFS (θ_avi = 0.15)",
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
        marker=:square,
        markersize=4,
        linewidth=0.0,
        legend=false)

    # Right axis (soy, $/lb)
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

    # conv-cs vertical lines
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

    # put together  
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
        title="Total land supply by σ_cet: RFS (θ_avi = 0.15)",
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

# sigma_trajectory.jl
# Trajectory analysis: how r_corn, r_soy, l_corn and l_soy move with sigma_cet
cd(@__DIR__)
println("Working directory: ", pwd())

include(joinpath(@__DIR__, "model_cet_land.jl"))
include(joinpath(@__DIR__, "analysis_cet_land.jl"))
using .ModelCETLand
import .ModelCETLand: params, build_unified_model, extract_solution, is_solved_and_feasible
using JuMP, PATHSolver, DataFrames, Printf, Plots


# =================================================================================
# 0. Setup
# =================================================================================

# Policy fixed (RFS theta_avi = 0.15)
policy_rfs = (t=0.0, θ_avi=0.15, σ=0.0, p=0.0, carbon_tax_scope=:aviation)

# sigma_cet range: dense from 0 to 1.0
sigma_values = vcat(
    0.0,
    0.001, 0.005,
    0.01:0.01:0.09...,
    0.1:0.05:1.0...
)

# =================================================================================
# 1. Run by sigma and collect results
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

        # r_corn and r_soy exist only in the CET model
        r_corn_val = σ_val > 0 ? value(model[:r_corn]) : NaN
        r_soy_val = σ_val > 0 ? value(model[:r_soy]) : NaN
        r_land_val = sol.duals.r_land

        # l_corn and l_soy are expressions in the CET model
        l_n_corn_val = σ_val > 0 ? value(model[:l_n_corn]) : sol.l_n * 0.54
        l_cs_corn_val = σ_val > 0 ? value(model[:l_cs_corn]) : sol.l_cs * 0.54
        l_n_soy_val = σ_val > 0 ? value(model[:l_n_soy]) : sol.l_n * 0.46
        l_cs_soy_val = σ_val > 0 ? value(model[:l_cs_soy]) : sol.l_cs * 0.46

        l_corn_total = l_n_corn_val + l_cs_corn_val
        l_soy_total = l_n_soy_val + l_cs_soy_val
        l_total = sol.l_n + sol.l_cs

        # Actual shares
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
# 2. Plots
# =================================================================================

gr()


# ── Plot 1: r_corn, r_soy, r_land by σ ────────────────────────────────────────
p1 = plot(
    #title="Land rent by σ_cet  (RFS θ_avi=0.15)",
    xlabel="Elasticity of transformation (σ_cet)", ylabel="Land rent (\$/acre)",
    size=(400, 400), background_color=:white, framestyle=:box,
    gridcolor=:lightgrey, legend=:topright,
    xlims=(0.0, 0.5),
    left_margin=10Plots.mm, bottom_margin=8Plots.mm
)
plot!(p1, df_ok.sigma, df_ok.r_corn,
    label="r_corn", color=:goldenrod, marker=:circle, markersize=3,
    linewidth=1.5, markerstrokewidth=0)
plot!(p1, df_ok.sigma, df_ok.r_soy,
    label="r_soy", color=:forestgreen, marker=:circle, markersize=3,
    linewidth=1.5, markerstrokewidth=0)
plot!(p1, df_ok.sigma, df_ok.r_land,
    label="r_land", color=:navy, marker=:diamond, markersize=3,
    linewidth=2.0, linestyle=:dash, markerstrokewidth=0)
# Mark the stretches that failed
for row in eachrow(df_fail)
    vline!(p1, [row.sigma], color=:red, alpha=0.3, linewidth=1, label="")
end
if !isempty(df_fail)
    vline!(p1, [df_fail.sigma[1]], color=:red, alpha=0.3, linewidth=1, label="Failed σ")
end
p1
savefig(p1, joinpath(OUTPUT_DIR, "cet_land_rent.png"))
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
    #title="Corn/Soy land share by σ_cet",
    xlabel="Elasticity of transformation (σ_cet)", ylabel="Land share",
    size=(400, 400), background_color=:white, framestyle=:box,
    gridcolor=:lightgrey, legend=:topright,
    left_margin=10Plots.mm, bottom_margin=8Plots.mm,
    xlims=(0.0, 0.5),
    ylims=(0.3, 0.7)
)
hline!(p3, [0.54], color=:goldenrod, linestyle=:dash, linewidth=1, label="ω=0.54 (Leontief)")
hline!(p3, [0.46], color=:forestgreen, linestyle=:dash, linewidth=1, label="1-ω=0.46 (Leontief)")
plot!(p3, df_ok.sigma, df_ok.corn_share,
    label="Corn share (CET)", color=:goldenrod, marker=:circle, markersize=3,
    linewidth=1.5, markerstrokewidth=0)
plot!(p3, df_ok.sigma, df_ok.soy_share,
    label="Soybeans share (CET)", color=:forestgreen, marker=:circle, markersize=3,
    linewidth=1.5, markerstrokewidth=0)
for row in eachrow(df_fail)
    vline!(p3, [row.sigma], color=:red, alpha=0.3, linewidth=1, label="")
end
p3
savefig(p3, joinpath(OUTPUT_DIR, "cet_corn_soy_land_share.png"))

# ── Combined Plot: p1 and p3 side by side ───────────────────────────────────
p13 = plot(p1, p3, layout=(1, 2), size=(900, 420))
savefig(p13, joinpath(OUTPUT_DIR, "cet_land.png"))

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
display(p3)

# ── Save ──────────────────────────────────────────────────────────────────────
savefig(p1, joinpath(OUTPUT_DIR, "traj_land_rent.png"))
savefig(p2, joinpath(OUTPUT_DIR, "traj_rent_ratio.png"))
savefig(p3, joinpath(OUTPUT_DIR, "traj_land_share.png"))
savefig(p4, joinpath(OUTPUT_DIR, "traj_land_alloc.png"))
savefig(p5, joinpath(OUTPUT_DIR, "traj_feedstock_price.png"))

println("\n✓ All plots saved.")
display(plot(p1, p2, p3, p4, p5, layout=(5, 1), size=(900, 2000)))
