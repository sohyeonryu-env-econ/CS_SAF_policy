# extract_results.jl
cd(@__DIR__)
println("Working directory: ", pwd())

using JLD2
using DataFrames
using CSV
using JuMP

const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"

# =================================================================================
# 1. Load results
# =================================================================================

@load joinpath(OUTPUT_DIR, "results_base.jld2") base_results base_configs
@load joinpath(OUTPUT_DIR, "results_all_cases.jld2") all_case_results TARGET_SAF

# =================================================================================
# 2. Observed and Validation Data
# =================================================================================

observed_data = Dict(
    ("Production", "jet_fuel") => 20.3386,
    ("Production", "gasoline") => 125.613,
    ("Production", "ethanol") => 14.245,
    ("Production", "diesel") => 43.9,
    ("Production", "biodiesel_soy") => 1.09326,
    ("Production", "biodiesel_nonsoy") => 0.82474,
    ("Production", "biodiesel_total") => 1.09326 + 0.82474,
    ("Production", "rd_soy") => 0.98982,
    ("Production", "rd_nonsoy") => 2.67618,
    ("Production", "rd_total") => 0.98982 + 2.67618,
    ("Demand", "avi") => 1199.1328,
    ("Demand", "gas") => 2672.651,
    ("Demand", "die") => 301.86,
    ("Demand", "corn") => 7.1951 + 1.89,
    ("Demand", "soyoil") => 14.164,
    ("Demand", "soymeal") => 49.633,
    ("Price", "feedstock_corn_n") => 4.55,
    ("Price", "feedstock_corn_cs") => missing,
    ("Price", "feedstock_soy_n") => 0.465,
    ("Price", "feedstock_soy_cs") => missing,
    ("Price_consumer", "avi") => 0.0396,
    ("Price_consumer", "gas") => 0.128,
    ("Price_consumer", "die") => 0.336,
    ("Price_consumer", "soymeal") => 423.41,
    ("Land", "Conventional") => 120.09,
    ("Land", "Climate Smart") => missing,
    ("Land", "Total") => 120.09,
    ("Feedstock production", "corn_n") => 13.0,
    ("Feedstock production", "corn_cs") => missing,
    ("Feedstock production", "soy_n") => 27.77,
    ("Feedstock production", "soy_cs") => missing,
    ("Byproduct production", "DDGS") => 1.89,
    ("Byproduct production", "Soymeal") => 49.633,
)

validation_data = Dict(
    ("Demand", "avi") => 1204.79,
    ("Demand", "gas") => 2856.73,
    ("Demand", "die") => 357.28,
    ("Demand", "corn") => 7.1951 + 1.313879,
    ("Demand", "soyoil") => 14.164,
    ("Demand", "soymeal") => 62.272,
    ("Price_consumer", "avi") => 0.0397,
    ("Price_consumer", "gas") => 0.1279,
    ("Price_consumer", "die") => 0.3356,
    ("Price_consumer", "soymeal") => 423.41,
    ("Land", "Conventional") => 117.68,
    ("Land", "Climate Smart") => missing,
    ("Land", "Total") => 117.68,
    ("Feedstock production", "corn_n") => 12.04,
    ("Feedstock production", "corn_cs") => missing,
    ("Feedstock production", "soy_n") => 30.96,
    ("Feedstock production", "soy_cs") => missing,
    ("Byproduct production", "DDGS") => 1.313879,
    ("Byproduct production", "Soymeal") => 62.272,
)

# =================================================================================
# 3. Scenario Order and Labels
# =================================================================================

# 케이스별 정책 목록
case_policies = Dict(
    :case1 => [:carbontax, :rfs, :lcfs, :taxcredit],
    :case2 => [:rfs, :lcfs, :taxcredit],
    :case3 => [:carbontax, :rfs, :lcfs, :taxcredit],
    :case4 => [:rfs, :lcfs, :taxcredit],
)

# scenario_order: (policy, case) 튜플
scenario_order = [
    (:observed, :observed),
    (:validation, :validation),
    (:statusquo, :base),
    (:firstbest, :base),
]

for case_name in [:case1, :case2, :case3, :case4]
    for policy in case_policies[case_name]
        push!(scenario_order, (policy, case_name))
    end
end

# 컬럼 레이블
policy_label_map = Dict(
    :carbontax => "CarbonTax",
    :rfs => "RFS",
    :lcfs => "LCFS",
    :taxcredit => "TaxCredit",
)

case_label_map = Dict(
    :case1 => "CS_noCIthreshold",
    :case2 => "CS_CIthreshold",
    :case3 => "noCS_noCIthreshold",
    :case4 => "noCS_CIthreshold",
)

scenario_labels = Dict(
    (:observed, :observed) => "Observed",
    (:validation, :validation) => "Validation",
    (:statusquo, :base) => "StatusQuo",
    (:firstbest, :base) => "FirstBestTax",
)


for case_name in [:case1, :case2, :case3, :case4]
    for policy in case_policies[case_name]
        key = (policy, case_name)
        scenario_labels[key] = "$(case_label_map[case_name])_$(policy_label_map[policy])"
    end
end

println("\n--- Scenario Order ---")
for key in scenario_order
    println("  $(key) => $(scenario_labels[key])")
end

# =================================================================================
# 4. Helper Functions
# =================================================================================

function get_solution(scenario, group)
    if group in (:observed, :validation)
        return nothing
    elseif group == :base
        return base_results[scenario]
    else
        return all_case_results[group].results[scenario]
    end
end

function safe_get_value(arr, key)
    try
        if hasproperty(arr, :data) && hasproperty(arr, :axes)
            axes_symbols = arr.axes[1]
            idx = findfirst(==(key), axes_symbols)
            !isnothing(idx) && return arr.data[idx]
        end
        return arr[key]
    catch e
        println("Warning: Could not access $key: $e")
        return missing
    end
end

# =================================================================================
# 5. Initialize DataFrame
# =================================================================================

df = DataFrame()
insertcols!(df, :Category => Any[])
insertcols!(df, :Variable => Any[])
insertcols!(df, :Unit => Any[])

for key in scenario_order
    insertcols!(df, Symbol(scenario_labels[key]) => Any[])
end

# =================================================================================
# 6. Add Row Helper
# =================================================================================

function add_row!(df, category, variable, unit, values_dict)
    new_row = Dict{Symbol,Any}(
        :Category => category,
        :Variable => variable,
        :Unit => unit,
    )
    for key in scenario_order
        col = Symbol(scenario_labels[key])
        new_row[col] = get(values_dict, key, missing)
    end
    push!(df, new_row)
end

# =================================================================================
# 7. Policy Stringency
# =================================================================================

stringency_dict = Dict()
for (scenario, group) in scenario_order
    if group in (:observed, :validation)
        stringency_dict[(scenario, group)] = missing
    elseif group == :base
        stringency_dict[(scenario, group)] = scenario == :firstbest ? 190.0 : 0.0
    else
        config = all_case_results[group].policy_configs[scenario]
        stringency_dict[(scenario, group)] = if scenario == :carbontax
            config.t
        elseif scenario == :rfs
            config.θ_avi
        elseif scenario == :lcfs
            config.σ
        elseif scenario == :taxcredit
            config.p
        else
            0.0
        end
    end
end
add_row!(df, "Policy", "Stringency", "Various", stringency_dict)

# =================================================================================
# 8. Target Emissions and Reduction
# =================================================================================

target_em_dict = Dict()
target_red_dict = Dict()
for (scenario, group) in scenario_order
    if group in (:observed, :validation, :base)
        target_em_dict[(scenario, group)] = missing
        target_red_dict[(scenario, group)] = missing
    else
        target_em_dict[(scenario, group)] = all_case_results[group].target_emissions
        target_red_dict[(scenario, group)] = all_case_results[group].target_reduction
    end
end
add_row!(df, "Target", "Target_Emissions_BtonCO2e", "B ton CO2e", target_em_dict)
add_row!(df, "Target", "Target_Reduction_BtonCO2e", "B ton CO2e", target_red_dict)

# =================================================================================
# 9. Production
# =================================================================================

PRODUCTION_GOODS = [
    :jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy,
    :gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy,
]

for good in PRODUCTION_GOODS
    q_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            q_dict[(scenario, group)] = get(observed_data, ("Production", string(good)), missing)
        elseif group == :validation
            q_dict[(scenario, group)] = get(validation_data, ("Production", string(good)), missing)
        else
            sol = get_solution(scenario, group)
            q_dict[(scenario, group)] = safe_get_value(sol.q, good)
        end
    end
    add_row!(df, "Production", string(good), "B gal", q_dict)

    if good == :saf_hefa_nonsoy
        saf_total_dict = Dict()
        for (scenario, group) in scenario_order
            if group in (:observed, :validation)
                saf_total_dict[(scenario, group)] = missing
            else
                sol = get_solution(scenario, group)
                vals = [safe_get_value(sol.q, k) for k in
                        [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]]
                saf_total_dict[(scenario, group)] = any(ismissing, vals) ? missing : sum(vals)
            end
        end
        add_row!(df, "Production", "saf_total", "B gal", saf_total_dict)
    end
end

# biodiesel_total
bd_total_dict = Dict()
for (scenario, group) in scenario_order
    if group == :observed
        bd_total_dict[(scenario, group)] = get(observed_data, ("Production", "biodiesel_total"), missing)
    elseif group == :validation
        bd_total_dict[(scenario, group)] = get(validation_data, ("Production", "biodiesel_total"), missing)
    else
        sol = get_solution(scenario, group)
        bd_soy = safe_get_value(sol.q, :biodiesel_soy)
        bd_nonsoy = safe_get_value(sol.q, :biodiesel_nonsoy)
        bd_total_dict[(scenario, group)] = (ismissing(bd_soy) || ismissing(bd_nonsoy)) ? missing : bd_soy + bd_nonsoy
    end
end
add_row!(df, "Production", "biodiesel_total", "B gal", bd_total_dict)

# rd_total
rd_total_dict = Dict()
for (scenario, group) in scenario_order
    if group == :observed
        rd_total_dict[(scenario, group)] = get(observed_data, ("Production", "rd_total"), missing)
    elseif group == :validation
        rd_total_dict[(scenario, group)] = get(validation_data, ("Production", "rd_total"), missing)
    else
        sol = get_solution(scenario, group)
        rd_soy = safe_get_value(sol.q, :rd_soy)
        rd_nonsoy = safe_get_value(sol.q, :rd_nonsoy)
        rd_total_dict[(scenario, group)] = (ismissing(rd_soy) || ismissing(rd_nonsoy)) ? missing : rd_soy + rd_nonsoy
    end
end
add_row!(df, "Production", "rd_total", "B gal", rd_total_dict)

# =================================================================================
# 10. Demand
# =================================================================================

demand_sectors = [
    (:avi, "B gal"),
    (:gas, "B gal"),
    (:die, "B gal"),
    (:corn, "B bushel"),
    (:soyoil, "B lb"),
    (:soymeal, "MMT"),
]

for (sector, unit) in demand_sectors
    x_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            x_dict[(scenario, group)] = get(observed_data, ("Demand", string(sector)), missing)
        elseif group == :validation
            x_dict[(scenario, group)] = get(validation_data, ("Demand", string(sector)), missing)
        else
            sol = get_solution(scenario, group)
            x_dict[(scenario, group)] = safe_get_value(sol.x, sector)
        end
    end
    add_row!(df, "Demand", string(sector), unit, x_dict)
end

# =================================================================================
# 11. Prices
# =================================================================================

feedstock_prices = [
    (:feedstock_corn_n, "\$/bushel"),
    (:feedstock_corn_cs, "\$/bushel"),
    (:feedstock_soy_n, "\$/lb"),
    (:feedstock_soy_cs, "\$/lb"),
]

for (feedstock, unit) in feedstock_prices
    pf_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            pf_dict[(scenario, group)] = get(observed_data, ("Price", string(feedstock)), missing)
        elseif group == :validation
            pf_dict[(scenario, group)] = get(validation_data, ("Price", string(feedstock)), missing)
        else
            sol = get_solution(scenario, group)
            pf_dict[(scenario, group)] = safe_get_value(sol.p_f, feedstock)
        end
    end
    add_row!(df, "Price", string(feedstock), unit, pf_dict)
end

consumer_prices = [
    (:avi, "\$/mile"),
    (:gas, "\$/mile"),
    (:die, "\$/mile"),
    (:soymeal, "\$/metric ton"),
]

for (good, unit) in consumer_prices
    pc_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            pc_dict[(scenario, group)] = get(observed_data, ("Price_consumer", string(good)), missing)
        elseif group == :validation
            pc_dict[(scenario, group)] = get(validation_data, ("Price_consumer", string(good)), missing)
        else
            sol = get_solution(scenario, group)
            pc_dict[(scenario, group)] = safe_get_value(sol.p_c, good)
        end
    end
    add_row!(df, "Price_consumer", string(good), unit, pc_dict)
end

# =================================================================================
# 12. Land
# =================================================================================

ln_dict = Dict()
lcs_dict = Dict()
ltot_dict = Dict()

for (scenario, group) in scenario_order
    if group == :observed
        ln_dict[(scenario, group)] = get(observed_data, ("Land", "Conventional"), missing)
        lcs_dict[(scenario, group)] = get(observed_data, ("Land", "Climate Smart"), missing)
        ltot_dict[(scenario, group)] = get(observed_data, ("Land", "Total"), missing)
    elseif group == :validation
        ln_dict[(scenario, group)] = get(validation_data, ("Land", "Conventional"), missing)
        lcs_dict[(scenario, group)] = get(validation_data, ("Land", "Climate Smart"), missing)
        ltot_dict[(scenario, group)] = get(validation_data, ("Land", "Total"), missing)
    else
        sol = get_solution(scenario, group)
        ln_dict[(scenario, group)] = sol.l_n * 1000
        lcs_dict[(scenario, group)] = sol.l_cs * 1000
        ltot_dict[(scenario, group)] = (sol.l_n + sol.l_cs) * 1000
    end
end

add_row!(df, "Land", "Conventional", "M acres", ln_dict)
add_row!(df, "Land", "Climate Smart", "M acres", lcs_dict)
add_row!(df, "Land", "Total", "M acres", ltot_dict)

# =================================================================================
# 13. Feedstock Production
# =================================================================================

feedstock_production = [
    (:corn_n, "B bushel"),
    (:corn_cs, "B bushel"),
    (:soy_n, "B lb"),
    (:soy_cs, "B lb"),
]

for (ftype, unit) in feedstock_production
    qf_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            qf_dict[(scenario, group)] = get(observed_data, ("Feedstock production", string(ftype)), missing)
        elseif group == :validation
            qf_dict[(scenario, group)] = get(validation_data, ("Feedstock production", string(ftype)), missing)
        else
            sol = get_solution(scenario, group)
            qf_dict[(scenario, group)] = safe_get_value(sol.q_feedstock, ftype)
        end
    end
    add_row!(df, "Feedstock production", string(ftype), unit, qf_dict)
end

# =================================================================================
# 14. Byproducts
# =================================================================================

ddgs_dict = Dict()
soymeal_dict = Dict()

for (scenario, group) in scenario_order
    if group == :observed
        ddgs_dict[(scenario, group)] = get(observed_data, ("Byproduct production", "DDGS"), missing)
        soymeal_dict[(scenario, group)] = get(observed_data, ("Byproduct production", "Soymeal"), missing)
    elseif group == :validation
        ddgs_dict[(scenario, group)] = get(validation_data, ("Byproduct production", "DDGS"), missing)
        soymeal_dict[(scenario, group)] = get(validation_data, ("Byproduct production", "Soymeal"), missing)
    else
        sol = get_solution(scenario, group)
        ddgs_dict[(scenario, group)] = sol.ddgs
        soymeal_dict[(scenario, group)] = sol.soymeal_produ
    end
end

add_row!(df, "Byproduct production", "DDGS", "B bushel", ddgs_dict)
add_row!(df, "Byproduct production", "Soymeal", "MMT", soymeal_dict)

# =================================================================================
# 15. Dual Variables
# =================================================================================

dual_vars = [
    (:λ_rfs, "lambda_rfs"),
    (:λ_rfs_avi, "lambda_rfs_avi"),
    (:λ_lcfs, "lambda_lcfs"),
    (:r_land, "r_land"),
    (:λ_blendwall_ethanol, "lambda_blendwall_ethanol"),
    (:λ_blendwall_biodiesel, "lambda_blendwall_biodiesel"),
    (:λ_nonsoy_capacity, "lambda_nonsoy_capacity"),
]

for (dv, dv_label) in dual_vars
    dv_dict = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation)
            dv_dict[(scenario, group)] = missing
        else
            sol = get_solution(scenario, group)
            dv_dict[(scenario, group)] = safe_get_value(sol.duals, dv)
        end
    end
    add_row!(df, "Shadow prices", dv_label, "\$", dv_dict)
end

# =================================================================================
# 16. Emissions
# =================================================================================

for em_type in [:aviation, :road, :food, :total]
    em_dict = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation)
            em_dict[(scenario, group)] = missing
        else
            sol = get_solution(scenario, group)
            em_dict[(scenario, group)] = safe_get_value(sol.emissions, em_type)
        end
    end
    add_row!(df, "Emissions", string(em_type), "B ton CO2e", em_dict)
end

# =================================================================================
# 17. Welfare
# =================================================================================

# CS changes
for good in [:avi, :gas, :die, :corn, :soyoil, :soymeal, :total]
    cs_dict = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation, :base)
            cs_dict[(scenario, group)] = missing
        else
            cs_dict[(scenario, group)] = all_case_results[group].welfare.cs_changes[scenario][good]
        end
    end
    add_row!(df, "Welfare", "CS_change_$(good)", "B\$", cs_dict)
end

# PS land
ps_dict = Dict()
for (scenario, group) in scenario_order
    if group in (:observed, :validation, :base)
        ps_dict[(scenario, group)] = missing
    else
        ps_dict[(scenario, group)] = all_case_results[group].welfare.ps_land[scenario].ps_change
    end
end
add_row!(df, "Welfare", "PS_change", "B\$", ps_dict)

# Government revenue
gr_dict = Dict()
for (scenario, group) in scenario_order
    if group in (:observed, :validation, :base)
        gr_dict[(scenario, group)] = missing
    else
        gr_dict[(scenario, group)] = all_case_results[group].welfare.gr_changes[scenario].total
    end
end
add_row!(df, "Welfare", "GovRevenue", "B\$", gr_dict)

# Environmental benefits
for component in [:avi_benefit, :road_benefit, :food_benefit, :total_benefit]
    env_dict = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation, :base)
            env_dict[(scenario, group)] = missing
        else
            env_dict[(scenario, group)] = getproperty(
                all_case_results[group].welfare.env_benefits[scenario], component)
        end
    end
    add_row!(df, "Welfare", "EnvBenefit_$(component)", "B\$", env_dict)
end

# Welfare summary
for component in [:private_surplus, :social_welfare]
    w_dict = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation, :base)
            w_dict[(scenario, group)] = missing
        else
            w_dict[(scenario, group)] = getproperty(
                all_case_results[group].welfare.welfare_summary[scenario], component)
        end
    end
    add_row!(df, "Welfare", string(component), "B\$", w_dict)
end

# AAC
for component in [:aac_private, :aac_social]
    aac_dict = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation, :base)
            aac_dict[(scenario, group)] = missing
        else
            aac_dict[(scenario, group)] = getproperty(
                all_case_results[group].welfare.aac_results[scenario], component)
        end
    end
    add_row!(df, "Average Abatement Cost", string(component), "\$/tonCO2", aac_dict)
end

# =================================================================================
# 18. Rounding and Save
# =================================================================================

function round_value_custom(x, category)
    ismissing(x) && return missing
    !(x isa Number) && return x
    if category in ("Price_consumer", "Price")
        return round(x, digits=3)
    elseif category in ("Emissions", "Welfare")
        return round(x, digits=5)
    else
        return round(x, digits=2)
    end
end

for i in 1:nrow(df)
    cat = df[i, :Category]
    for col in names(df)
        col ∉ ["Category", "Variable", "Unit"] &&
            (df[i, col] = round_value_custom(df[i, col], cat))
    end
end

output_file = joinpath(OUTPUT_DIR, "results_comprehensive.csv")
CSV.write(output_file, df)

println("\n" * "="^80)
println("✓ CSV saved: $output_file")
println("  Rows: $(nrow(df)), Columns: $(ncol(df))")
println("="^80)
println("\nFirst 20 rows:")
show(first(df, 20), allrows=true, allcols=false)