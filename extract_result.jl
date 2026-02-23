# extract results in to a CSV file
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
# Status quo and the first-best carbon tax
@load joinpath(OUTPUT_DIR, "results_base_welfare.jld2") results_base_analysis policy_configs_base status_quo cs_changes_base ps_land_base gr_changes_base env_benefits_base welfare_summary_base aac_results_base

# define observed and validation scenarios
observed_data = Dict(
    # Production (billion gallons)
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

    # Demand (billion miles or billion bushels)
    ("Demand", "avi") => 1199.1328,
    ("Demand", "gas") => 2672.651,
    ("Demand", "die") => 301.86,
    ("Demand", "corn") => 7.1951 + 1.89,
    ("Demand", "soyoil") => 14.164,
    ("Demand", "soymeal") => 49.633,

    # Feedstock prices ($/bushel)
    ("Price", "feedstock_corn_n") => 4.55,
    ("Price", "feedstock_corn_cs") => missing,
    ("Price", "feedstock_soy_n") => 0.465,
    ("Price", "feedstock_soy_cs") => missing,

    # Consumer prices ($/mile or $/unit)
    ("Price_consumer", "avi") => 0.0396,
    ("Price_consumer", "gas") => 0.128,
    ("Price_consumer", "die") => 0.336,
    ("Price_consumer", "soymeal") => 423.41,

    # Land (million acres)
    ("Land", "Conventional") => 120.09,
    ("Land", "Climate Smart") => missing,
    ("Land", "Total") => 120.09,

    # Feedstock production (billion bushels)
    ("Feedstock production", "corn_n") => 13.0,
    ("Feedstock production", "corn_cs") => missing,
    ("Feedstock production", "soy_n") => 27.77,
    ("Feedstock production", "soy_cs") => missing,

    # Byproduct production
    ("Byproduct production", "DDGS") => 1.89,
    ("Byproduct production", "Soymeal") => 49.633,)

# define observed and validation scenarios
validation_data = Dict(

    # Demand (billion miles or billion bushels)
    ("Demand", "avi") => 1204.79,
    ("Demand", "gas") => 2856.73,
    ("Demand", "die") => 357.28,
    ("Demand", "corn") => 7.1951 + 1.313879,
    ("Demand", "soyoil") => 14.164,
    ("Demand", "soymeal") => 62.272,

    # Consumer prices ($/mile or $/unit)
    ("Price_consumer", "avi") => 0.0397,
    ("Price_consumer", "gas") => 0.1279,
    ("Price_consumer", "die") => 0.3356,
    ("Price_consumer", "soymeal") => 423.41,

    # Land (million acres)
    ("Land", "Conventional") => 117.68,
    ("Land", "Climate Smart") => missing,
    ("Land", "Total") => 117.68,

    # Feedstock production (billion bushels)
    ("Feedstock production", "corn_n") => 12.04,
    ("Feedstock production", "corn_cs") => missing,
    ("Feedstock production", "soy_n") => 30.96,
    ("Feedstock production", "soy_cs") => missing,

    # Byproduct production
    ("Byproduct production", "DDGS") => 1.313879,
    ("Byproduct production", "Soymeal") => 62.272,)

# 각 타겟별 결과를 저장할 Dictionary
all_target_results = Dict()

const TARGET_SAF_VALUES = [3.0, 5.0]  # 타겟 SAF 양 (3B, 5B gallons)
for target_saf in TARGET_SAF_VALUES
    suffix = target_saf == 3.0 ? "" : "_$(Int(target_saf))"

    println("Loading results for target SAF = $(target_saf)B...")

    if target_saf == 3.0
        @load joinpath(OUTPUT_DIR, "results_equivalent_emissions_complete.jld2") equivalent_emission_policies equivalent_emission_solutions results_equiv_emission_analysis target_total_emission policy_configs_emission cs_changes_equiv_emission ps_land_equiv_emission gr_changes_equiv_emission env_benefits_equiv_emission welfare_summary_equiv_emission aac_equiv_emission SCC
    else
        filename = joinpath(OUTPUT_DIR, "results_equivalent_emissions_complete$(suffix).jld2")
        file_data = load(filename)

        equivalent_emission_policies = file_data["equivalent_emission_policies"]
        equivalent_emission_solutions = file_data["equivalent_emission_solutions"]
        results_equiv_emission_analysis = file_data["results_equiv_emission_analysis"]
        target_total_emission = file_data["target_total_emission"]
        policy_configs_emission = file_data["policy_configs_emission"]
        cs_changes_equiv_emission = file_data["cs_changes_equiv_emission"]
        ps_land_equiv_emission = file_data["ps_land_equiv_emission"]
        gr_changes_equiv_emission = file_data["gr_changes_equiv_emission"]
        env_benefits_equiv_emission = file_data["env_benefits_equiv_emission"]
        welfare_summary_equiv_emission = file_data["welfare_summary_equiv_emission"]
        aac_equiv_emission = file_data["aac_equiv_emission"]
        SCC = file_data["SCC"]
    end

    # 각 타겟별 결과를 저장
    all_target_results[target_saf] = (
        equivalent_emission_policies=equivalent_emission_policies,
        equivalent_emission_solutions=equivalent_emission_solutions,
        results_equiv_emission_analysis=results_equiv_emission_analysis,
        target_total_emission=target_total_emission,
        policy_configs_emission=policy_configs_emission,
        cs_changes_equiv_emission=cs_changes_equiv_emission,
        ps_land_equiv_emission=ps_land_equiv_emission,
        gr_changes_equiv_emission=gr_changes_equiv_emission,
        env_benefits_equiv_emission=env_benefits_equiv_emission,
        welfare_summary_equiv_emission=welfare_summary_equiv_emission,
        aac_equiv_emission=aac_equiv_emission,
        SCC=SCC
    )
end

# =================================================================================
# 2. Define Scenario Order and Labels
# =================================================================================

# 모든 타겟에 대한 시나리오 추가
scenario_order = [(:observed, :observed), (:validation, :validation), (:statusquo, :base), (:carbontax, :base)]

scenario_labels = Dict(
    (:observed, :observed) => "Observed",
    (:validation, :validation) => "Validation",
    (:statusquo, :base) => "Status Quo",
    (:carbontax, :base) => "First Best CarbonTax"
)

for target_saf in TARGET_SAF_VALUES
    suffix_label = target_saf == 3.0 ? "" : "_$(Int(target_saf))B"

    policy_label_map = Dict(
        :carbontax => "CarbonTax",
        :rfs => "RFS",
        :lcfs => "LCFS",
        :taxcredit => "TaxCredit"
    )

    for policy in [:carbontax, :rfs, :lcfs, :taxcredit]
        scenario_key = (policy, Symbol("equiv_emission_$(Int(target_saf))"))
        push!(scenario_order, scenario_key)

        scenario_labels[scenario_key] = "EquivEmission$(suffix_label)_$(policy_label_map[policy])"
    end
end

# 디버깅: 생성된 시나리오 확인
println("\n--- Generated Scenarios ---")
for scenario_key in scenario_order
    println("Key: $scenario_key => Label: $(scenario_labels[scenario_key])")
end

# =================================================================================
# 3. Initialize DataFrame
# =================================================================================

df = DataFrame()

insertcols!(df, :Category => Any[])
insertcols!(df, :Variable => Any[])
insertcols!(df, :Unit => Any[])

for scenario_key in scenario_order
    col_name = Symbol(scenario_labels[scenario_key])
    insertcols!(df, col_name => Any[])
end

println("DataFrame columns: ", names(df))

# =================================================================================
# 4. Helper Function to Add Rows
# =================================================================================

function add_row!(df, category, variable, unit, values_dict)
    new_row = Dict{Symbol,Any}(
        :Category => category,
        :Variable => variable,
        :Unit => unit
    )

    for scenario_key in scenario_order
        col_name = Symbol(scenario_labels[scenario_key])
        value = get(values_dict, scenario_key, missing)
        new_row[col_name] = value
    end

    push!(df, new_row)
end

function get_solution(scenario, group)
    if group == :observed || group == :validation
        return nothing  # Observed/Validation는 별도 처리
    elseif group == :base
        return results_base_analysis[scenario]
    else
        # Extract target_saf from group symbol
        group_str = String(group)
        target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])

        target_data = all_target_results[target_saf]
        return target_data.results_equiv_emission_analysis[scenario]
    end
end

function safe_get_value(arr, key)
    try
        if hasproperty(arr, :data) && hasproperty(arr, :axes)
            axes_symbols = arr.axes[1]
            idx = findfirst(==(key), axes_symbols)
            if !isnothing(idx)
                return arr.data[idx]
            end
        end
        return arr[key]
    catch e
        println("Warning: Could not access $key from array: $e")
        return missing
    end
end

# =================================================================================
# 6. Add Scenario Names
# =================================================================================

scenario_name_dict = Dict()
for scenario_key in scenario_order
    scenario_name_dict[scenario_key] = scenario_labels[scenario_key]
end

add_row!(df, "Metadata", "Scenario_Name", "", scenario_name_dict)

# =================================================================================
# 7. Add Policy Stringency Parameters
# =================================================================================

policy_stringency_dict = Dict()

for (scenario, group) in scenario_order
    if group == :observed || group == :validation
        policy_stringency_dict[(scenario, group)] = 0.0  # No policy
    elseif group == :base
        config = getproperty(policy_configs_base, scenario)

        value = if scenario == :carbontax
            config.t
        elseif scenario == :rfs
            config.θ_avi
        elseif scenario == :lcfs
            config.σ
        elseif scenario == :taxcredit
            config.p
        else  # statusquo
            0.0
        end

        policy_stringency_dict[(scenario, group)] = value
    else
        group_str = String(group)
        target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])
        target_data = all_target_results[target_saf]
        config = getproperty(target_data.policy_configs_emission, scenario)

        value = if scenario == :carbontax
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

        policy_stringency_dict[(scenario, group)] = value
    end
end

add_row!(df, "Policy", "Stringency", "Various", policy_stringency_dict)

# =================================================================================
# 8. Add Target Emissions
# =================================================================================

target_emission_dict = Dict()
for (scenario, group) in scenario_order
    if group == :observed || group == :validation || group == :base
        target_emission_dict[(scenario, group)] = missing
    else
        group_str = String(group)
        target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])
        target_data = all_target_results[target_saf]
        target_emission_dict[(scenario, group)] = target_data.target_total_emission
    end
end

add_row!(df, "Target", "Total_Emissions_BtonCO2e", "BtonCO2e", target_emission_dict)

# =================================================================================
# 9. Add Model Results (Values)
# =================================================================================

# Production quantities (q)
PRODUCTION_GOOD = [
    :jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy,
    :gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy,
]

for good in PRODUCTION_GOOD
    q_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            obs_key = ("Production", string(good))
            q_dict[(scenario, group)] = get(observed_data, obs_key, missing)
        elseif group == :validation
            val_key = ("Production", string(good))
            q_dict[(scenario, group)] = get(validation_data, val_key, missing)
        else
            sol = get_solution(scenario, group)
            q_dict[(scenario, group)] = safe_get_value(sol.q, good)
        end
    end
    add_row!(df, "Production", string(good), "B gal", q_dict)

    # saf_hefa_nonsoy 직후에 saf_total 삽입
    if good == :saf_hefa_nonsoy
        saf_total_dict = Dict()
        for (scenario, group) in scenario_order
            if group == :observed || group == :validation
                saf_total_dict[(scenario, group)] = missing
            else
                sol = get_solution(scenario, group)
                vals = [safe_get_value(sol.q, k) for k in [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]]
                saf_total_dict[(scenario, group)] = any(ismissing, vals) ? missing : sum(vals)
            end
        end
        add_row!(df, "Production", "saf_total", "B gal", saf_total_dict)
    end
end

# Add biodiesel_total
biodiesel_total_dict = Dict()
for (scenario, group) in scenario_order
    if group == :observed
        biodiesel_total_dict[(scenario, group)] = get(observed_data, ("Production", "biodiesel_total"), missing)
    elseif group == :validation
        biodiesel_total_dict[(scenario, group)] = get(validation_data, ("Production", "biodiesel_total"), missing)
    else
        sol = get_solution(scenario, group)
        bd_soy = safe_get_value(sol.q, :biodiesel_soy)
        bd_nonsoy = safe_get_value(sol.q, :biodiesel_nonsoy)
        if !ismissing(bd_soy) && !ismissing(bd_nonsoy)
            biodiesel_total_dict[(scenario, group)] = bd_soy + bd_nonsoy
        else
            biodiesel_total_dict[(scenario, group)] = missing
        end
    end
end
add_row!(df, "Production", "biodiesel_total", "B gal", biodiesel_total_dict)

# Add rd_total
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
        if !ismissing(rd_soy) && !ismissing(rd_nonsoy)
            rd_total_dict[(scenario, group)] = rd_soy + rd_nonsoy
        else
            rd_total_dict[(scenario, group)] = missing
        end
    end
end
add_row!(df, "Production", "rd_total", "B gal", rd_total_dict)

# Demand (x)
demand_sectors = [
    (:avi, "B gal"),
    (:gas, "B gal"),
    (:die, "B gal"),
    (:corn, "B bushel"),
    (:soyoil, "B lb"),
    (:soymeal, "MMT")
]

for (sector, unit) in demand_sectors
    x_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            obs_key = ("Demand", string(sector))
            x_dict[(scenario, group)] = get(observed_data, obs_key, missing)
        elseif group == :validation
            val_key = ("Demand", string(sector))
            x_dict[(scenario, group)] = get(validation_data, val_key, missing)
        else
            sol = get_solution(scenario, group)
            x_dict[(scenario, group)] = safe_get_value(sol.x, sector)
        end
    end
    add_row!(df, "Demand", string(sector), unit, x_dict)
end

# Feedstock prices (p_f)
feedstock_prices = [
    (:feedstock_corn_n, "\$/bushel"),
    (:feedstock_corn_cs, "\$/bushel"),
    (:feedstock_soy_n, "\$/lb"),
    (:feedstock_soy_cs, "\$/lb")
]

for (feedstock, unit) in feedstock_prices
    pf_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            obs_key = ("Price", string(feedstock))
            pf_dict[(scenario, group)] = get(observed_data, obs_key, missing)
        elseif group == :validation
            val_key = ("Price", string(feedstock))
            pf_dict[(scenario, group)] = get(validation_data, val_key, missing)
        else
            sol = get_solution(scenario, group)
            pf_dict[(scenario, group)] = safe_get_value(sol.p_f, feedstock)
        end
    end
    add_row!(df, "Price", string(feedstock), unit, pf_dict)
end

# Consumer prices (p_c)
consumer_prices = [
    (:avi, "\$/mile"),
    (:gas, "\$/mile"),
    (:die, "\$/mile"),
    (:soymeal, "\$/metric ton")
]

for (good, unit) in consumer_prices
    pc_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            obs_key = ("Price_consumer", string(good))
            pc_dict[(scenario, group)] = get(observed_data, obs_key, missing)
        elseif group == :validation
            val_key = ("Price_consumer", string(good))
            pc_dict[(scenario, group)] = get(validation_data, val_key, missing)
        else
            sol = get_solution(scenario, group)
            pc_dict[(scenario, group)] = safe_get_value(sol.p_c, good)
        end
    end
    add_row!(df, "Price_consumer", string(good), unit, pc_dict)
end

# Land use
ln_dict = Dict()
lcs_dict = Dict()
ltotal_dict = Dict()

for (scenario, group) in scenario_order
    if group == :observed
        ln_dict[(scenario, group)] = get(observed_data, ("Land", "Conventional"), missing)
        lcs_dict[(scenario, group)] = get(observed_data, ("Land", "Climate Smart"), missing)
        ltotal_dict[(scenario, group)] = get(observed_data, ("Land", "Total"), missing)
    elseif group == :validation
        ln_dict[(scenario, group)] = get(validation_data, ("Land", "Conventional"), missing)
        lcs_dict[(scenario, group)] = get(validation_data, ("Land", "Climate Smart"), missing)
        ltotal_dict[(scenario, group)] = get(validation_data, ("Land", "Total"), missing)
    else
        sol = get_solution(scenario, group)
        l_n = sol.l_n * 1000
        l_cs = sol.l_cs * 1000
        ln_dict[(scenario, group)] = l_n
        lcs_dict[(scenario, group)] = l_cs
        ltotal_dict[(scenario, group)] = l_n + l_cs
    end
end

add_row!(df, "Land", "Conventional", "M acres", ln_dict)
add_row!(df, "Land", "Climate Smart", "M acres", lcs_dict)
add_row!(df, "Land", "Total", "M acres", ltotal_dict)

# Feedstock production
feedstock_production = [
    (:corn_n, "B bushel"),
    (:corn_cs, "B bushel"),
    (:soy_n, "B lb"),
    (:soy_cs, "B lb")
]

for (ftype, unit) in feedstock_production
    qf_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            obs_key = ("Feedstock production", string(ftype))
            qf_dict[(scenario, group)] = get(observed_data, obs_key, missing)
        elseif group == :validation
            val_key = ("Feedstock production", string(ftype))
            qf_dict[(scenario, group)] = get(validation_data, val_key, missing)
        else
            sol = get_solution(scenario, group)
            qf_dict[(scenario, group)] = safe_get_value(sol.q_feedstock, ftype)
        end
    end
    add_row!(df, "Feedstock production", string(ftype), unit, qf_dict)
end

# DDGS and Soymeal
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

# Dual variables
dual_vars = [:λ_rfs, :λ_rfs_avi, :λ_lcfs, :r_land, :λ_blendwall_ethanol, :λ_blendwall_biodiesel, :λ_nonsoy_capacity]

for dual_var in dual_vars
    dual_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            obs_key = ("Shadow prices", string(dual_var))
            dual_dict[(scenario, group)] = get(observed_data, obs_key, missing)
        elseif group == :validation
            val_key = ("Shadow prices", string(dual_var))
            dual_dict[(scenario, group)] = get(validation_data, val_key, missing)
        else
            sol = get_solution(scenario, group)
            dual_dict[(scenario, group)] = safe_get_value(sol.duals, dual_var)
        end
    end
    add_row!(df, "Shadow prices", string(dual_var), "\$", dual_dict)
end

# Emissions
for emission_type in [:Aviation, :Road, :Food, :Total]
    emissions_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed
            obs_key = ("Emissions", string(emission_type))
            emissions_dict[(scenario, group)] = get(observed_data, obs_key, missing)
        elseif group == :validation
            val_key = ("Emissions", string(emission_type))
            emissions_dict[(scenario, group)] = get(validation_data, val_key, missing)
        else
            sol = get_solution(scenario, group)
            # Symbol을 소문자로 변환
            emission_key = Symbol(lowercase(string(emission_type)))
            emissions_dict[(scenario, group)] = safe_get_value(sol.emissions, emission_key)
        end
    end
    add_row!(df, "Emissions", string(emission_type), "B ton CO2e", emissions_dict)
end

# =================================================================================
# 10. Add Consumer Surplus Changes
# =================================================================================

CS_GOODS = [:avi, :gas, :die, :corn, :soyoil, :soymeal, :total]

for good in CS_GOODS
    cs_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed || group == :validation || scenario == :statusquo
            cs_dict[(scenario, group)] = missing  # Baseline, no change
        else
            if group == :base
                cs_data = cs_changes_base[scenario]
            else
                group_str = String(group)
                target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])
                target_data = all_target_results[target_saf]
                cs_data = target_data.cs_changes_equiv_emission[scenario]
            end
            cs_dict[(scenario, group)] = cs_data[good]
        end
    end
    add_row!(df, "Welfare", "CS change_$(good)", "B\$", cs_dict)
end

# =================================================================================
# 11. Add Producer Surplus Changes (Land)
# =================================================================================

ps_dict = Dict()
for (scenario, group) in scenario_order
    if group == :observed || group == :validation || scenario == :statusquo
        ps_dict[(scenario, group)] = missing
    else
        if group == :base
            ps_data = ps_land_base[scenario]
        else
            group_str = String(group)
            target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])
            target_data = all_target_results[target_saf]
            ps_data = target_data.ps_land_equiv_emission[scenario]
        end
        ps_dict[(scenario, group)] = ps_data.ps_change
    end
end
add_row!(df, "Welfare", "PS change", "B\$", ps_dict)

# =================================================================================
# 12. Add Government Revenue Changes
# =================================================================================

gr_dict = Dict()
for (scenario, group) in scenario_order
    if group == :observed || group == :validation || scenario == :statusquo
        gr_dict[(scenario, group)] = missing
    else
        if group == :base
            gr_data = gr_changes_base[scenario]
        else
            group_str = String(group)
            target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])
            target_data = all_target_results[target_saf]
            gr_data = target_data.gr_changes_equiv_emission[scenario]
        end
        gr_dict[(scenario, group)] = gr_data.total
    end
end
add_row!(df, "Welfare", "GovRevenue", "B\$", gr_dict)

# =================================================================================
# 13. Add Environmental Benefits
# =================================================================================

ENV_COMPONENTS = [:avi_benefit, :road_benefit, :food_benefit, :total_benefit]

for component in ENV_COMPONENTS
    env_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed || group == :validation || scenario == :statusquo
            env_dict[(scenario, group)] = missing
        else
            if group == :base
                env_data = env_benefits_base[scenario]
            else
                group_str = String(group)
                target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])
                target_data = all_target_results[target_saf]
                env_data = target_data.env_benefits_equiv_emission[scenario]
            end
            env_dict[(scenario, group)] = getproperty(env_data, component)
        end
    end
    add_row!(df, "Welfare", "EnvBenefit change_$(component)", "B\$", env_dict)
end

# =================================================================================
# 14. Add Welfare Summary
# =================================================================================

WELFARE_COMPONENTS = [:private_surplus, :social_welfare]

for component in WELFARE_COMPONENTS
    welfare_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed || group == :validation || scenario == :statusquo
            welfare_dict[(scenario, group)] = missing
        else
            if group == :base
                welfare_data = welfare_summary_base[scenario]
            else
                group_str = String(group)
                target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])
                target_data = all_target_results[target_saf]
                welfare_data = target_data.welfare_summary_equiv_emission[scenario]
            end
            welfare_dict[(scenario, group)] = getproperty(welfare_data, component)
        end
    end
    add_row!(df, "Welfare", "$(component)", "B\$", welfare_dict)
end

# =================================================================================
# 15. Add Average Abatement Cost
# =================================================================================

AAC_COMPONENTS = [:aac_private, :aac_social]

for component in AAC_COMPONENTS
    aac_dict = Dict()
    for (scenario, group) in scenario_order
        if group == :observed || group == :validation || scenario == :statusquo
            aac_dict[(scenario, group)] = missing
        else
            if group == :base
                aac_data = aac_results_base[scenario]
            else
                group_str = String(group)
                target_saf = parse(Float64, match(r"equiv_emission_(\d+)", group_str).captures[1])
                target_data = all_target_results[target_saf]
                aac_data = target_data.aac_equiv_emission[scenario]
            end
            aac_dict[(scenario, group)] = getproperty(aac_data, component)
        end
    end
    add_row!(df, "Average Abatement Cost", "$(component)", "\$/tonCO2", aac_dict)
end

# =================================================================================
# 16. Save to CSV with Custom Rounding
# =================================================================================

function round_value_custom(x, category, variable)
    if ismissing(x)
        return missing
    elseif !(x isa Number)
        return x
    end

    # Price_consumer: 3 decimal places
    if category in ["Price_consumer", "Price"]
        return round(x, digits=3)
        # Emissions and Welfare: 5 decimal places
    elseif category in ["Emissions", "Welfare"]
        return round(x, digits=5)
        # Default: 2 decimal places
    else
        return round(x, digits=2)
    end
end

# Apply custom rounding
for i in 1:nrow(df)
    category = df[i, :Category]
    variable = df[i, :Variable]

    for col in names(df)
        if col ∉ [:Category, :Variable, :Unit]
            df[i, col] = round_value_custom(df[i, col], category, variable)
        end
    end
end

println("\n✓ Custom rounding applied:")
println("  - Price_consumer: 3 decimal places")
println("  - Emissions & Welfare: 5 decimal places")
println("  - Others: 2 decimal places")

output_file = joinpath(OUTPUT_DIR, "results_comprehensive.csv")
CSV.write(output_file, df)

println("\n" * "="^80)
println("✓ CSV file created successfully: $output_file")
println("  Total rows: $(nrow(df))")
println("  Total columns: $(ncol(df))")
println("="^80)

# Display first few rows
println("\nFirst 20 rows of the CSV:")
show(first(df, 20), allrows=true, allcols=false)