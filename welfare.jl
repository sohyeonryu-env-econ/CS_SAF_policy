# welfare.jl
cd(@__DIR__)
include(joinpath(@__DIR__, "SAFModel.jl"))
using JLD2
using DataFrames
using Printf
using .SAFModel
const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results_trials/"


# =================================================================================
# 2. Welfare Analysis Functions
# =================================================================================

# Clean small values
clean_small(val, threshold=1e-10) = abs(val) < threshold ? 0.0 : val

# =====================
# 1) Consumer Surplus Change
# =====================
# Calculate consumer surplus change for a single good
function calc_cs_change(A, k, x_policy, x_sq, p_policy, p_sq)
    return (A / (k + 1)) * (x_policy^(k + 1) - x_sq^(k + 1)) - p_policy * x_policy + p_sq * x_sq
end

# CS change calculate for all goods and scenarios
function calculate_cs_changes(solutions, solution_sq, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    # Remove statusquo from scenario list if present
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    demand = params.demand

    # Use status quo solution from base analysis
    x_sq = solution_sq.x
    p_sq_c = solution_sq.p_c
    p_sq_f = solution_sq.p_f

    cs_changes_all = Dict()

    for scenario in scenario_list
        solution_policy = solutions[scenario]
        x_policy = solution_policy.x
        p_policy_c = solution_policy.p_c
        p_policy_f = solution_policy.p_f

        cs_changes = Dict(
            :avi => calc_cs_change(
                demand[:avi].A, demand[:avi].k,
                x_policy[:avi], x_sq[:avi],
                p_policy_c[:avi], p_sq_c[:avi]
            ),
            :gas => calc_cs_change(
                demand[:gas].A, demand[:gas].k,
                x_policy[:gas], x_sq[:gas],
                p_policy_c[:gas], p_sq_c[:gas]
            ),
            :die => calc_cs_change(
                demand[:die].A, demand[:die].k,
                x_policy[:die], x_sq[:die],
                p_policy_c[:die], p_sq_c[:die]
            ),
            :corn => calc_cs_change(
                demand[:corn].A, demand[:corn].k,
                x_policy[:corn], x_sq[:corn],
                p_policy_f[:feedstock_corn_n], p_sq_f[:feedstock_corn_n]
            ),
            :soyoil => calc_cs_change(
                demand[:soyoil].A, demand[:soyoil].k,
                x_policy[:soyoil], x_sq[:soyoil],
                p_policy_f[:feedstock_soy_n], p_sq_f[:feedstock_soy_n]
            ),
            :soymeal => calc_cs_change(
                demand[:soymeal].A, demand[:soymeal].k,
                x_policy[:soymeal], x_sq[:soymeal],
                p_policy_c[:soymeal], p_sq_c[:soymeal]
            ) / 1000 # Convert to billion dollars (soymeal is in million metric ton)
        )

        cs_changes[:total] = sum(v for (k, v) in cs_changes if k != :total)
        cs_changes_all[scenario] = cs_changes
    end

    return cs_changes_all
end

# Create consumer surplus change table
function make_cs_change_table(cs_changes_all; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(cs_changes_all)) : scenarios

    df = DataFrame(Good=String[])
    for scenario in scenario_list
        df[!, String(scenario)] = Float64[]
    end

    goods = [
        (:avi, "Aviation"),
        (:gas, "Road Gasoline"),
        (:die, "Road Diesel"),
        (:corn, "Food: Corn"),
        (:soyoil, "Food: Soyoil"),
        (:soymeal, "Food: Soymeal"),
        (:total, "TOTAL")
    ]

    for (good_key, good_label) in goods
        push!(df.Good, good_label)
        for scenario in scenario_list
            value = clean_small(cs_changes_all[scenario][good_key])
            push!(df[!, String(scenario)], value)
        end
    end

    return df
end

# Display
function display_cs_changes(cs_changes_all; scenarios=nothing, title="CONSUMER SURPLUS CHANGES (billion \$)")
    println("\n" * "="^130)
    println(title)
    println("="^130)

    cs_table = make_cs_change_table(cs_changes_all; scenarios=scenarios)
    show(cs_table, allrows=true)

    println("\n" * "="^130)
end


# =====================
# 2. LAND PRODUCER SURPLUS CHANGES
# =====================
# Fossil fuel producer surplus is always zero in this model (perfectly elastic supply at fixed price)
"""
Calculate land producer surplus changes
Land supply: L = L0 * (r/r0)^ε
Inverse supply (MC): r(L) = r0 * (L/L0)^(1/ε)
PS Change: ΔPS = r_policy*L_policy - r_sq*L_sq - ∫[L_sq to L_policy] r(L) dL
"""
# calculate PS changes for all scenarios
function calculate_ps_land_changes(solutions, solution_sq, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    land_supply = params.supply.land
    L0 = land_supply.L0
    r0 = land_supply.r0_land
    ε = land_supply.ϵ_land

    # Status quo land rent and use
    r_sq = solution_sq.duals.r_land
    L_sq = solution_sq.l_n + solution_sq.l_cs

    ps_land_changes = Dict()

    for scenario in scenario_list
        solution_policy = solutions[scenario]
        r_policy = solution_policy.duals.r_land
        L_policy = solution_policy.l_n + solution_policy.l_cs

        # ΔPS = r_policy*L_policy - r_sq*L_sq - ∫[L_sq to L_policy] r(L) dL
        # where r(L) = r0 * (L/L0)^(1/ε)
        # ∫[L_sq to L_policy] r(L) dL = r0 * L0^(-1/ε) * ε/(ε+1) * [L^((ε+1)/ε)]|[L_sq to L_policy]

        integral_term = r0 * (L0^(-1 / ε)) * (ε / (ε + 1)) *
                        (L_policy^((ε + 1) / ε) - L_sq^((ε + 1) / ε))

        ps_land_change = r_policy * L_policy - r_sq * L_sq - integral_term

        ps_land_changes[scenario] = (
            ps_change=clean_small(ps_land_change),
            r_sq=clean_small(r_sq),
            r_policy=clean_small(r_policy),
            L_sq=clean_small(L_sq),
            L_policy=clean_small(L_policy),
            integral_term=clean_small(integral_term)
        )
    end

    return ps_land_changes
end


# Display land producer surplus changes
function display_ps_land_changes(ps_land_changes; scenarios=nothing,
    title="LAND PRODUCER SURPLUS CHANGES (billion \$)")
    scenario_list = isnothing(scenarios) ? collect(keys(ps_land_changes)) : scenarios

    println("\n" * "="^130)
    println(title)
    println("="^130)

    # Create DataFrame with metric as first column and scenarios as other columns
    df = DataFrame(Metric=String[])

    # Add scenario columns
    for scenario in scenario_list
        df[!, scenario] = Float64[]
    end

    # Add PS Change row
    push!(df.Metric, "PS Change (Land)")
    for scenario in scenario_list
        ps = ps_land_changes[scenario]
        push!(df[!, scenario], ps.ps_change)
    end

    show(df, allrows=true)

    println("\n" * "="^130)
end


# =====================
# 3) GOVERNMENT REVENUE CHANGES (only for carbon tax and tax credit)
# =====================
# GR in Status quo is zero
function calculate_gov_revenue_change(solution_policy, implicit_taxes_policy, scenario)
    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ELIGIBLE_SAF = [:saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    # ⭐ startswith로 정책 타입 판별
    scenario_str = String(scenario)
    is_carbontax = startswith(scenario_str, "carbontax")
    is_taxcredit = startswith(scenario_str, "taxcredit")

    if !(is_carbontax || is_taxcredit)
        return 0.0
    end

    gov_revenue = 0.0

    if is_carbontax
        for fuel in AVIATION_FUELS
            t_i = implicit_taxes_policy[fuel][:carbon_tax]
            q_i = solution_policy.q[fuel]
            gov_revenue += t_i * q_i
        end
    elseif is_taxcredit
        for saf in ELIGIBLE_SAF
            if haskey(implicit_taxes_policy, saf) && haskey(implicit_taxes_policy[saf], :tax_credit)
                s_i = implicit_taxes_policy[saf][:tax_credit]
                q_i = solution_policy.q[saf]
                gov_revenue += s_i * q_i
            end
        end
    end

    return gov_revenue
end

function calculate_gr_changes(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)
    gr_changes = Dict()

    for scenario in scenario_list
        solution_policy = solutions[scenario]
        implicit_taxes_policy = solution_policy.implicit_taxes

        gr = calculate_gov_revenue_change(solution_policy, implicit_taxes_policy, scenario)

        # ⭐ startswith로 분류
        scenario_str = String(scenario)
        gr_changes[scenario] = (
            total=clean_small(gr),
            carbon_tax=startswith(scenario_str, "carbontax") ? clean_small(gr) : 0.0,
            tax_credit=startswith(scenario_str, "taxcredit") ? clean_small(gr) : 0.0
        )
    end

    return gr_changes
end

# Display government revenue
function display_gr_changes(gr_changes; scenarios=nothing, title="GOVERNMENT REVENUE CHANGES (billion \$)")
    scenario_list = isnothing(scenarios) ? collect(keys(gr_changes)) : scenarios

    println("\n" * "="^80)
    println(title)
    println("="^80)

    df = DataFrame()

    # Add scenario columns
    for scenario in scenario_list
        df[!, scenario] = [gr_changes[scenario].total]
    end

    # Add row label
    insertcols!(df, 1, :Metric => ["Total GR"])

    show(df, allrows=true)

    println("\n" * "="^80)
end


# =================================================================================
# ENVIRONMENTAL BENEFIT ANALYSIS
# =================================================================================

"""
Calculate environmental benefit from emission reduction
Uses social cost of carbon (SCC)

Note on units:
- delta[g]: ton CO2e per gallon (carbon intensity)
- sol.q[g]: billion gallons (production quantity)
- emissions in solutions: billion ton CO2e
- SCC: \$ per ton CO2e
- Output: billion \$
"""
function calculate_environmental_benefit(solutions, solution_sq, scc; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    # Status quo emissions (billion ton CO2e)
    emissions_sq = solution_sq.emissions

    env_benefits = Dict()

    for scenario in scenario_list
        solution_policy = solutions[scenario]
        emissions_policy = solution_policy.emissions

        # Emission reductions (billion ton CO2e)
        avi_reduction = emissions_sq.aviation - emissions_policy.aviation
        road_reduction = emissions_sq.road - emissions_policy.road
        food_reduction = emissions_sq.food - emissions_policy.food
        total_reduction = emissions_sq.total - emissions_policy.total

        # Environmental benefits (billion $)
        avi_benefit = avi_reduction * scc
        road_benefit = road_reduction * scc
        food_benefit = food_reduction * scc
        total_benefit = total_reduction * scc

        env_benefits[scenario] = (
            # Environmental benefits (billion $)
            avi_benefit=clean_small(avi_benefit),
            road_benefit=clean_small(road_benefit),
            food_benefit=clean_small(food_benefit),
            total_benefit=clean_small(total_benefit)
        )
    end

    return env_benefits
end

"""
Display environmental benefits
"""
function display_environmental_benefits(env_benefits, scc; scenarios=nothing,
    title="ENVIRONMENTAL BENEFITS")
    scenario_list = isnothing(scenarios) ? collect(keys(env_benefits)) : scenarios

    println("\n" * "="^130)
    println(title)
    println("Social Cost of Carbon (SCC) = \$$(scc) per ton CO2e")
    println("="^130)

    # Environmental benefits table (billion $)
    df_benefit = DataFrame(Sector=String[])

    # 각 시나리오를 열로 추가
    for scenario in scenario_list
        df_benefit[!, scenario] = Float64[]
    end

    # 각 부문을 행으로 추가
    sectors = [
        ("Aviation", :avi_benefit),
        ("Road", :road_benefit),
        ("Food", :food_benefit),
        ("Total", :total_benefit)
    ]

    for (sector_name, benefit_key) in sectors
        push!(df_benefit.Sector, sector_name)
        for scenario in scenario_list
            eb = env_benefits[scenario]
            push!(df_benefit[!, scenario], getfield(eb, benefit_key))
        end
    end

    println("\n--- Environmental Benefits (billion \$) ---")
    show(df_benefit, allrows=true)

    println("\n" * "="^130)
end


# =============================================================
# 4) TOTAL WELFARE CALCULATION
# =============================================================

# Calculate total welfare changes combining all components
function calculate_total_welfare(cs_changes, ps_land_changes, gr_changes, env_benefits; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(cs_changes)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    welfare_summary = Dict()

    for scenario in scenario_list
        cs_change = cs_changes[scenario][:total]
        ps_land_change = ps_land_changes[scenario].ps_change
        gr_change = gr_changes[scenario].total
        env_benefit = env_benefits[scenario].total_benefit

        # Private surplus = CS + PS (Land)
        private_surplus = cs_change + ps_land_change + gr_change

        # Social welfare = Private surplus + Environmental benefit
        social_welfare = private_surplus + env_benefit

        welfare_summary[scenario] = (
            cs_change=clean_small(cs_change),
            ps_land_change=clean_small(ps_land_change),
            gr_change=clean_small(gr_change),
            env_benefit=clean_small(env_benefit),
            private_surplus=clean_small(private_surplus),
            social_welfare=clean_small(social_welfare)
        )
    end

    return welfare_summary
end

# Display total welfare summary table
function display_welfare_summary(welfare_summary; scenarios=nothing,
    title="WELFARE SUMMARY")
    scenario_list = isnothing(scenarios) ? collect(keys(welfare_summary)) : scenarios

    println("\n" * "="^130)
    println(title)
    println("="^130)

    # Create DataFrame with metrics as rows and scenarios as columns
    df = DataFrame(Metric=String[])

    # Add scenario columns
    for scenario in scenario_list
        df[!, scenario] = Float64[]
    end

    # Define metrics to display
    metrics = [
        ("CS Change (a)", :cs_change),
        ("PS Change (b)", :ps_land_change),
        ("Gov Revenue (c)", :gr_change),
        ("Env Benefit (d)", :env_benefit),
        ("Private Surplus (∆=a+b+c)", :private_surplus),
        ("Social Welfare (∆+d)", :social_welfare)
    ]

    # Fill in the data
    for (metric_name, metric_key) in metrics
        push!(df.Metric, metric_name)
        for scenario in scenario_list
            w = welfare_summary[scenario]
            push!(df[!, scenario], getfield(w, metric_key))
        end
    end

    println("\n--- All values in billion \$ ---")
    show(df, allrows=true)

    println("\n" * "="^130)
end


# =============================================================
# 3. AVERAGE ABATEMENT COST CALCULATION
# =============================================================

"""
Calculate average abatement cost (AAC)
AAC measures the cost per unit of emission reduction

AAC_private = -ΔPrivate Surplus / ΔEmissions
AAC_social = -ΔSocial Welfare / ΔEmissions

where:
- ΔPrivate Surplus = CS Change + PS Change + Gov Revenue
- ΔSocial Welfare = Private Surplus + Environmental Benefit
- ΔEmissions = Emissions_SQ - Emissions_Policy (positive for reduction)

Units:
- Welfare changes: billion \$
- Emission changes: billion ton CO2e
- AAC: \$ per ton CO2e
"""
function calculate_average_abatement_cost(welfare_summary, solutions, solution_sq; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(welfare_summary)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    # Status quo emissions (billion ton CO2e)
    emissions_sq = solution_sq.emissions.total

    aac_results = Dict()

    for scenario in scenario_list
        solution_policy = solutions[scenario]
        emissions_policy = solution_policy.emissions.total

        # Emission reduction (billion ton CO2e)
        emission_reduction = emissions_sq - emissions_policy

        w = welfare_summary[scenario]

        # AAC calculations ($ per ton CO2e)
        # Note: We use negative welfare change because welfare loss represents cost
        if abs(emission_reduction) > 1e-10
            aac_private = -w.private_surplus / emission_reduction
            aac_social = -w.social_welfare / emission_reduction
        else
            aac_private = 0.0
            aac_social = 0.0
        end

        aac_results[scenario] = (
            emission_reduction=clean_small(emission_reduction),
            private_surplus=clean_small(w.private_surplus),
            social_welfare=clean_small(w.social_welfare),
            aac_private=clean_small(aac_private),
            aac_social=clean_small(aac_social)
        )
    end

    return aac_results
end

"""
Display average abatement cost analysis
"""
function display_aac_analysis(aac_results; scenarios=nothing,
    title="AVERAGE ABATEMENT COST ANALYSIS")
    scenario_list = isnothing(scenarios) ? collect(keys(aac_results)) : scenarios

    println("\n" * "="^130)
    println(title)
    println("="^130)

    # Create DataFrame with metrics as rows and scenarios as columns
    df = DataFrame(Metric=String[])

    # Add scenario columns
    for scenario in scenario_list
        df[!, scenario] = Float64[]
    end

    # Define metrics to display
    metrics = [
        ("Emission Reduction (B ton CO2e)", :emission_reduction),
        ("∆Private welfare (billion \$)", :private_surplus),
        ("∆Social welfare (billion \$)", :social_welfare),
        ("AAC (Private) (\$/ton CO2e)", :aac_private),
        ("AAC (Social) (\$/ton CO2e)", :aac_social)
    ]

    # Fill in the data
    for (metric_name, metric_key) in metrics
        push!(df.Metric, metric_name)
        for scenario in scenario_list
            aac = aac_results[scenario]
            push!(df[!, scenario], getfield(aac, metric_key))
        end
    end

    show(df, allrows=true)

    println("\n" * "="^130)
end


# =================================================================================
# 4. Run and Save
# =================================================================================

# 1) Base scenarios
@load joinpath(OUTPUT_DIR, "results_base_analysis.jld2") results_base_analysis policy_configs_base

# Status quo
status_quo = results_base_analysis[:statusquo]

cs_changes_base = calculate_cs_changes(
    results_base_analysis, status_quo, params;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)
display_cs_changes(cs_changes_base;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIOS: CONSUMER SURPLUS CHANGES (vs Status Quo, billion \$)")

ps_land_base = calculate_ps_land_changes(
    results_base_analysis, status_quo, params;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)
display_ps_land_changes(ps_land_base;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIOS: LAND PRODUCER SURPLUS CHANGES")

gr_changes_base = calculate_gr_changes(
    results_base_analysis;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)
display_gr_changes(gr_changes_base;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIOS: GOVERNMENT REVENUE CHANGES")

SCC = 190
env_benefits_base = calculate_environmental_benefit(
    results_base_analysis, status_quo, SCC;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)
display_environmental_benefits(env_benefits_base, SCC;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIOS: ENVIRONMENTAL BENEFITS (vs Status Quo)")

welfare_summary_base = calculate_total_welfare(
    cs_changes_base, ps_land_base, gr_changes_base, env_benefits_base;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)
display_welfare_summary(welfare_summary_base;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIOS: WELFARE SUMMARY (vs Status Quo)")

aac_results_base = calculate_average_abatement_cost(
    welfare_summary_base, results_base_analysis, status_quo;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)
display_aac_analysis(aac_results_base;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIOS: AVERAGE ABATEMENT COST")

@save joinpath(OUTPUT_DIR, "results_base_welfare.jld2") results_base_analysis policy_configs_base status_quo cs_changes_base ps_land_base gr_changes_base env_benefits_base welfare_summary_base aac_results_base
println("✓ Saved results_base_welfare.jld2")

# 2) Target SAF Welfare Analysis (3B and 5B)


const TARGET_SAF_VALUES = [3.0, 5.0]

for target_saf in TARGET_SAF_VALUES
    suffix = target_saf == 3.0 ? "" : "_$(Int(target_saf))"

    println("\n" * "="^130)
    println("TARGET SAF = $(target_saf)B WELFARE ANALYSIS")
    println("="^130)

    # Load
    if target_saf == 3.0
        @load joinpath(OUTPUT_DIR, "results_equivalent_analysis.jld2") results_equivalent_analysis policy_configs_target equivalent_policies target_saf
    else
        file_data = load(joinpath(OUTPUT_DIR, "results_equivalent_analysis_5.jld2"))
        results_equivalent_analysis = file_data["results_equivalent_analysis"]
        policy_configs_target = file_data["policy_configs_target"]
        equivalent_policies = file_data["equivalent_policies"]
        target_saf = file_data["target_saf"]
    end
    println("✓ Loaded results_equivalent_analysis$(suffix).jld2")

    # Welfare analysis
    cs_changes_equivalent = calculate_cs_changes(
        results_equivalent_analysis, status_quo, params;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
    )
    display_cs_changes(cs_changes_equivalent;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
        title="EQUIVALENT SCENARIOS: CONSUMER SURPLUS CHANGES (Target SAF = $(target_saf)B, billion \$)")

    ps_land_equivalent = calculate_ps_land_changes(
        results_equivalent_analysis, status_quo, params;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
    )
    display_ps_land_changes(ps_land_equivalent;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
        title="EQUIVALENT SCENARIOS: LAND PRODUCER SURPLUS CHANGES (Target SAF = $(target_saf)B)")

    gr_changes_equivalent = calculate_gr_changes(
        results_equivalent_analysis;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
    )
    display_gr_changes(gr_changes_equivalent;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
        title="EQUIVALENT SCENARIOS: GOVERNMENT REVENUE CHANGES (Target SAF = $(target_saf)B)")

    env_benefits_equivalent = calculate_environmental_benefit(
        results_equivalent_analysis, status_quo, SCC;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
    )
    display_environmental_benefits(env_benefits_equivalent, SCC;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
        title="EQUIVALENT SCENARIOS: ENVIRONMENTAL BENEFITS (Target SAF = $(target_saf)B)")

    welfare_summary_equivalent = calculate_total_welfare(
        cs_changes_equivalent, ps_land_equivalent, gr_changes_equivalent, env_benefits_equivalent;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
    )
    display_welfare_summary(welfare_summary_equivalent;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
        title="EQUIVALENT SCENARIOS: WELFARE SUMMARY (Target SAF = $(target_saf)B)")

    aac_results_equivalent = calculate_average_abatement_cost(
        welfare_summary_equivalent, results_equivalent_analysis, status_quo;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
    )
    display_aac_analysis(aac_results_equivalent;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
        title="EQUIVALENT SCENARIOS: AVERAGE ABATEMENT COST (Target SAF = $(target_saf)B)")

    # Save
    @save joinpath(OUTPUT_DIR, "results_target_welfare$(suffix).jld2") results_equivalent_analysis policy_configs_target status_quo cs_changes_equivalent ps_land_equivalent gr_changes_equivalent env_benefits_equivalent target_saf welfare_summary_equivalent aac_results_equivalent
    println("✓ Saved results_target_welfare$(suffix).jld2")
end