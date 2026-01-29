# welfare.jl
using JLD2
using DataFrames
using Printf

# SAFModel에서 params 불러오기 (coefficients 필요)
include(joinpath(@__DIR__, "SAFModel.jl"))
using .SAFModel

cd(@__DIR__)  # 작업 디렉토리 설정

# =================================================================================
# LOAD RESULTS WITH IMPLICIT TAXES
# =================================================================================

println("\n" * "="^80)
println("WELFARE ANALYSIS")
println("="^80)

# Base scenarios 불러오기
@load "results_base_implicit_tax.jld2" results_base_implicit_tax policy_configs_base
println("✓ Loaded base results with implicit taxes")

# Target scenarios 불러오기
@load "results_target_implicit_tax.jld2" results_target_implicit_tax policy_configs_target equivalent_policies target_saf
println("✓ Loaded target results with implicit taxes")

# Base: Status quo
status_quo = results_base_implicit_tax[:statusquo]
println("✓ Extracted status quo as baseline")
sq_x = status_quo.x          # 소비
sq_q = status_quo.q          # 생산
sq_p_c = status_quo.p_c      # 소비재 가격
sq_p_f = status_quo.p_f      # 원료 가격
sq_l_n = status_quo.l_n      # 일반 토지
sq_l_cs = status_quo.l_cs    # CS 토지


# =====================
# Welfare Analysis Functions
# =====================

# Consumer Surplus Change Calculations
clean_small(val, threshold=1e-10) = abs(val) < threshold ? 0.0 : val

# Calculate consumer surplus change for a single good
function calc_cs_change(A, k, x_policy, x_sq, p_policy, p_sq)
    return (A / (k + 1)) * (x_policy^(k + 1) - x_sq^(k + 1)) - p_policy * x_policy + p_sq * x_sq
end

"""
Calculate consumer surplus changes for all goods
"""
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
            ) / 1000 # Convert to billion dollars
        )

        cs_changes[:total] = sum(v for (k, v) in cs_changes if k != :total)
        cs_changes_all[scenario] = cs_changes
    end

    return cs_changes_all
end

"""
Create consumer surplus change table
"""
function make_cs_change_table(cs_changes_all; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(cs_changes_all)) : scenarios
    # Remove statusquo if present
    scenario_list = filter(s -> s != :statusquo, scenario_list)

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
            # Clean small values
            value = clean_small(cs_changes_all[scenario][good_key])
            push!(df[!, String(scenario)], value)
        end
    end

    return df
end

"""
Display consumer surplus changes
"""
function display_cs_changes(cs_changes_all; scenarios=nothing, title="CONSUMER SURPLUS CHANGES (billion \$)")
    println("\n" * "="^130)
    println(title)
    println("="^130)

    cs_table = make_cs_change_table(cs_changes_all; scenarios=scenarios)
    show(cs_table, allrows=true)

    println("\n" * "="^130)
end

# =================================================================================
# RUN CONSUMER SURPLUS ANALYSIS
# =================================================================================

# Base scenarios
cs_changes_base = calculate_cs_changes(
    results_base_implicit_tax,
    status_quo,
    params;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_cs_changes(
    cs_changes_base;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIOS: CONSUMER SURPLUS CHANGES (vs Status Quo, billion \$)"
)

# Target scenarios
cs_changes_target = calculate_cs_changes(
    results_target_implicit_tax,
    status_quo,
    params;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_cs_changes(
    cs_changes_target;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="TARGET SCENARIOS: CONSUMER SURPLUS CHANGES (Target SAF = $(target_saf) billion gallons, billion \$)"
)



# =================================================================================
# Producer Surplus Changes
# =================================================================================

"""
Calculate scarcity rents from binding common constraints
"""
function calc_scarcity_rents(fuel, result, params)
    duals = result.duals
    scarcity_rent = 0.0

    theta = 0.125  # RFS D6 mandate share

    # =========================================================================
    # 1. Road RFS D6
    # =========================================================================
    if fuel == :gasoline || fuel == :diesel
        scarcity_rent += duals.λ_rfs * theta
    elseif fuel == :ethanol
        scarcity_rent += duals.λ_rfs * (-1.0)
    elseif fuel == :biodiesel_soy || fuel == :biodiesel_nonsoy
        scarcity_rent += duals.λ_rfs * (-1.5)
    elseif fuel == :rd_soy || fuel == :rd_nonsoy
        scarcity_rent += duals.λ_rfs * (-1.7)
    end

    # =========================================================================
    # 2. Blend Wall - Ethanol
    # =========================================================================
    if fuel == :gasoline
        scarcity_rent += duals.λ_blendwall_ethanol * (-0.1)
    elseif fuel == :ethanol
        scarcity_rent += duals.λ_blendwall_ethanol * 0.9
    end

    # =========================================================================
    # 3. Blend Wall - Biodiesel
    # =========================================================================
    if fuel == :diesel
        scarcity_rent += duals.λ_blendwall_biodiesel * (-0.05)
    elseif fuel == :biodiesel_soy || fuel == :biodiesel_nonsoy
        scarcity_rent += duals.λ_blendwall_biodiesel * 0.95
    end

    # =========================================================================
    # 4. Non-soy Capacity
    # =========================================================================
    if fuel in [:saf_hefa_nonsoy, :biodiesel_nonsoy, :rd_nonsoy]
        alpha = params.coeff.alpha

        if fuel == :saf_hefa_nonsoy
            scarcity_rent += duals.λ_nonsoy_capacity * alpha[:saf_hefa_nonsoy]
        elseif fuel == :biodiesel_nonsoy
            scarcity_rent += duals.λ_nonsoy_capacity * alpha[:biodiesel_nonsoy]
        elseif fuel == :rd_nonsoy
            scarcity_rent += duals.λ_nonsoy_capacity * alpha[:rd_nonsoy]
        end
    end

    return scarcity_rent
end

"""
Calculate producer net price (should equal c0)
"""
function calc_producer_net_price(fuel, result, params)
    r = params.coeff.r
    beta = params.coeff.beta

    # Energy-adjusted consumer price
    price_per_unit = if fuel == :jet_fuel
        r[:jet_fuel] * result.p_c[:avi]
    elseif fuel in [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
        r[:jet_fuel] * beta[(:saf, :jet_fuel)] * result.p_c[:avi]
    elseif fuel == :gasoline
        r[:gasoline] * result.p_c[:gas]
    elseif fuel == :ethanol
        r[:gasoline] * beta[(:ethanol, :gasoline)] * result.p_c[:gas]
    elseif fuel == :diesel
        r[:diesel] * result.p_c[:die]
    elseif fuel in [:biodiesel_soy, :biodiesel_nonsoy]
        r[:diesel] * beta[(:biodiesel, :diesel)] * result.p_c[:die]
    elseif fuel in [:rd_soy, :rd_nonsoy]
        r[:diesel] * beta[(:rd, :diesel)] * result.p_c[:die]
    else
        0.0
    end

    # Policy-specific implicit tax
    implicit_taxes = get(result, :implicit_taxes, Dict())
    implicit_tax = get(get(implicit_taxes, fuel, Dict()), :total, 0.0)

    # Common constraint scarcity rents
    scarcity_rent = calc_scarcity_rents(fuel, result, params)

    # Non-soy feedstock cost (exogenous price)
    feedstock_cost = 0.0
    if fuel in [:saf_hefa_nonsoy, :biodiesel_nonsoy, :rd_nonsoy]
        alpha = params.coeff.alpha
        nonsoy_feedstock_price = 0.48  # $/lb (exogenous)

        if fuel == :saf_hefa_nonsoy
            feedstock_cost = alpha[:saf_hefa_nonsoy] * nonsoy_feedstock_price
        elseif fuel == :biodiesel_nonsoy
            feedstock_cost = alpha[:biodiesel_nonsoy] * nonsoy_feedstock_price
        elseif fuel == :rd_nonsoy
            feedstock_cost = alpha[:rd_nonsoy] * nonsoy_feedstock_price
        end
    end

    # ⭐ HEFA SAF premium (only for SAF HEFA)
    hefa_saf_premium = 0.0
    if fuel == :saf_hefa_nonsoy
        hefa_saf_premium = 0.08  # $/gallon (correct value!)
    end

    # Producer net price (should equal c0)
    return price_per_unit - implicit_tax - scarcity_rent - feedstock_cost - hefa_saf_premium
end

"""
Calculate producer surplus for a single fuel
"""
function calc_fuel_ps(p_net, q, c0, c1, c2, v)
    revenue = p_net * q
    cost = c0 * q + (c1 / 2) * q^2
    if q > v
        cost += (c2 / 3) * (q - v)^3
    end
    return revenue - cost
end

"""
Calculate non-agricultural producer surplus changes
"""
function calculate_ps_nonag_changes(solutions, solution_sq, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    supply = params.supply.fuel

    nonag_fuels = Dict(
        :jet_fuel => :jet_fuel,
        :gasoline => :gasoline,
        :diesel => :diesel,
        :saf_hefa_nonsoy => :saf_hefa_shared,
        :biodiesel_nonsoy => :biodiesel_nonsoy,
        :rd_nonsoy => :saf_hefa_shared
    )

    # Status quo PS
    ps_sq_by_fuel = Dict()
    for (fuel, cost_key) in nonag_fuels
        fc = supply[cost_key]
        p_net_sq = calc_producer_net_price(fuel, solution_sq, params)
        q_sq = solution_sq.q[fuel]
        ps_sq = calc_fuel_ps(p_net_sq, q_sq, fc.c0, fc.c1, fc.c2, fc.v)
        scarcity_rent_sq = calc_scarcity_rents(fuel, solution_sq, params)

        ps_sq_by_fuel[fuel] = (
            ps=ps_sq,
            p_net=p_net_sq,
            q=q_sq,
            c0=fc.c0,
            scarcity_rent=scarcity_rent_sq
        )
    end

    # Policy PS changes
    results = Dict()
    for scenario in scenario_list
        sol_policy = solutions[scenario]
        ps_by_fuel = Dict()

        for (fuel, cost_key) in nonag_fuels
            fc = supply[cost_key]

            p_net_policy = calc_producer_net_price(fuel, sol_policy, params)
            q_policy = sol_policy.q[fuel]
            ps_policy = calc_fuel_ps(p_net_policy, q_policy, fc.c0, fc.c1, fc.c2, fc.v)
            scarcity_rent_policy = calc_scarcity_rents(fuel, sol_policy, params)

            sq_data = ps_sq_by_fuel[fuel]
            ps_change = ps_policy - sq_data.ps

            ps_by_fuel[fuel] = (
                ps_change=clean_small(ps_change),
                ps_policy=ps_policy,
                ps_sq=sq_data.ps,
                p_net_policy=p_net_policy,
                p_net_sq=sq_data.p_net,
                q_policy=q_policy,
                q_sq=sq_data.q,
                c0=fc.c0,
                scarcity_rent_policy=scarcity_rent_policy,
                scarcity_rent_sq=sq_data.scarcity_rent
            )
        end

        total_ps_change = sum(data.ps_change for data in values(ps_by_fuel))

        results[scenario] = (
            by_fuel=ps_by_fuel,
            total=clean_small(total_ps_change)
        )
    end

    return results, ps_sq_by_fuel
end

"""
Display non-ag PS changes
"""
function display_ps_nonag_changes(ps_nonag_all, ps_sq_by_fuel; scenarios=nothing,
    title="NON-AG PRODUCER SURPLUS CHANGES (billion \$)")
    scenario_list = isnothing(scenarios) ? collect(keys(ps_nonag_all)) : scenarios

    println("\n" * "="^130)
    println(title)
    println("="^130)

    # Summary table
    df = DataFrame(Scenario=Symbol[], JetFuel=Float64[], Gasoline=Float64[],
        Diesel=Float64[], NonSoyHEFA=Float64[], NonSoyBD=Float64[],
        NonSoyRD=Float64[], Total=Float64[])

    for scenario in scenario_list
        ps = ps_nonag_all[scenario].by_fuel
        push!(df, (
            scenario,
            ps[:jet_fuel].ps_change,
            ps[:gasoline].ps_change,
            ps[:diesel].ps_change,
            ps[:saf_hefa_nonsoy].ps_change,
            ps[:biodiesel_nonsoy].ps_change,
            ps[:rd_nonsoy].ps_change,
            ps_nonag_all[scenario].total
        ))
    end

    show(df, allrows=true)

    # Verification
    println("\n\n--- STATUS QUO VERIFICATION: p_net should equal c0 ---")
    @printf("%-20s %12s %12s %12s %12s\n", "Fuel", "p_net_sq", "c0", "Diff", "PS_sq")
    println("-"^68)

    for fuel in [:jet_fuel, :gasoline, :diesel, :saf_hefa_nonsoy, :biodiesel_nonsoy, :rd_nonsoy]
        sq = ps_sq_by_fuel[fuel]
        diff = sq.p_net - sq.c0
        check = abs(diff) < 1e-6 ? "✓" : "✗"
        @printf("%-20s %12.4f %12.4f %12.2e %12.2f %s\n",
            fuel, sq.p_net, sq.c0, diff, sq.ps, check)
    end

    println("\n" * "="^130)
end

# Base
ps_nonag_base, ps_sq_by_fuel = calculate_ps_nonag_changes(
    results_base_implicit_tax,
    status_quo,
    params;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_ps_nonag_changes(
    ps_nonag_base,
    ps_sq_by_fuel;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="BASE SCENARIOS: NON-AG PRODUCER SURPLUS CHANGES"
)

# Target
ps_nonag_target, _ = calculate_ps_nonag_changes(
    results_target_implicit_tax,
    status_quo,
    params;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit]
)

display_ps_nonag_changes(
    ps_nonag_target,
    ps_sq_by_fuel;
    scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="TARGET SCENARIOS: NON-AG PRODUCER SURPLUS CHANGES"
)

println("\n" * "="^80)
println("WELFARE ANALYSIS COMPLETE")
println("="^80)


# =================================================================================


"""
Calculate government revenue change for a single scenario
"""
function calculate_gov_revenue_change(solution_policy, implicit_taxes_policy, scenario)

    # Only carbon tax and tax credit have explicit government revenue
    if !(scenario in [:carbontax, :taxcredit])
        return 0.0
    end

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    gov_revenue = 0.0

    if scenario == :carbontax
        # Carbon tax: all aviation fuels
        for fuel in AVIATION_FUELS
            t_i = implicit_taxes_policy[fuel][:carbon_tax]
            q_i = solution_policy.q[fuel]
            gov_revenue += t_i * q_i
        end

    elseif scenario == :taxcredit
        # Tax credit: eligible SAF only (already negative in implicit tax)
        ELIGIBLE_SAF = [:saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

        for saf in ELIGIBLE_SAF
            s_i = implicit_taxes_policy[saf][:tax_credit]  # Already negative
            q_i = solution_policy.q[saf]
            gov_revenue += s_i * q_i
        end
    end

    return gov_revenue
end

"""
Calculate government revenue changes for all scenarios
"""
function calculate_gov_revenue_changes(solutions, implicit_taxes_all; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    gov_revenue_changes = Dict()

    for scenario in scenario_list
        gov_revenue_changes[scenario] = calculate_gov_revenue_change(
            solutions[scenario],
            implicit_taxes_all[scenario],
            scenario
        )
    end

    return gov_revenue_changes
end

# Implicit taxes 먼저 계산 (이미 했음)
implicit_taxes = calculate_all_implicit_taxes(solutions, params, POLICY_MATRIX)

# Government revenue (implicit tax 재사용!)
gov_revenue_changes = calculate_gov_revenue_changes(solutions, implicit_taxes;
    scenarios=SCENARIOS)

"""
Display detailed price information for all fuels
"""
function display_fuel_prices(solutions, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    println("\n" * "="^130)
    println("DETAILED FUEL PRICES")
    println("="^130)

    for scenario in scenario_list
        println("\n--- Scenario: $scenario ---")
        sol = solutions[scenario]

        # Consumer prices
        println("\nConsumer Prices (per unit energy):")
        @printf("  Aviation (p_c[:avi]):    %.4f\n", sol.p_c[:avi])
        @printf("  Gasoline (p_c[:gas]):    %.4f\n", sol.p_c[:gas])
        @printf("  Diesel (p_c[:die]):      %.4f\n", sol.p_c[:die])

        # Producer prices (price per gallon)
        println("\nProducer Prices (per gallon):")

        r = params.coeff.r
        beta = params.coeff.beta

        # Aviation fuels
        println("  Aviation Fuels:")
        p_jet = r[:jet_fuel] * sol.p_c[:avi]
        p_saf = r[:jet_fuel] * beta[(:saf, :jet_fuel)] * sol.p_c[:avi]
        @printf("    Jet Fuel:              %.4f\n", p_jet)
        @printf("    SAF (all types):       %.4f\n", p_saf)

        # Road gasoline
        println("  Road Gasoline:")
        p_gas = r[:gasoline] * sol.p_c[:gas]
        p_eth = r[:gasoline] * beta[(:ethanol, :gasoline)] * sol.p_c[:gas]
        @printf("    Gasoline:              %.4f\n", p_gas)
        @printf("    Ethanol:               %.4f\n", p_eth)

        # Road diesel
        println("  Road Diesel:")
        p_die = r[:diesel] * sol.p_c[:die]
        p_bio = r[:diesel] * beta[(:biodiesel, :diesel)] * sol.p_c[:die]
        p_rd = r[:diesel] * beta[(:rd, :diesel)] * sol.p_c[:die]
        @printf("    Diesel:                %.4f\n", p_die)
        @printf("    Biodiesel:             %.4f\n", p_bio)
        @printf("    Renewable Diesel:      %.4f\n", p_rd)

        # Feedstock prices
        println("\nFeedstock Prices:")
        @printf("  Conv Corn (per bu):      %.4f\n", sol.p_f[:feedstock_corn_n])
        @printf("  CS Corn (per bu):        %.4f\n", sol.p_f[:feedstock_corn_cs])
        @printf("  Conv Soy (per lb oil):   %.4f\n", sol.p_f[:feedstock_soy_n])
        @printf("  CS Soy (per lb oil):     %.4f\n", sol.p_f[:feedstock_soy_cs])
        @printf("  Soymeal (per ton):       %.4f\n", sol.p_c[:soymeal])

        println("-"^130)
    end

    println("="^130)
end

"""
Create price comparison table for specific fuels
"""
function make_fuel_price_table(solutions, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    r = params.coeff.r
    beta = params.coeff.beta

    df = DataFrame(Scenario=String[])

    # Add columns for each fuel
    fuels = [
        "Jet_Fuel", "SAF", "Gasoline", "Ethanol",
        "Diesel", "Biodiesel", "RD"
    ]

    for fuel in fuels
        df[!, fuel] = Float64[]
    end

    for scenario in scenario_list
        sol = solutions[scenario]

        row = [
            String(scenario),
            r[:jet_fuel] * sol.p_c[:avi],  # Jet fuel
            r[:jet_fuel] * beta[(:saf, :jet_fuel)] * sol.p_c[:avi],  # SAF
            r[:gasoline] * sol.p_c[:gas],  # Gasoline
            r[:gasoline] * beta[(:ethanol, :gasoline)] * sol.p_c[:gas],  # Ethanol
            r[:diesel] * sol.p_c[:die],  # Diesel
            r[:diesel] * beta[(:biodiesel, :diesel)] * sol.p_c[:die],  # Biodiesel
            r[:diesel] * beta[(:rd, :diesel)] * sol.p_c[:die]  # RD
        ]

        push!(df, row)
    end

    return df
end

"""
Compare prices between two scenarios
"""
function compare_prices(sol_policy, sol_sq, params, scenario_name)
    println("\n" * "="^130)
    println("PRICE COMPARISON: $scenario_name vs Status Quo")
    println("="^130)

    r = params.coeff.r
    beta = params.coeff.beta

    # Helper function to calculate and display price change
    function show_price_change(name, p_policy, p_sq)
        change = p_policy - p_sq
        pct = 100 * change / p_sq
        @printf("%-30s  SQ: %8.4f  Policy: %8.4f  Change: %8.4f (%6.2f%%)\n",
            name, p_sq, p_policy, change, pct)
    end

    println("\nConsumer Prices (per unit energy):")
    show_price_change("Aviation", sol_policy.p_c[:avi], sol_sq.p_c[:avi])
    show_price_change("Gasoline", sol_policy.p_c[:gas], sol_sq.p_c[:gas])
    show_price_change("Diesel", sol_policy.p_c[:die], sol_sq.p_c[:die])

    println("\nProducer Prices - Aviation (per gallon):")
    show_price_change("Jet Fuel",
        r[:jet_fuel] * sol_policy.p_c[:avi],
        r[:jet_fuel] * sol_sq.p_c[:avi])
    show_price_change("SAF",
        r[:jet_fuel] * beta[(:saf, :jet_fuel)] * sol_policy.p_c[:avi],
        r[:jet_fuel] * beta[(:saf, :jet_fuel)] * sol_sq.p_c[:avi])

    println("\nProducer Prices - Road Gasoline (per gallon):")
    show_price_change("Gasoline",
        r[:gasoline] * sol_policy.p_c[:gas],
        r[:gasoline] * sol_sq.p_c[:gas])
    show_price_change("Ethanol",
        r[:gasoline] * beta[(:ethanol, :gasoline)] * sol_policy.p_c[:gas],
        r[:gasoline] * beta[(:ethanol, :gasoline)] * sol_sq.p_c[:gas])

    println("\nProducer Prices - Road Diesel (per gallon):")
    show_price_change("Diesel",
        r[:diesel] * sol_policy.p_c[:die],
        r[:diesel] * sol_sq.p_c[:die])
    show_price_change("Biodiesel",
        r[:diesel] * beta[(:biodiesel, :diesel)] * sol_policy.p_c[:die],
        r[:diesel] * beta[(:biodiesel, :diesel)] * sol_sq.p_c[:die])
    show_price_change("Renewable Diesel",
        r[:diesel] * beta[(:rd, :diesel)] * sol_policy.p_c[:die],
        r[:diesel] * beta[(:rd, :diesel)] * sol_sq.p_c[:die])

    println("\nFeedstock Prices:")
    show_price_change("Conv Corn (per bu)",
        sol_policy.p_f[:feedstock_corn_n],
        sol_sq.p_f[:feedstock_corn_n])
    show_price_change("CS Corn (per bu)",
        sol_policy.p_f[:feedstock_corn_cs],
        sol_sq.p_f[:feedstock_corn_cs])
    show_price_change("Conv Soy (per lb)",
        sol_policy.p_f[:feedstock_soy_n],
        sol_sq.p_f[:feedstock_soy_n])
    show_price_change("CS Soy (per lb)",
        sol_policy.p_f[:feedstock_soy_cs],
        sol_sq.p_f[:feedstock_soy_cs])

    println("="^130)
end

# 사용 예시:
# 모든 시나리오의 가격 출력
display_fuel_prices(solutions, params; scenarios=SCENARIOS)

# 테이블 형태로 보기
price_table = make_fuel_price_table(solutions, params; scenarios=SCENARIOS)
show(price_table, allrows=true)

# 특정 정책과 status quo 비교
compare_prices(solutions[:carbontax], solutions[:statusquo], params, "Carbon Tax")

# Jet fuel c0 확인
println("Jet fuel c0: ", params.supply.fuel[:jet_fuel].c0)
println("Status quo jet fuel price: ", solutions[:statusquo].q[:jet_fuel] |> q -> 2.338)
println("Carbon tax jet fuel price: ", solutions[:carbontax].q[:jet_fuel] |> q -> 5.227)

# Implicit tax 확인
println("\nCarbon tax implicit tax for jet fuel: ")
println(all_implicit_taxes[:carbontax][:jet_fuel][:total])

# 디버깅 함수
function debug_fuel_ps(scenario, fuel, solutions, solution_sq, params, all_implicit_taxes)
    sol = solutions[scenario]

    # Fuel parameters
    f_key = (fuel in [:saf_hefa_nonsoy, :rd_nonsoy]) ? :saf_hefa_shared : fuel
    f_params = params.supply.fuel[f_key]

    c0 = f_params.c0
    c1 = f_params.c1
    c2 = f_params.c2
    v = f_params.v

    # Quantities
    Q_sq = solution_sq.q[fuel]
    Q_policy = sol.q[fuel]

    # Prices
    P_sq = get_fuel_price_per_unit(fuel, solution_sq, params)
    P_policy = get_fuel_price_per_unit(fuel, sol, params)

    # Adjustments
    adj_sq = 0.0
    policy_taxes = all_implicit_taxes[scenario]
    adj_policy = haskey(policy_taxes, fuel) ? policy_taxes[fuel][:total] : 0.0

    # Net prices
    net_P_sq = P_sq - adj_sq
    net_P_policy = P_policy - adj_policy

    # Calculate TVC
    function calc_tvc(Q)
        tvc = c0 * Q + 0.5 * c1 * Q^2
        if Q > v
            tvc += (c2 / 3) * (Q - v)^3
        end
        return tvc
    end

    tvc_sq = calc_tvc(Q_sq)
    tvc_policy = calc_tvc(Q_policy)

    # Calculate PS
    ps_sq = net_P_sq * Q_sq - tvc_sq
    ps_policy = net_P_policy * Q_policy - tvc_policy
    ps_change = ps_policy - ps_sq

    println("\n" * "="^80)
    println("DEBUG: $fuel in $scenario")
    println("="^80)
    println("Cost parameters:")
    @printf("  c0 = %.4f, c1 = %.4f, c2 = %.4f, v = %.4f\n", c0, c1, c2, v)

    println("\nStatus Quo:")
    @printf("  Q = %.4f\n", Q_sq)
    @printf("  P = %.4f\n", P_sq)
    @printf("  adj = %.4f\n", adj_sq)
    @printf("  net_P = %.4f (P - adj)\n", net_P_sq)
    @printf("  TVC = %.4f\n", tvc_sq)
    @printf("  PS = %.4f (net_P * Q - TVC)\n", ps_sq)
    @printf("  Q > v? %s\n", Q_sq > v ? "YES" : "NO")

    println("\nPolicy:")
    @printf("  Q = %.4f (change: %.4f)\n", Q_policy, Q_policy - Q_sq)
    @printf("  P = %.4f (change: %.4f)\n", P_policy, P_policy - P_sq)
    @printf("  adj = %.4f\n", adj_policy)
    @printf("  net_P = %.4f (P - adj)\n", net_P_policy)
    @printf("  TVC = %.4f (change: %.4f)\n", tvc_policy, tvc_policy - tvc_sq)
    @printf("  PS = %.4f (net_P * Q - TVC)\n", ps_policy)
    @printf("  Q > v? %s\n", Q_policy > v ? "YES" : "NO")

    println("\nResult:")
    @printf("  PS_change = %.4f\n", ps_change)
    @printf("  net_P_sq vs net_P_policy: %.4f vs %.4f (diff: %.4f)\n",
        net_P_sq, net_P_policy, net_P_policy - net_P_sq)
    @printf("  c0 vs net_P_sq: %.4f vs %.4f (diff: %.4f)\n",
        c0, net_P_sq, net_P_sq - c0)

    println("="^80)

    return ps_change
end

# 사용
debug_fuel_ps(:carbontax, :jet_fuel, solutions, solutions[:statusquo], params, all_implicit_taxes)
debug_fuel_ps(:carbontax, :gasoline, solutions, solutions[:statusquo], params, all_implicit_taxes)
debug_fuel_ps(:carbontax, :diesel, solutions, solutions[:statusquo], params, all_implicit_taxes)

# Equivalent policies를 위한 디버깅 코드

"""
Debug function for equivalent policies
"""
function debug_equivalent_fuel_ps(scenario, fuel, equivalent_solutions, solution_sq,
    params, equivalent_implicit_taxes, equivalent_policies)
    sol = equivalent_solutions[scenario]

    # Fuel parameters
    f_key = (fuel in [:saf_hefa_nonsoy, :rd_nonsoy]) ? :saf_hefa_shared : fuel
    f_params = params.supply.fuel[f_key]

    c0 = f_params.c0
    c1 = f_params.c1
    c2 = f_params.c2
    v = f_params.v

    # Quantities
    Q_sq = solution_sq.q[fuel]
    Q_policy = sol.q[fuel]

    # Prices
    P_sq = get_fuel_price_per_unit(fuel, solution_sq, params)
    P_policy = get_fuel_price_per_unit(fuel, sol, params)

    # Adjustments
    adj_sq = 0.0
    policy_taxes = equivalent_implicit_taxes[scenario]
    adj_policy = haskey(policy_taxes, fuel) ? policy_taxes[fuel][:total] : 0.0

    # Net prices
    net_P_sq = P_sq - adj_sq
    net_P_policy = P_policy - adj_policy

    # Calculate TVC
    function calc_tvc(Q)
        tvc = c0 * Q + 0.5 * c1 * Q^2
        if Q > v
            tvc += (c2 / 3) * (Q - v)^3
        end
        return tvc
    end

    tvc_sq = calc_tvc(Q_sq)
    tvc_policy = calc_tvc(Q_policy)

    # Calculate PS
    ps_sq = net_P_sq * Q_sq - tvc_sq
    ps_policy = net_P_policy * Q_policy - tvc_policy
    ps_change = ps_policy - ps_sq

    # Get policy stringency
    config = equivalent_policies[scenario].config
    policy_param = if scenario == :carbontax
        "t = $(config.t)"
    elseif scenario == :rfs
        "θ_avi = $(config.θ_avi)"
    elseif scenario == :lcfs
        "σ = $(config.σ)"
    else
        "p = $(config.p)"
    end

    println("\n" * "="^100)
    println("DEBUG EQUIVALENT POLICY: $fuel in $scenario ($policy_param)")
    println("="^100)
    println("Cost parameters:")
    @printf("  c0 = %.4f, c1 = %.4f, c2 = %.4f, v = %.4f\n", c0, c1, c2, v)

    println("\nStatus Quo:")
    @printf("  Q = %.6f\n", Q_sq)
    @printf("  P = %.6f\n", P_sq)
    @printf("  adj = %.6f\n", adj_sq)
    @printf("  net_P = %.6f (P - adj)\n", net_P_sq)
    @printf("  Revenue = %.6f (net_P * Q)\n", net_P_sq * Q_sq)
    @printf("  TVC = %.6f\n", tvc_sq)
    @printf("  PS = %.6f (Revenue - TVC)\n", ps_sq)
    @printf("  Q > v? %s (v = %.4f)\n", Q_sq > v ? "YES" : "NO", v)
    if Q_sq > v
        @printf("    Q - v = %.6f\n", Q_sq - v)
        @printf("    Cubic term = %.6f\n", (c2 / 3) * (Q_sq - v)^3)
    end

    println("\nPolicy:")
    @printf("  Q = %.6f (change: %.6f, %%change: %.2f%%)\n",
        Q_policy, Q_policy - Q_sq, 100 * (Q_policy - Q_sq) / Q_sq)
    @printf("  P = %.6f (change: %.6f, %%change: %.2f%%)\n",
        P_policy, P_policy - P_sq, 100 * (P_policy - P_sq) / P_sq)
    @printf("  adj = %.6f\n", adj_policy)
    @printf("  net_P = %.6f (P - adj)\n", net_P_policy)
    @printf("  Revenue = %.6f (net_P * Q)\n", net_P_policy * Q_policy)
    @printf("  TVC = %.6f (change: %.6f)\n", tvc_policy, tvc_policy - tvc_sq)
    @printf("  PS = %.6f (Revenue - TVC)\n", ps_policy)
    @printf("  Q > v? %s (v = %.4f)\n", Q_policy > v ? "YES" : "NO", v)
    if Q_policy > v
        @printf("    Q - v = %.6f\n", Q_policy - v)
        @printf("    Cubic term = %.6f\n", (c2 / 3) * (Q_policy - v)^3)
    end

    println("\nComparison:")
    @printf("  ΔQ = %.6f (%.2f%%)\n", Q_policy - Q_sq, 100 * (Q_policy - Q_sq) / Q_sq)
    @printf("  ΔP = %.6f (%.2f%%)\n", P_policy - P_sq, 100 * (P_policy - P_sq) / P_sq)
    @printf("  Δnet_P = %.6f (%.6f)\n", net_P_policy - net_P_sq, net_P_policy - net_P_sq)
    @printf("  ΔTVC = %.6f\n", tvc_policy - tvc_sq)
    @printf("  ΔRevenue = %.6f\n", net_P_policy * Q_policy - net_P_sq * Q_sq)

    println("\nZero-profit check:")
    @printf("  c0 = %.6f\n", c0)
    @printf("  net_P_sq = %.6f (should ≈ c0)\n", net_P_sq)
    @printf("  net_P_policy = %.6f (should ≈ c0)\n", net_P_policy)
    @printf("  Difference (net_P_sq - c0) = %.10f\n", net_P_sq - c0)
    @printf("  Difference (net_P_policy - c0) = %.10f\n", net_P_policy - c0)

    println("\nResult:")
    @printf("  PS_change = %.6f billion \$\n", ps_change)

    # Breakdown
    println("\nPS Change Breakdown:")
    revenue_change = net_P_policy * Q_policy - net_P_sq * Q_sq
    tvc_change = tvc_policy - tvc_sq
    @printf("  Revenue change = %.6f\n", revenue_change)
    @printf("  TVC change = %.6f\n", tvc_change)
    @printf("  PS change = %.6f (Revenue change - TVC change)\n", revenue_change - tvc_change)
    @printf("  Check: %.6f (should match PS_change above)\n", ps_change)
    println("Applied Carbon Tax in Debug: ", config.t)

    println("="^100)

    return ps_change
end

"""
Debug all fuels for a scenario
"""
function debug_all_fuels_equiv(scenario, equivalent_solutions, solution_sq,
    params, equivalent_implicit_taxes, equivalent_policies)

    println("\n" * "█"^100)
    println("DEBUGGING SCENARIO: $scenario")
    println("█"^100)

    target_fuels = [:jet_fuel, :gasoline, :diesel,
        :saf_hefa_nonsoy, :biodiesel_nonsoy, :rd_nonsoy]

    results = Dict()
    for fuel in target_fuels
        results[fuel] = debug_equivalent_fuel_ps(scenario, fuel, equivalent_solutions,
            solution_sq, params,
            equivalent_implicit_taxes,
            equivalent_policies)
    end

    println("\n" * "─"^100)
    println("SUMMARY for $scenario:")
    println("─"^100)
    for fuel in target_fuels
        @printf("  %-25s: PS change = %10.4f billion \$\n", fuel, results[fuel])
    end
    println("─"^100)

    return results
end

"""
Compare all equivalent scenarios
"""
function debug_all_equiv_scenarios(equivalent_solutions, solution_sq, params,
    equivalent_implicit_taxes, equivalent_policies)

    scenarios = [:carbontax, :rfs, :lcfs, :taxcredit]

    all_results = Dict()
    for scenario in scenarios
        all_results[scenario] = debug_all_fuels_equiv(scenario, equivalent_solutions,
            solution_sq, params,
            equivalent_implicit_taxes,
            equivalent_policies)
    end

    # Cross-scenario comparison
    println("\n\n" * "█"^100)
    println("CROSS-SCENARIO COMPARISON")
    println("█"^100)

    target_fuels = [:jet_fuel, :gasoline, :diesel,
        :saf_hefa_nonsoy, :biodiesel_nonsoy, :rd_nonsoy]

    for fuel in target_fuels
        println("\n" * "─"^100)
        println("Fuel: $fuel")
        println("─"^100)
        @printf("%-15s %15s %15s %15s %15s\n",
            "Scenario", "PS Change", "Q_policy", "P_policy", "adj_policy")
        println("-"^100)

        for scenario in scenarios
            sol = equivalent_solutions[scenario]
            Q = sol.q[fuel]
            P = get_fuel_price_per_unit(fuel, sol, params)

            policy_taxes = equivalent_implicit_taxes[scenario]
            adj = haskey(policy_taxes, fuel) ? policy_taxes[fuel][:total] : 0.0

            ps_change = all_results[scenario][fuel]

            @printf("%-15s %15.6f %15.6f %15.6f %15.6f\n",
                scenario, ps_change, Q, P, adj)
        end
    end

    println("█"^100)

    return all_results
end

# 사용 예시:

# 1. 특정 시나리오의 특정 연료 디버깅
debug_equivalent_fuel_ps(:carbontax, :jet_fuel, equivalent_solutions,
    solutions[:statusquo], params,
    equivalent_implicit_taxes, equivalent_policies)

# 2. 특정 시나리오의 모든 연료 디버깅
debug_all_fuels_equiv(:carbontax, equivalent_solutions, solutions[:statusquo],
    params, equivalent_implicit_taxes, equivalent_policies)

# 3. 모든 시나리오 비교
all_debug_results = debug_all_equiv_scenarios(equivalent_solutions, solutions[:statusquo],
    params, equivalent_implicit_taxes,
    equivalent_policies)