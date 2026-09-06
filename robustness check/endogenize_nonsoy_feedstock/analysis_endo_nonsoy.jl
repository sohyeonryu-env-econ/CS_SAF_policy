# analysis_endo_nonsoy.jl

module AnalysisEndoNonsoy
import Main.ModelEndoNonsoy: build_unified_model, extract_solution, tax_credit_rate, FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS
using JLD2
using DataFrames
using Printf
using JuMP

const SCC = 190
const POL = [:carbontax, :rfs, :lcfs, :taxcredit]
const SQ_CONFIG = (t=0.0, θ_avi=0.0, σ=0.0, p=0.0, carbon_tax_scope=:aviation, use_ci_threshold=false, recognize_cs=true)

# =====================
# 1) Implicit Tax/Subsidy Calculation
# =====================

# calculate implicit tax and subsidy
function calculate_implicit_taxes(solution, params, config)
    if isnothing(solution)
        return nothing
    end

    # Tuple unpacking
    t = config.t
    θ_avi = config.θ_avi
    σ = config.σ
    p = config.p
    use_ci_threshold = config.use_ci_threshold
    recognize_cs = config.recognize_cs

    # Get coefficients
    delta = params.coeff.delta
    delta_mj = params.coeff.delta_mj
    baselineCI = params.coeff.baselineCI

    # Get dual variables
    duals = solution.duals
    γ_avi = duals.λ_rfs_avi  # RFS aviation dual
    μ = duals.λ_lcfs          # LCFS dual

    # Aviation fuels
    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    CS_PATHS = [:saf_atj_cs, :saf_hefa_cs]

    # Eligibility must mirror build_unified_model's policy_adjustment exactly; these
    # implicit taxes feed calculate_gov_revenue_change, so a mismatch means the
    # equilibrium is solved under one set of policy prices and scored under another.
    saf_eligible(g) = (!use_ci_threshold || delta[g] <= 0.5 * delta[:jet_fuel]) &&
                      (recognize_cs || g ∉ CS_PATHS)

    tax_base(g) = (g == :saf_atj_cs && !recognize_cs) ? delta[:saf_atj_conv] :
                  (g == :saf_hefa_cs && !recognize_cs) ? delta[:saf_hefa_conv] :
                  delta[g]

    # Initialize results
    implicit_tax = Dict(g => Dict(
        :carbon_tax => 0.0,
        :rfs_avi => 0.0,
        :lcfs => 0.0,
        :tax_credit => 0.0,
        :total => 0.0
    ) for g in AVIATION_FUELS)

    for g in AVIATION_FUELS
        # 1. Carbon tax component
        implicit_tax[g][:carbon_tax] = t * tax_base(g)

        # 2. RFS aviation component
        if g == :jet_fuel
            implicit_tax[g][:rfs_avi] = γ_avi * θ_avi
        else  # SAF goods
            implicit_tax[g][:rfs_avi] = saf_eligible(g) ? -γ_avi * 1.6 : 0.0
        end

        # 3. LCFS component. Ineligible SAF sits outside the LCFS pool entirely.
        implicit_tax[g][:lcfs] =
            g == :jet_fuel ? -μ * ((1 - σ) * delta[:jet_fuel] - delta[g]) :
            saf_eligible(g) ? -μ * ((1 - σ) * delta[:jet_fuel] - delta[g]) : 0.0

        # 4. Tax credit component (eligible SAF only)
        if g != :jet_fuel
            implicit_tax[g][:tax_credit] = saf_eligible(g) ?
                                           -tax_credit_rate(delta_mj[g], baselineCI, p) : 0.0
        end

        # Total
        implicit_tax[g][:total] = (
            implicit_tax[g][:carbon_tax] +
            implicit_tax[g][:rfs_avi] +
            implicit_tax[g][:lcfs] +
            implicit_tax[g][:tax_credit]
        )
    end

    return implicit_tax
end


# Calculate implicit taxes for all solutions
function calculate_all_implicit_taxes(solutions, params, policy_configs)
    implicit_taxes = Dict()
    for (scenario, solution) in solutions
        if !isnothing(solution)
            config = getproperty(policy_configs, scenario)
            implicit_taxes[scenario] = calculate_implicit_taxes(solution, params, config)
        end
    end
    return implicit_taxes
end

# =================================================================================
# 2. Emissions
# =================================================================================

function calculate_emissions_detail(solution, params)
    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel,
        :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    delta = params.coeff.delta

    avi = sum(delta[g] * solution.q[g] for g in AVIATION_FUELS)
    road = sum(delta[g] * solution.q[g] for g in ROAD_FUELS)
    food = delta[:corn] * (solution.x[:corn] - solution.ddgs) +
           delta[:soyoil] * solution.x[:soyoil]

    return (
        aviation=avi, road=road, food=food, total=avi + road + food,
        by_fuel=Dict(g => delta[g] * solution.q[g] for g in union(AVIATION_FUELS, ROAD_FUELS)),
        by_food=Dict(:corn => delta[:corn] * (solution.x[:corn] - solution.ddgs),
            :soyoil => delta[:soyoil] * solution.x[:soyoil])
    )
end

# =================================================================================
# 3. Table Makers
# =================================================================================

function make_implicit_tax_table(implicit_taxes, params; scenarios=nothing, exclude_statusquo=true)
    scenario_list = isnothing(scenarios) ? collect(keys(implicit_taxes)) : scenarios
    exclude_statusquo && (scenario_list = filter(s -> s != :statusquo, scenario_list))
    labels = params.meta[:process_labels]

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    scenario_tax_type = Dict(:statusquo => :total, :carbontax => :carbon_tax,
        :rfs => :rfs_avi, :lcfs => :lcfs, :taxcredit => :tax_credit)

    df = DataFrame(Fuel=String[])
    for s in scenario_list
        df[!, String(s)] = Float64[]
    end

    for g in AVIATION_FUELS
        push!(df.Fuel, labels[g])
        for s in scenario_list
            v = (haskey(implicit_taxes, s) && !isnothing(implicit_taxes[s])) ?
                implicit_taxes[s][g][get(scenario_tax_type, s, :total)] : 0.0
            push!(df[!, String(s)], v)
        end
    end
    return df
end

function make_production_table(solutions, params; scenarios=nothing)
    labels = params.meta[:process_labels]
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(Product=String[])
    for s in scenario_list
        df[!, s] = Float64[]
    end

    for g in FUEL_GOODS
        push!(df.Product, labels[g])
        for s in scenario_list
            push!(df[!, s], solutions[s].q[g])
        end
    end

    feedstock_goods = [:feedstock_corn_n, :feedstock_corn_cs, :feedstock_soy_n, :feedstock_soy_cs]
    feedstock_keys = [:corn_n, :corn_cs, :soy_n, :soy_cs]
    for (g, key) in zip(feedstock_goods, feedstock_keys)
        push!(df.Product, labels[g])
        for s in scenario_list
            push!(df[!, s], solutions[s].q_feedstock[key])
        end
    end

    push!(df.Product, "DDGS")
    for s in scenario_list
        push!(df[!, s], solutions[s].ddgs)
    end

    push!(df.Product, "Soymeal Production (MMT)")
    for s in scenario_list
        push!(df[!, s], solutions[s].soymeal_produ)
    end

    return df
end

function make_demand_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    df = DataFrame(Scenario=String[], Aviation=Float64[], Gasoline=Float64[],
        Diesel=Float64[], Corn=Float64[], Corn_excl_DDGS=Float64[],
        Soyoil=Float64[], Soymeal=Float64[])
    for s in scenario_list
        sol = solutions[s]
        push!(df, (String(s), sol.x[:avi], sol.x[:gas], sol.x[:die],
            sol.x[:corn], sol.x[:corn] - sol.ddgs, sol.x[:soyoil], sol.x[:soymeal]))
    end
    return df
end

function make_price_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    df = DataFrame(Scenario=String[], Aviation=Float64[], Gasoline=Float64[],
        Diesel=Float64[], ConvCorn=Float64[], CSCorn=Float64[],
        ConvSoy=Float64[], CSSoy=Float64[], Soymeal=Float64[])
    for s in scenario_list
        sol = solutions[s]
        push!(df, (String(s), sol.p_c[:avi], sol.p_c[:gas], sol.p_c[:die],
            sol.p_f[:feedstock_corn_n], sol.p_f[:feedstock_corn_cs],
            sol.p_f[:feedstock_soy_n], sol.p_f[:feedstock_soy_cs],
            sol.p_c[:soymeal]))
    end
    return df
end

function make_land_table(solutions, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    omega = params.coeff.omega
    df = DataFrame(Scenario=String[], Corn_N=Float64[], Corn_CS=Float64[],
        Corn_Total=Float64[], Soy_N=Float64[], Soy_CS=Float64[],
        Soy_Total=Float64[], Total_N=Float64[], Total_CS=Float64[],
        Grand_Total=Float64[], Land_Rent=Float64[])
    for s in scenario_list
        sol = solutions[s]
        cn = omega * sol.l_n
        sn = (1 - omega) * sol.l_n
        cc = omega * sol.l_cs
        sc = (1 - omega) * sol.l_cs
        push!(df, (String(s), 1000cn, 1000cc, 1000(cn + cc),
            1000sn, 1000sc, 1000(sn + sc),
            1000sol.l_n, 1000sol.l_cs, 1000(sol.l_n + sol.l_cs),
            sol.duals.r_land))
    end
    return df
end

function make_emissions_table(solutions, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel,
        :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    delta = params.coeff.delta

    df = DataFrame(Scenario=String[], Aviation=Float64[], Road=Float64[],
        Food=Float64[], Total=Float64[])
    for s in scenario_list
        sol = solutions[s]
        avi = sum(delta[g] * sol.q[g] for g in AVIATION_FUELS)
        road = sum(delta[g] * sol.q[g] for g in ROAD_FUELS)
        food = delta[:corn] * (sol.x[:corn] - sol.ddgs) + delta[:soyoil] * sol.x[:soyoil]
        push!(df, (String(s), 1000avi, 1000road, 1000food, 1000(avi + road + food)))
    end
    return df
end

function make_duals_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    df = DataFrame(Scenario=String[], λ_rfs=Float64[], λ_rfs_avi=Float64[],
        λ_lcfs=Float64[], r_land=Float64[], λ_blendwall_ethanol=Float64[],
        λ_blendwall_biodiesel=Float64[], #λ_nonsoy_capacity=Float64[]
    )
    for s in scenario_list
        d = solutions[s].duals
        push!(df, (String(s), d.λ_rfs, d.λ_rfs_avi, d.λ_lcfs, d.r_land,
            d.λ_blendwall_ethanol, d.λ_blendwall_biodiesel, #d.λ_nonsoy_capacity
        ))
    end
    return df
end

# =================================================================================
# 4. Welfare Functions
# =================================================================================

clean_small(val, threshold=1e-10) = abs(val) < threshold ? 0.0 : val

function calc_cs_change(A, k, x_policy, x_sq, p_policy, p_sq)
    return (A / (k + 1)) * (x_policy^(k + 1) - x_sq^(k + 1)) - p_policy * x_policy + p_sq * x_sq
end

function calculate_cs_changes(solutions, solution_sq, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)
    demand = params.demand
    x_sq = solution_sq.x
    p_sq_c = solution_sq.p_c
    p_sq_f = solution_sq.p_f

    cs_changes_all = Dict()
    for s in scenario_list
        sol = solutions[s]
        cs = Dict(
            :avi => calc_cs_change(demand[:avi].A, demand[:avi].k,
                sol.x[:avi], x_sq[:avi], sol.p_c[:avi], p_sq_c[:avi]),
            :gas => calc_cs_change(demand[:gas].A, demand[:gas].k,
                sol.x[:gas], x_sq[:gas], sol.p_c[:gas], p_sq_c[:gas]),
            :die => calc_cs_change(demand[:die].A, demand[:die].k,
                sol.x[:die], x_sq[:die], sol.p_c[:die], p_sq_c[:die]),
            :corn => calc_cs_change(demand[:corn].A, demand[:corn].k,
                sol.x[:corn], x_sq[:corn],
                sol.p_f[:feedstock_corn_n], p_sq_f[:feedstock_corn_n]),
            :soyoil => calc_cs_change(demand[:soyoil].A, demand[:soyoil].k,
                sol.x[:soyoil], x_sq[:soyoil],
                sol.p_f[:feedstock_soy_n], p_sq_f[:feedstock_soy_n]),
            :soymeal => calc_cs_change(demand[:soymeal].A, demand[:soymeal].k,
                sol.x[:soymeal], x_sq[:soymeal],
                sol.p_c[:soymeal], p_sq_c[:soymeal]) / 1000
        )
        cs[:total] = sum(v for (k, v) in cs if k != :total)
        cs_changes_all[s] = cs
    end
    return cs_changes_all
end

function make_cs_change_table(cs_changes_all; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(cs_changes_all)) : scenarios
    df = DataFrame(Good=String[])
    for s in scenario_list
        df[!, String(s)] = Float64[]
    end
    for (gk, gl) in [(:avi, "Aviation"), (:gas, "Road Gasoline"), (:die, "Road Diesel"),
        (:corn, "Food: Corn"), (:soyoil, "Food: Soyoil"),
        (:soymeal, "Food: Soymeal"), (:total, "TOTAL")]
        push!(df.Good, gl)
        for s in scenario_list
            push!(df[!, String(s)], clean_small(cs_changes_all[s][gk]))
        end
    end
    return df
end

function display_cs_changes(cs_changes_all; scenarios=nothing,
    title="CONSUMER SURPLUS CHANGES (billion \$)")
    println("\n" * "="^130)
    println(title)
    println("="^130)
    show(make_cs_change_table(cs_changes_all; scenarios=scenarios), allrows=true)
    println("\n" * "="^130)
end

function calculate_ps_land_changes(solutions, solution_sq, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)
    L0 = params.supply.land.L0
    r0 = params.supply.land.r0_land
    ε = params.supply.land.ϵ_land
    r_sq = solution_sq.duals.r_land
    L_sq = solution_sq.l_n + solution_sq.l_cs

    ps_land_changes = Dict()
    for s in scenario_list
        sol = solutions[s]
        r_pol = sol.duals.r_land
        L_pol = sol.l_n + sol.l_cs
        integral = r0 * (L0^(-1 / ε)) * (ε / (ε + 1)) *
                   (L_pol^((ε + 1) / ε) - L_sq^((ε + 1) / ε))
        ps_land_changes[s] = (
            ps_change=clean_small(r_pol * L_pol - r_sq * L_sq - integral),
            r_sq=clean_small(r_sq),
            r_policy=clean_small(r_pol),
            L_sq=clean_small(L_sq),
            L_policy=clean_small(L_pol),
            integral_term=clean_small(integral)
        )
    end
    return ps_land_changes
end


function display_ps_land_changes(ps_land_changes; scenarios=nothing,
    title="LAND PRODUCER SURPLUS CHANGES (billion \$)")
    scenario_list = isnothing(scenarios) ? collect(keys(ps_land_changes)) : scenarios
    println("\n" * "="^130)
    println(title)
    println("="^130)
    df = DataFrame(Metric=String[])
    for s in scenario_list
        df[!, s] = Float64[]
    end
    push!(df.Metric, "PS Change (Land)")
    for s in scenario_list
        push!(df[!, s], ps_land_changes[s].ps_change)
    end
    show(df, allrows=true)
    println("\n" * "="^130)
end

"""
    calculate_ps_nonsoy_changes(solutions, solution_sq, params; scenarios=nothing)

Non-soy feedstock producer surplus changes, ENDOGENISED-SUPPLY version.

Unlike the main model (perfectly elastic at c up to a hard capacity K, so the rent is
either exactly zero or the rectangle λ·K), here supply slopes upward:

    ns(p)  = ns0 * (p / p0)^ϵ           (supply)
    p(ns)  = p0 * (ns / ns0)^(1/ϵ)      (inverse supply = marginal cost)

Every inframarginal unit costs less than the marginal one, so the rent is the area
between the equilibrium price and the supply curve. Exactly parallel to land rent:

    ΔPS = p_pol*ns_pol - p_sq*ns_sq - ∫[ns_sq → ns_pol] p(n) dn
    ∫   = p0 * ns0^(-1/ϵ) * ϵ/(1+ϵ) * [ns^((1+ϵ)/ϵ)]  evaluated ns_sq → ns_pol

CAUTION ON LEVELS: the inverse supply scales linearly in p0, so ΔPS scales linearly
in p0 too. p0_ns is calibrated at 0.28 rather than the observed weighted-average
non-soy price of 0.49 because 0.28 is what the status-quo RD/BD zero-profit
conditions imply once road-sector subsidies (BTC, LCFS) are omitted from the model.
ΔPS reported here is therefore roughly 0.28/0.49 ≈ 57% of what it would be if the
supply curve were anchored at the observed market price. Treat the level as a lower
bound and check that policy rankings do not hinge on it.
"""
function calculate_ps_nonsoy_changes(solutions, solution_sq, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    ns0 = params.supply.nonsoy.ns0
    p0 = params.supply.nonsoy.p0_ns
    ε = params.supply.nonsoy.ϵ_ns

    p_sq = solution_sq.p_f[:feedstock_nonsoy]
    ns_sq = solution_sq.q_feedstock.nonsoy

    ps_nonsoy_changes = Dict()
    for s in scenario_list
        sol = solutions[s]
        p_pol = sol.p_f[:feedstock_nonsoy]
        ns_pol = sol.q_feedstock.nonsoy

        integral = p0 * (ns0^(-1 / ε)) * (ε / (ε + 1)) *
                   (ns_pol^((ε + 1) / ε) - ns_sq^((ε + 1) / ε))

        ps_nonsoy_changes[s] = (
            ps_change=clean_small(p_pol * ns_pol - p_sq * ns_sq - integral),
            p_sq=clean_small(p_sq),
            p_policy=clean_small(p_pol),
            ns_sq=clean_small(ns_sq),
            ns_policy=clean_small(ns_pol),
            integral_term=clean_small(integral)
        )
    end
    return ps_nonsoy_changes
end

function display_ps_nonsoy_changes(ps_nonsoy_changes; scenarios=nothing,
    title="NON-SOY FEEDSTOCK PRODUCER SURPLUS CHANGES (billion \$)")
    scenario_list = isnothing(scenarios) ? collect(keys(ps_nonsoy_changes)) : scenarios
    println("\n" * "="^130)
    println(title)
    println("="^130)
    if isempty(scenario_list)
        println("(no policy scenarios)")
        println("="^130)
        return nothing
    end
    df = DataFrame(Metric=String[])
    for s in scenario_list
        df[!, s] = Float64[]
    end
    for (mname, mkey) in [("Non-soy price (\$/lb)", :p_policy),
        ("Feedstock used (billion lb)", :ns_policy),
        ("PS Change (Non-soy)", :ps_change)]
        push!(df.Metric, mname)
        for s in scenario_list
            push!(df[!, s], Float64(getfield(ps_nonsoy_changes[s], mkey)))
        end
    end
    show(df, allrows=true)
    p_sq = ps_nonsoy_changes[first(scenario_list)].p_sq
    ns_sq = ps_nonsoy_changes[first(scenario_list)].ns_sq
    println("\n(status quo: p = $(round(p_sq, digits=4)) \$/lb, " *
            "ns = $(round(ns_sq, digits=3)) billion lb)")
    println("="^130)
end

function calculate_gov_revenue_change(solution_policy, implicit_taxes_policy, scenario)
    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    # Sum over ALL SAF goods; calculate_implicit_taxes returns 0 for ineligible ones,
    # so eligibility is decided in exactly one place.
    ELIGIBLE_SAF = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    scenario_str = String(scenario)
    is_carbontax = startswith(scenario_str, "carbontax")
    is_taxcredit = startswith(scenario_str, "taxcredit")
    (is_carbontax || is_taxcredit) || return 0.0

    gov_revenue = 0.0
    if is_carbontax
        for f in AVIATION_FUELS
            gov_revenue += implicit_taxes_policy[f][:carbon_tax] * solution_policy.q[f]
        end
    else
        for saf in ELIGIBLE_SAF
            if haskey(implicit_taxes_policy, saf) && haskey(implicit_taxes_policy[saf], :tax_credit)
                gov_revenue += implicit_taxes_policy[saf][:tax_credit] * solution_policy.q[saf]
            end
        end
    end
    return gov_revenue
end

function calculate_gr_changes(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)
    gr_changes = Dict()
    for s in scenario_list
        sol = solutions[s]
        gr = calculate_gov_revenue_change(sol, sol.implicit_taxes, s)
        str = String(s)
        gr_changes[s] = (
            total=clean_small(gr),
            carbon_tax=startswith(str, "carbontax") ? clean_small(gr) : 0.0,
            tax_credit=startswith(str, "taxcredit") ? clean_small(gr) : 0.0
        )
    end
    return gr_changes
end

function display_gr_changes(gr_changes; scenarios=nothing,
    title="GOVERNMENT REVENUE CHANGES (billion \$)")
    scenario_list = isnothing(scenarios) ? collect(keys(gr_changes)) : scenarios
    println("\n" * "="^80)
    println(title)
    println("="^80)
    df = DataFrame()
    for s in scenario_list
        df[!, s] = [gr_changes[s].total]
    end
    insertcols!(df, 1, :Metric => ["Total GR"])
    show(df, allrows=true)
    println("\n" * "="^80)
end

function calculate_environmental_benefit(solutions, solution_sq, scc; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)
    em_sq = solution_sq.emissions
    env_benefits = Dict()
    for s in scenario_list
        em = solutions[s].emissions
        env_benefits[s] = (
            avi_benefit=clean_small((em_sq.aviation - em.aviation) * scc),
            road_benefit=clean_small((em_sq.road - em.road) * scc),
            food_benefit=clean_small((em_sq.food - em.food) * scc),
            total_benefit=clean_small((em_sq.total - em.total) * scc)
        )
    end
    return env_benefits
end

function display_environmental_benefits(env_benefits, scc; scenarios=nothing,
    title="ENVIRONMENTAL BENEFITS")
    scenario_list = isnothing(scenarios) ? collect(keys(env_benefits)) : scenarios
    println("\n" * "="^130)
    println(title)
    println("Social Cost of Carbon (SCC) = \$$(scc) per ton CO2e")
    println("="^130)
    df = DataFrame(Sector=String[])
    for s in scenario_list
        df[!, s] = Float64[]
    end
    for (sname, bkey) in [("Aviation", :avi_benefit), ("Road", :road_benefit),
        ("Food", :food_benefit), ("Total", :total_benefit)]
        push!(df.Sector, sname)
        for s in scenario_list
            push!(df[!, s], getfield(env_benefits[s], bkey))
        end
    end
    println("\n--- Environmental Benefits (billion \$) ---")
    show(df, allrows=true)
    println("\n" * "="^130)
end

"""
    calculate_total_welfare(cs_changes, ps_land_changes, gr_changes, env_benefits;
                            ps_nonsoy_changes=nothing, scenarios=nothing)

Producer surplus is reported in three pieces:
  ps_land_change   land rent
  ps_nonsoy_change non-soy feedstock rent (area above the CES supply curve)
  ps_total_change  the sum, which is what enters private surplus

`ps_nonsoy_changes` is a keyword so older 4-argument calls still run, but those
silently drop the non-soy rent. In THIS model non-soy supply is upward sloping, so
that rent is generally non-zero and dropping it makes the welfare measure internally
inconsistent: the model would charge a rising resource cost for non-soy feedstock
while recording no corresponding surplus for its suppliers. Always pass it.
"""
function calculate_total_welfare(cs_changes, ps_land_changes, gr_changes, env_benefits;
    ps_nonsoy_changes=nothing, scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(cs_changes)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)
    welfare_summary = Dict()
    for s in scenario_list
        cs = cs_changes[s][:total]
        ps_land = ps_land_changes[s].ps_change
        ps_nonsoy = isnothing(ps_nonsoy_changes) ? 0.0 : ps_nonsoy_changes[s].ps_change
        ps_total = ps_land + ps_nonsoy
        gr = gr_changes[s].total
        env = env_benefits[s].total_benefit
        welfare_summary[s] = (
            cs_change=clean_small(cs),
            cs_by_sector=cs_changes[s],
            ps_land_change=clean_small(ps_land),
            ps_nonsoy_change=clean_small(ps_nonsoy),
            ps_total_change=clean_small(ps_total),
            gr_change=clean_small(gr),
            env_benefit=clean_small(env),
            private_surplus=clean_small(cs + ps_total + gr),
            social_welfare=clean_small(cs + ps_total + gr + env)
        )
    end
    return welfare_summary
end

function display_welfare_summary(welfare_summary; scenarios=nothing, title="WELFARE SUMMARY")
    scenario_list = isnothing(scenarios) ? collect(keys(welfare_summary)) : scenarios
    println("\n" * "="^130)
    println(title)
    println("="^130)
    df = DataFrame(Metric=String[])
    for s in scenario_list
        df[!, s] = Float64[]
    end
    for (mname, mkey) in [("CS Change (a)", :cs_change),
        ("PS Land (b1)", :ps_land_change),
        ("PS Non-soy (b2)", :ps_nonsoy_change),
        ("PS Total (b=b1+b2)", :ps_total_change),
        ("Gov Revenue (c)", :gr_change),
        ("Env Benefit (d)", :env_benefit),
        ("Private Surplus (∆=a+b+c)", :private_surplus),
        ("Social Welfare (∆+d)", :social_welfare)]
        push!(df.Metric, mname)
        for s in scenario_list
            push!(df[!, s], getfield(welfare_summary[s], mkey))
        end
    end
    println("\n--- All values in billion \$ ---")
    show(df, allrows=true)
    println("\n" * "="^130)
end

function calculate_average_abatement_cost(welfare_summary, solutions, solution_sq;
    scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(welfare_summary)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)
    em_sq = solution_sq.emissions.total
    aac_results = Dict()
    for s in scenario_list
        em_red = em_sq - solutions[s].emissions.total
        w = welfare_summary[s]
        aac_results[s] = (
            emission_reduction=clean_small(em_red),
            private_surplus=clean_small(w.private_surplus),
            social_welfare=clean_small(w.social_welfare),
            aac_private=abs(em_red) > 1e-10 ? clean_small(-w.private_surplus / em_red) : 0.0,
            aac_social=abs(em_red) > 1e-10 ? clean_small(-w.social_welfare / em_red) : 0.0
        )
    end
    return aac_results
end

function display_aac_analysis(aac_results; scenarios=nothing,
    title="AVERAGE ABATEMENT COST ANALYSIS")
    scenario_list = isnothing(scenarios) ? collect(keys(aac_results)) : scenarios
    println("\n" * "="^130)
    println(title)
    println("="^130)
    df = DataFrame(Metric=String[])
    for s in scenario_list
        df[!, s] = Float64[]
    end
    for (mname, mkey) in [("Emission Reduction (B ton CO2e)", :emission_reduction),
        ("∆Private welfare (billion \$)", :private_surplus),
        ("∆Social welfare (billion \$)", :social_welfare),
        ("AAC (Private) (\$/ton CO2e)", :aac_private),
        ("AAC (Social) (\$/ton CO2e)", :aac_social)]
        push!(df.Metric, mname)
        for s in scenario_list
            push!(df[!, s], getfield(aac_results[s], mkey))
        end
    end
    show(df, allrows=true)
    println("\n" * "="^130)
end

# =================================================================================
# 5. Master Display Function
# =================================================================================

function display_comparison_tables(solutions, params, policy_configs;
    scenarios=nothing,
    title="RESULTS",
    show_policy_params=true,
    equivalent_policies=nothing)

    println("\n" * "="^130)
    println(title)
    println("="^130)

    # ── Policy Parameters ─────────────────────────────────────────────────────
    if show_policy_params
        policy_labels = Dict(
            :statusquo => ("Status Quo", "No Policy"),
            :carbontax => ("Carbon Tax", "Carbon Tax (\$/ton CO2e)"),
            :rfs => ("RFS Aviation", "Mandate Share"),
            :lcfs => ("LCFS", "CI Reduction (σ)"),
            :taxcredit => ("Tax Credit", "Rate (\$/gal)")
        )
        println("\n--- Policy Parameters ---")

        if !isnothing(equivalent_policies)
            param_df = DataFrame(Policy=String[], Parameter_Name=String[],
                Parameter_Value=Float64[], Target_Metric=String[],
                Actual_Value=Float64[])
            for pt in [:statusquo, :carbontax, :rfs, :lcfs, :taxcredit]
                haskey(equivalent_policies, pt) || continue
                result = equivalent_policies[pt]
                config = result.config
                label, pname = policy_labels[pt]
                pval = if pt == :carbontax
                    config.t
                elseif pt == :rfs
                    config.θ_avi
                elseif pt == :lcfs
                    config.σ
                elseif pt == :taxcredit
                    config.p
                else
                    0.0
                end
                tm, av = if haskey(result, :actual_saf)
                    "SAF (billion gal)", result.actual_saf
                elseif haskey(result, :actual_emission)
                    "Emissions (Bton CO2e)", result.actual_emission
                else
                    "N/A", NaN
                end
                push!(param_df, (label, pname, pval, tm, av))
            end
        else
            param_df = DataFrame(Policy=String[], Parameter_Name=String[],
                Parameter_Value=Float64[])
            for pt in [:statusquo, :carbontax, :rfs, :lcfs, :taxcredit]
                haskey(policy_configs, pt) || continue
                config = policy_configs[pt]
                label, pname = policy_labels[pt]
                pval = if pt == :carbontax
                    config.t
                elseif pt == :rfs
                    config.θ_avi
                elseif pt == :lcfs
                    config.σ
                elseif pt == :taxcredit
                    config.p
                else
                    0.0
                end
                push!(param_df, (label, pname, pval))
            end
        end
        show(param_df, allrows=true)
    end

    # ── Analysis Tables ───────────────────────────────────────────────────────
    implicit_taxes = calculate_all_implicit_taxes(solutions, params, policy_configs)

    for (table_name, table) in [
        ("Implicit Taxes/Subsidies (\$/gallon)",
            make_implicit_tax_table(implicit_taxes, params; scenarios=scenarios, exclude_statusquo=true)),
        ("Production", make_production_table(solutions, params; scenarios=scenarios)),
        ("Demand", make_demand_table(solutions; scenarios=scenarios)),
        ("Prices", make_price_table(solutions; scenarios=scenarios)),
        ("Land Use (Million Acres)", make_land_table(solutions, params; scenarios=scenarios)),
        ("Emissions (MMT CO2e)", make_emissions_table(solutions, params; scenarios=scenarios)),
        ("Dual Variables", make_duals_table(solutions; scenarios=scenarios))
    ]
        println("\n--- $table_name ---")
        show(table, allrows=true)
    end

    println("\n" * "="^130)

    # ── Welfare ───────────────────────────────────────────────────────────────
    # The status quo is always re-solved with SQ_CONFIG
    println("\nComputing status quo for welfare comparison...")
    sq_model = build_unified_model(params, SQ_CONFIG)
    optimize!(sq_model)
    sq_sol = extract_solution(sq_model, :statusquo)
    enriched_sq = merge(sq_sol, (
        implicit_taxes=calculate_implicit_taxes(sq_sol, params, SQ_CONFIG),
        emissions=calculate_emissions_detail(sq_sol, params)
    ))

    # solutions enrich
    implicit_taxes_all = calculate_all_implicit_taxes(solutions, params, policy_configs)
    enriched = Dict(
        s => merge(sol, (
            implicit_taxes=implicit_taxes_all[s],
            emissions=calculate_emissions_detail(sol, params)
        ))
        for (s, sol) in solutions if !isnothing(sol)
    )

    pol_scenarios = filter(s -> s != :statusquo,
        isnothing(scenarios) ? collect(keys(solutions)) : scenarios)

    cs_changes = calculate_cs_changes(enriched, enriched_sq, params; scenarios=pol_scenarios)
    ps_land = calculate_ps_land_changes(enriched, enriched_sq, params; scenarios=pol_scenarios)
    ps_nonsoy = calculate_ps_nonsoy_changes(enriched, enriched_sq, params; scenarios=pol_scenarios)
    gr_changes = calculate_gr_changes(enriched; scenarios=pol_scenarios)
    env_benefits = calculate_environmental_benefit(enriched, enriched_sq, SCC; scenarios=pol_scenarios)
    welfare_summary = calculate_total_welfare(cs_changes, ps_land, gr_changes, env_benefits;
        ps_nonsoy_changes=ps_nonsoy, scenarios=pol_scenarios)
    aac_results = calculate_average_abatement_cost(welfare_summary, enriched, enriched_sq; scenarios=pol_scenarios)

    display_cs_changes(cs_changes;
        scenarios=pol_scenarios, title="$title: CS CHANGES (billion \$)")
    display_ps_land_changes(ps_land;
        scenarios=pol_scenarios, title="$title: LAND PS CHANGES (billion \$)")
    display_ps_nonsoy_changes(ps_nonsoy;
        scenarios=pol_scenarios, title="$title: NON-SOY FEEDSTOCK PS CHANGES (billion \$)")
    display_gr_changes(gr_changes;
        scenarios=pol_scenarios, title="$title: GOVERNMENT REVENUE (billion \$)")
    display_environmental_benefits(env_benefits, SCC;
        scenarios=pol_scenarios, title="$title: ENV BENEFITS (SCC = \$$SCC)")
    display_welfare_summary(welfare_summary;
        scenarios=pol_scenarios, title="$title: WELFARE SUMMARY (billion \$)")
    display_aac_analysis(aac_results;
        scenarios=pol_scenarios, title="$title: AVERAGE ABATEMENT COST")

    return (
        cs_changes=cs_changes,
        ps_land=ps_land,
        ps_nonsoy=ps_nonsoy,
        gr_changes=gr_changes,
        env_benefits=env_benefits,
        welfare_summary=welfare_summary,
        aac_results=aac_results
    )
end

# =================================================================================
# Exports
# =================================================================================
export display_comparison_tables,
    calculate_implicit_taxes, calculate_all_implicit_taxes, make_implicit_tax_table,
    make_production_table, make_demand_table, make_price_table,
    make_land_table, make_emissions_table, make_duals_table,
    calculate_emissions_detail,
    clean_small,
    calc_cs_change, calculate_cs_changes, make_cs_change_table, display_cs_changes,
    calculate_ps_land_changes, display_ps_land_changes,
    calculate_ps_nonsoy_changes, display_ps_nonsoy_changes,
    calculate_gov_revenue_change, calculate_gr_changes, display_gr_changes,
    calculate_environmental_benefit, display_environmental_benefits,
    calculate_total_welfare, display_welfare_summary,
    calculate_average_abatement_cost, display_aac_analysis,
    SCC, POL, SQ_CONFIG

end # module AnalysisEndoNonsoy
