# model_sp.jl
# Social-planner (primal) counterpart to the market-equilibrium MCP in main/model_mkt.jl.

module ModelSP

using JuMP, Ipopt, Printf

import Main.ModelMkt: params, saf_credit, policy_ci, FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS,
    PRETREAT_ROUTES, PRETREAT_FLOOR, SUPPLY_FLOOR, fossil_ces, supply_ps

# =================================================================================
# 1. Sets
# =================================================================================

const SECTORS = [:avi, :gas, :die, :corn, :soyoil, :soymeal]

const SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
const BIODIESEL_GOODS = [:biodiesel_soy, :biodiesel_nonsoy]
const RD_GOODS = [:rd_soy, :rd_nonsoy]
const AVIATION_FUELS = [:jet_fuel, SAF_GOODS...]
const NONSOY_GOODS = [:saf_hefa_nonsoy, :rd_nonsoy, :biodiesel_nonsoy]

# capacity groups
const CAP_GROUPS = [:jet_fuel, :saf_atj_shared, :saf_hefa_shared,
    :gasoline, :ethanol, :diesel, :biodiesel_shared]

const GROUP_MEMBERS = Dict(
    :jet_fuel => [:jet_fuel],
    :saf_atj_shared => [:saf_atj_conv, :saf_atj_cs],
    :saf_hefa_shared => [:saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy, :rd_soy, :rd_nonsoy],
    :gasoline => [:gasoline],
    :ethanol => [:ethanol],
    :diesel => [:diesel],
    :biodiesel_shared => [:biodiesel_soy, :biodiesel_nonsoy],
)

# DDGS yield per gallon 
const ξ_ethanol = 0.092
const ξ_atj = 0.159

# Initial values for IPOPT
const START_X = Dict(:avi => 1204.79, :gas => 2856.73, :die => 357.28,
    :corn => 8.5, :soyoil => 14.2, :soymeal => 62.3)
const START_Q = Dict(:jet_fuel => 20.0, :gasoline => 126.0, :diesel => 44.5, :ethanol => 14.0,
    :saf_atj_conv => 0.01, :saf_atj_cs => 0.01,
    :saf_hefa_conv => 1.0, :saf_hefa_cs => 0.01, :saf_hefa_nonsoy => 0.5,
    :biodiesel_soy => 1.5, :biodiesel_nonsoy => 0.8,
    :rd_soy => 1.0, :rd_nonsoy => 0.5)

# =================================================================================
# 2. Policy eligibility
# =================================================================================

function is_eligible(g, config, delta)
    (!config.use_ci_threshold || delta[g] <= 0.5 * delta[:jet_fuel]) &&
        (config.recognize_cs || g ∉ (:saf_atj_cs, :saf_hefa_cs))
end

# =================================================================================
# 3. Model building
# =================================================================================

function build_planner_model(params, config; warm_start=nothing, silent=true)

    coeff = params.coeff
    demand = params.demand
    fuel_cost = params.supply.fuel
    land = params.supply.land

    r = coeff.r
    beta = coeff.beta
    alpha = coeff.alpha
    delta = coeff.delta
    gamma = coeff.gamma
    theta = coeff.theta
    omega = coeff.omega
    kappa = coeff.kappa
    s2oil = coeff.soybean_to_oil
    s2meal = coeff.soybean_to_meal
    δ_mj = coeff.delta_mj
    baselineCI = coeff.baselineCI
    p_ns = coeff.nonsoy_feedstock_price
    K_ns = coeff.nonsoy_capacity
    ρ_hefa = coeff.hefa_saf_premium
    rho_pre = get(coeff, :rho_pretreat, Dict{Symbol,Float64}())
    L0, r0_land, ϵ_land = land.L0, land.r0_land, land.ϵ_land

    model = Model(Ipopt.Optimizer)
    silent && set_silent(model)
    set_optimizer_attribute(model, "bound_relax_factor", 0.0)
    set_optimizer_attribute(model, "honor_original_bounds", "yes")
    set_optimizer_attribute(model, "tol", 1e-10)
    set_optimizer_attribute(model, "acceptable_tol", 1e-8)
    set_optimizer_attribute(model, "max_iter", 5000)

    # ── Variables ────────────────────────────────────────────────────────────────
    # x >= 0.1 mirrors the MCP's lower bound so the two feasible sets match.
    @variable(model, x[s in SECTORS] >= 0.1)
    @variable(model, q[g in FUEL_GOODS] >= 0)
    @variable(model, l_n >= 0)
    @variable(model, l_cs >= 0)

    for s in SECTORS
        set_start_value(x[s], START_X[s])
    end
    for g in FUEL_GOODS
        set_start_value(q[g], START_Q[g])
    end
    set_start_value(l_n, 0.11)
    set_start_value(l_cs, 1e-3)

    if !isnothing(warm_start)
        apply_warm_start!(model, warm_start)
    end

    # ── Aggregates ───────────────────────────────────────────────────────────────
    @expression(model, Q_group[m in CAP_GROUPS], sum(q[g] for g in GROUP_MEMBERS[m]))
    @expression(model, ddgs, ξ_ethanol * q[:ethanol] + ξ_atj * (q[:saf_atj_conv] + q[:saf_atj_cs]))
    @expression(model, N_ns, sum(alpha[g] * q[g] for g in NONSOY_GOODS))

    # feedstock produced per practice
    @expression(model, q_corn_n, omega * gamma[:feedstock_corn_n] * l_n)
    @expression(model, q_corn_cs, omega * gamma[:feedstock_corn_cs] * l_cs)
    @expression(model, q_soy_n, (1 - omega) * gamma[:feedstock_soy_n] * l_n * s2oil)
    @expression(model, q_soy_cs, (1 - omega) * gamma[:feedstock_soy_cs] * l_cs * s2oil)

    # ── Objective ───────────────────────────────────

    # (A) consumer benefit: B̃_s(x) = A/(k+1) x^(k+1), the antiderivative of the inverse demand
    #     curve with the (arbitrary, divergent) constant left out.  soymeal is priced in $/mt on
    #     a quantity in million mt, so its product is in million $ -- /1000 to reach billions,
    #     exactly as analysis.jl:327 does for the consumer-surplus change.
    @expression(model, benefit[s in SECTORS],
        (s == :soymeal ? 1 / 1000 : 1.0) *
        demand[s].A / (demand[s].k + 1) * x[s]^(demand[s].k + 1))
    @expression(model, total_benefit, sum(benefit[s] for s in SECTORS))

    # (B) land cost: area under the inverse land supply curve.  Its derivative is
    #     r0(T/L0)^(1/eps), so the FOC delivers T = L(r_land) on its own. That equality is
    #     already implied here and must NOT also be imposed as a constraint.
    @expression(model, T_land, l_n + l_cs)
    @expression(model, land_cost,
        r0_land * L0 / (1 + 1 / ϵ_land) * (T_land / L0)^(1 + 1 / ϵ_land))

    # (C) climate-smart adoption
    @expression(model, adoption_cost, kappa * l_cs)

    # (D) production cost: 
    @expression(model, group_cost[m in CAP_GROUPS],
        (fossil_ces(fuel_cost[m]) ?
         fuel_cost[m].c0 * fuel_cost[m].Q0 / (1 + 1 / fuel_cost[m].eta) *
         ((Q_group[m] + SUPPLY_FLOOR) / fuel_cost[m].Q0)^(1 + 1 / fuel_cost[m].eta) :
         fuel_cost[m].c0 * Q_group[m] + fuel_cost[m].c1 / 2 * Q_group[m]^2) +
        fuel_cost[m].c2 / 3 * max(0, Q_group[m] - fuel_cost[m].v)^3)

    @expression(model, pretreat_cost,
        sum(rho_pre[k] / 2 *
            (sum(alpha[g] * q[g] for g in route.ns) -
             route.sbar * sum(alpha[g] * q[g] for g in [route.ns; route.soy]))^2 /
            (sum(alpha[g] * q[g] for g in [route.ns; route.soy]) + PRETREAT_FLOOR)
            for (k, route) in PRETREAT_ROUTES if get(rho_pre, k, 0.0) != 0.0;
            init=zero(AffExpr)))

    @expression(model, production_cost,
        sum(group_cost[m] for m in CAP_GROUPS) -
        ρ_hefa * (q[:rd_soy] + q[:rd_nonsoy]) +      # RD forgoes HEFA's extra processing
        p_ns * N_ns +                                 # non-soy feedstock bill
        pretreat_cost)

    # (E) policy wedge.  Both terms are written unconditionally; a scenario that does not use a
    #     lever passes 0 for it, exactly as the MCP does.
    @expression(model, policy_term,
        # carbon tax: aviation only, priced at the CI the regulator uses as the base
        -config.t * sum(policy_ci(g, config, delta) * q[g] for g in AVIATION_FUELS) +
        # tax credit: p*(delta_jet - delta_g) per gallon on the policy CI, zero when the CI
        # threshold screens the fuel out.  Same function the MCP calls, so the two cannot drift.
        sum(saf_credit(g, config, coeff) * q[g] for g in SAF_GOODS; init=zero(AffExpr)))

    @objective(model, Max,
        total_benefit - land_cost - adoption_cost - production_cost + policy_term)

    # ── Market clearing.  Multiplier = price. ────────────────────────────────────

    # corn
    @constraint(model, mc_corn,
        q_corn_n + ddgs - alpha[:saf_atj_conv] * q[:saf_atj_conv] - alpha[:ethanol] * q[:ethanol]
        -
        x[:corn] >= 0)

    # Conventional soy oil
    @constraint(model, mc_soyoil,
        q_soy_n - alpha[:saf_hefa_conv] * q[:saf_hefa_conv]
        -
        alpha[:rd_soy] * q[:rd_soy]
        -
        alpha[:biodiesel_soy] * q[:biodiesel_soy]
        -
        x[:soyoil] >= 0)

    # Soybean meal, from the PRODUCTION side.
    @constraint(model, mc_soymeal,
        1000 * s2meal * (1 - omega) *
        (gamma[:feedstock_soy_n] * l_n + gamma[:feedstock_soy_cs] * l_cs) - x[:soymeal] >= 0)

    # aviation, gasoline, diesel
    @constraint(model, mc_avi,
        r[:jet_fuel] * (q[:jet_fuel] + beta[(:saf, :jet_fuel)] * sum(q[g] for g in SAF_GOODS))
        -
        x[:avi] >= 0)
    @constraint(model, mc_gas,
        r[:gasoline] * (q[:gasoline] + beta[(:ethanol, :gasoline)] * q[:ethanol])
        -
        x[:gas] >= 0)
    @constraint(model, mc_die,
        r[:diesel] * (q[:diesel] +
                      beta[(:biodiesel, :diesel)] * sum(q[g] for g in BIODIESEL_GOODS) +
                      beta[(:rd, :diesel)] * sum(q[g] for g in RD_GOODS))
        -
        x[:die] >= 0)

    # ── Inequality constraints ─────────────────────────────────────────
    # The first two are the climate-smart feedstock balances.  They stay inequalities because CS
    # feedstock has no food outlet and can be left unused, in which case its price is zero.
    @constraint(model, c1_corn_cs, q_corn_cs - alpha[:saf_atj_cs] * q[:saf_atj_cs] >= 0)
    @constraint(model, c2_soy_cs, q_soy_cs - alpha[:saf_hefa_cs] * q[:saf_hefa_cs] >= 0)

    @constraint(model, c3_nonsoy, K_ns - N_ns >= 0)
    @constraint(model, c4_rfs,
        q[:ethanol] + 1.5 * sum(q[g] for g in BIODIESEL_GOODS)
        + 1.7 * sum(q[g] for g in RD_GOODS)
        -
        theta * (q[:gasoline] + q[:diesel]) >= 0)
    @constraint(model, c5_bw_ethanol, 0.1 * q[:gasoline] - 0.9 * q[:ethanol] >= 0)
    @constraint(model, c6_bw_biodiesel, 0.05 * q[:diesel]
                                        -
                                        0.95 * sum(q[g] for g in BIODIESEL_GOODS) >= 0)

    # ── The two quantity-based SAF policies (vol. mandate and CI standard)─────────────────────────────────
    elig = [g for g in SAF_GOODS if is_eligible(g, config, delta)]

    @constraint(model, c7_rfs_avi,
        1.6 * sum(q[g] for g in elig; init=zero(AffExpr)) - config.θ_avi * q[:jet_fuel] >= 0)
    lcfs_set = [:jet_fuel, elig...]
    @constraint(model, c8_lcfs,
        (1 - config.σ) * delta[:jet_fuel] * sum(q[g] for g in lcfs_set) -
        sum(delta[g] * q[g] for g in lcfs_set) >= 0)

    return model
end

# apply_warm_start!(model, sol):
function apply_warm_start!(model, sol)
    x, q = model[:x], model[:q]
    for s in SECTORS
        haskey(sol, :x) && set_start_value(x[s], max(sol.x[s], 0.11))
    end
    for g in FUEL_GOODS
        haskey(sol, :q) && set_start_value(q[g], max(sol.q[g], 1e-3))
    end
    haskey(sol, :l_n) && set_start_value(model[:l_n], max(sol.l_n, 1e-4))
    haskey(sol, :l_cs) && set_start_value(model[:l_cs], max(sol.l_cs, 1e-4))
    return model
end

# =================================================================================
# 4. Solving and extraction
# =================================================================================

function run_planner_scenario(scenario::Symbol, params, policy_configs; warm_start=nothing, silent=true)
    config = getproperty(policy_configs, scenario)
    model = build_planner_model(params, config; warm_start=warm_start, silent=silent)
    optimize!(model)

    st = termination_status(model)
    if st == MOI.LOCALLY_SOLVED || st == MOI.OPTIMAL || st == MOI.ALMOST_LOCALLY_SOLVED
        println("\n✓ $(scenario) [planner] solved ($(st))")
        return model
    else
        println("\n✗ $(scenario) [planner] failed: $(st)")
        return nothing
    end
end

# dual_sign(model): adjust the signs of dual variables
function dual_sign(model, params)
    d = dual(model[:mc_avi])
    p_ref = params.demand[:avi].A * value(model[:x][:avi])^params.demand[:avi].k
    if abs(d) < 1e-12
        return 1.0            # degenerate; nothing to normalise against
    end
    return sign(d) * sign(p_ref)
end

# extract_planner_solution(model, scenario, params)
function extract_planner_solution(model, scenario, params)
    isnothing(model) && return nothing

    coeff = params.coeff
    demand = params.demand
    land = params.supply.land

    xv = Dict(s => value(model[:x][s]) for s in SECTORS)
    qv = Dict(g => value(model[:q][g]) for g in FUEL_GOODS)
    l_n, l_cs = value(model[:l_n]), value(model[:l_cs])

    sgn = dual_sign(model, params)
    dl(c) = sgn * dual(model[c])

    # prices from the demand curves
    price(s) = demand[s].A * xv[s]^demand[s].k

    # p_c, p_f
    p_c = Dict(:avi => price(:avi), :gas => price(:gas), :die => price(:die),
        :soymeal => price(:soymeal))
    p_f = Dict(:feedstock_corn_n => price(:corn),
        :feedstock_soy_n => price(:soyoil),
        :feedstock_corn_cs => dl(:c1_corn_cs),    # multiplier on the CS corn balance
        :feedstock_soy_cs => dl(:c2_soy_cs))     # multiplier on the CS soy balance

    T = l_n + l_cs
    r_land = land.r0_land * (T / land.L0)^(1 / land.ϵ_land)

    duals = (
        λ_rfs=dl(:c4_rfs),
        λ_rfs_avi=dl(:c7_rfs_avi),
        λ_lcfs=dl(:c8_lcfs),
        r_land=r_land,
        λ_blendwall_ethanol=dl(:c5_bw_ethanol),
        λ_blendwall_biodiesel=dl(:c6_bw_biodiesel),
        λ_nonsoy_capacity=dl(:c3_nonsoy),
    )

    ddgs = ξ_ethanol * qv[:ethanol] + ξ_atj * (qv[:saf_atj_conv] + qv[:saf_atj_cs])

    # planner-only diagnostics
    sp = (
        W=objective_value(model),
        benefit=value(model[:total_benefit]),
        land_cost=value(model[:land_cost]),
        adoption_cost=value(model[:adoption_cost]),
        production_cost=value(model[:production_cost]),
        policy_term=value(model[:policy_term]),
        # multiplier-implied prices, for cross-checking against the D_s(x_s) values above
        dual_prices=Dict(:avi => dl(:mc_avi), :gas => dl(:mc_gas), :die => dl(:mc_die),
            :corn => dl(:mc_corn), :soyoil => dl(:mc_soyoil),
            :soymeal => 1000 * dl(:mc_soymeal)),
        slack=Dict(m => value(model[:Q_group][m]) - params.supply.fuel[m].v for m in CAP_GROUPS),
        iterations=try
            MOI.get(model, MOI.BarrierIterations())
        catch
            ;
            -1
        end,
        status=termination_status(model),
    )

    return (
        scenario=scenario,
        q=qv,
        x=xv,
        p_f=p_f,
        p_c=p_c,
        l_n=l_n,
        l_cs=l_cs,
        q_feedstock=(
            corn_n=coeff.omega * coeff.gamma[:feedstock_corn_n] * l_n,
            corn_cs=coeff.omega * coeff.gamma[:feedstock_corn_cs] * l_cs,
            soy_n=(1 - coeff.omega) * coeff.gamma[:feedstock_soy_n] * l_n * coeff.soybean_to_oil,
            soy_cs=(1 - coeff.omega) * coeff.gamma[:feedstock_soy_cs] * l_cs * coeff.soybean_to_oil,
        ),
        duals=duals,
        ddgs=ddgs,
        soymeal_produ=xv[:soymeal],
        sp=sp,
    )
end

export build_planner_model, run_planner_scenario, extract_planner_solution,
    is_eligible,
    SECTORS, SAF_GOODS, BIODIESEL_GOODS, RD_GOODS, AVIATION_FUELS, NONSOY_GOODS,
    CAP_GROUPS, GROUP_MEMBERS

end # module ModelSP
