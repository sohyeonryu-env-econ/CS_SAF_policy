module ModelMkt

using JuMP, PATHSolver
using Printf
using Plots
using DataFrames

# =================================================================================
# 1. Input
# =================================================================================

# =====================
# Goods and Scenario Sets
# =====================

# goods (q)
const GOODS = [
    # upstream crops
    :feedstock_corn_n, # "feedstock: conventional corn"
    :feedstock_corn_cs, # "feedstock: climate-smart corn"
    :feedstock_soy_n, #"feedstock: conventional soy"
    :feedstock_soy_cs, # "feedstock: climate-smart soy"
    # downstream fuel and food
    :jet_fuel, # "Jet fuel"
    :saf_atj_conv, # "Conventional ATJ-SAF"
    :saf_atj_cs, # "Climate-smart ATJ-SAF"
    :saf_hefa_conv, # "Conventional HEFA-SAF"
    :saf_hefa_cs, # "Climate-smart HEFA-SAF"
    :saf_hefa_nonsoy, # "Non soy HEFA-SAF"
    :gasoline, # "Gasoline"
    :ethanol, # "Ethanol"
    :diesel, # "Diesel"
    :biodiesel_soy, # "Soy Biodiesel"
    :biodiesel_nonsoy, # "Non soy Biodiesel"
    :rd_soy, # "Soy Renewable Diesel"
    :rd_nonsoy, # "Non soy Renewable diesel"
    :corn, # "food: Corn"
    :soyoil # "food: Soy oil"
]

# scenarios
const SCENARIOS = [
    :statusquo,
    :carbontax,
    :rfs,
    :lcfs,
    :taxcredit,
]

# Sectors (consumers' goods, x)
const SECTORS = [
    :avi,
    :gas,
    :die,
    :corn,
    :soyoil,
    :soymeal
]

const FEEDSTOCK_GOODS = GOODS[1:4] # all feedstock goods
const FUEL_GOODS = GOODS[5:17]  # all fuel goods
const FOOD_GOODS = GOODS[18:19]  # all food goods
const SAF_GOODS = GOODS[6:10]   # the five SAF pathways
const AVIATION_FUELS = GOODS[5:10]  # jet fuel + the five SAF pathways

# =====================
# parameters
# =====================

# δ: Carbon Intensity. Fuel goods are ton CO2e per gallon.
# Food goods differ: corn is ton CO2e per bushel, soyoil is ton CO2e per lb (see emissions calc in analysis.jl)
δ_vec = [
    0.01155398, 0.007320614, 0.004674426, 0.004609978, 0.003717805, 0.002029502,
    0.012039062, 0.003705445, 0.014101869, 0.003879759, 0.0022,
    0.003879759, 0.0022, 0.004623319, 0.002040691
]

δ = Dict(g => v for (g, v) in zip(union(FUEL_GOODS, FOOD_GOODS), δ_vec))

# gamma(γ): Feedstock Land use Coefficients (bu/acre)
γ_vec = [
    188.8, 171.6, 53.6, 48.7
]
γ = Dict(g => v for (g, v) in zip(FEEDSTOCK_GOODS, γ_vec))

# r: energy conversion rate
r = Dict(
    :jet_fuel => 58.95847369,
    :gasoline => 21.10983193,
    :diesel => 7.256737051
)

# β: biofuel to fossil fuel conversion rate
β = Dict(
    (:saf, :jet_fuel) => 0.973424742,
    (:ethanol, :gasoline) => 0.681920857,
    (:biodiesel, :diesel) => 0.937978731,
    (:rd, :diesel) => 0.964155574
)

# α: feedstock to biofuel conversion coefficients (bu/gal, lb/gal)
α = Dict(
    :saf_atj_conv => 0.59,
    :saf_atj_cs => 0.59,
    :ethanol => 0.3448,
    :saf_hefa_conv => 9.0,
    :saf_hefa_cs => 9.0,
    :saf_hefa_nonsoy => 9.0,
    :biodiesel_soy => 7.55,
    :biodiesel_nonsoy => 7.55,
    :rd_soy => 8.5,
    :rd_nonsoy => 8.5
)

# theta: road sector RFS D6 blending share
θ = 0.125

# Soybeans to oil conversion factor (lb oil per bushel soybeans)
soybean_to_oil = 10.71 # lb oil per bushel of soybeans
soybean_to_meal = 0.02155 # metric ton / bushel of soybeans

# meal per oil ratio: (million metric ton of meal per billion lb of oil)
meal_per_oil = soybean_to_meal / soybean_to_oil * 1000

# delta_mj for IRA credit calculation excluding ILUC
δ_mj_vec = [
    89.0,      # 1. jet_fuel
    50.86,   # 2. saf_atj_conv
    30.0,   # 3. saf_atj_cs
    22.91,   # 4. saf_hefa_conv
    15.85,   # 5. saf_hefa_cs
    16.06,     # 6. saf_hefa_nonsoy
    100.72,    # 7. gasoline
    39.71,     # 8. ethanol
    104.87,    # 9. diesel
    18.86,     # 10. biodiesel_soy
    17.25,     # 11. biodiesel_nonsoy
    22.4,     # 12. rd_soy
    15.74      # 13. rd_nonsoy
]

δ_mj = Dict(g => v for (g, v) in zip(FUEL_GOODS, δ_mj_vec))

# baseline CI for IRA credit calculation
const baselineCI = 50.0

# demand functions: p(x) = p0*(x/x0)^(1/sigma) = A*x^k where sigma < 0, s = x0*(p_sub/p0)^(-1/sigma)
# sigma: price elasticity of demand
# k = 1 / sigma
# A = p0 * x0^(-1/sigma)
# s: choke quantity 

function create_demand_params(sigma, p0, x0, p_high)
    A_val = p0 * x0^(-1 / sigma)
    return (k=1 / sigma,
        A=A_val,
        s=(p_high / A_val)^sigma)
end

demand = Dict(
    :avi => create_demand_params(-0.4, 0.04, 1204.79, 500.0),
    :gas => create_demand_params(-0.2, 0.127, 2856.73, 500.0),
    :die => create_demand_params(-0.1, 0.3356, 357.28, 500.0),
    :corn => create_demand_params(-0.23, 4.55, 7.1951 + 1.313879, 500.0), # food+DDGS
    :soyoil => create_demand_params(-0.18, 0.465, 14.164, 500.0),
    :soymeal => create_demand_params(-0.941, 423.41, 62.27, 50000.0)
)

# Fuel supply functions: fuel hockey stick = c0 + c1*q + c2*(x-v)^2
c0_vec = [
    2.338,  # 1. jet_fuel
    2.3,               # 2. saf_atj (shared by conv & cs ATJ SAF)
    1.145,               # 3. saf_hefa (shared by conv, cs, nonsoy HEFA SAF, rd_soy, rd_nonsoy)
    2.7,               # 4. gasoline
    0.23,                # 5. ethanol
    2.435,              # 6. diesel
    1.1               # 7. biodiesel (shared by soy and nonsoy biodiesel)
]

c1_vec = [
    0.0,  # 1. jet_fuel
    0.0,     # 2. saf_atj (shared)
    0.0,     # 3. saf_hefa (shared)
    0.0,   # 4. gasoline
    0.0,   # 5. ethanol
    0.0,  # 6. diesel
    0.0     # 7. biodiesel_soy
]

c2_vec = [
    50.0,  # 1. jet_fuel
    5.0,   # 2. saf_atj (shared)
    5.0,   # 3. saf_hefa (shared)
    50.0,  # 4. gasoline
    50.0,  # 5. ethanol
    50.0,  # 6. diesel
    10.0  # 7. biodiesel_soy
]

v_vec = [
    25.34,  # 1. jet_fuel
    6.9,      # 2. saf_atj (shared)
    15.0,     # 3. saf_hefa (shared)
    130.61,  # 4. gasoline
    18.01,    # 5. ethanol
    48.9,    # 6. diesel
    5.0    # 7. biodiesel_soy
]

# Create mapping for fuel cost parameters
fuel_goods_cost_map = [
    :jet_fuel,
    :saf_atj_shared,      # Used for both saf_atj_conv and saf_atj_cs
    :saf_hefa_shared,     # Used for saf_hefa_conv, saf_hefa_cs, saf_hefa_nonsoy, rd_soy, rd_nonsoy
    :gasoline,
    :ethanol,
    :diesel,
    :biodiesel_shared # Used for both biodiesel_soy and biodiesel_nonsoy
]

# Fossil fuel supply: constant elasticity instead of perfectly elastic.
const FOSSIL_GROUPS = (:jet_fuel, :gasoline, :diesel)
const fossil_eta = 2.0
const fossil_Q0 = Dict(
    :jet_fuel => 20.3386,
    :gasoline => 125.613,
    :diesel => 43.9,
)

fuel_cost = Dict()
for (key, c0, c1, c2, v) in zip(fuel_goods_cost_map, c0_vec, c1_vec, c2_vec, v_vec)
    fuel_cost[key] = key in FOSSIL_GROUPS ?
                     (c0=c0, c1=c1, c2=c2, v=v, eta=fossil_eta, Q0=fossil_Q0[key]) :
                     (c0=c0, c1=c1, c2=c2, v=v, eta=Inf, Q0=0.0)
end

# if the fuel is fossil, then the supply is constant elasticity; otherwise, it is perfectly elastic.
fossil_ces(fc) = isfinite(fc.eta)

function supply_ps(fc, Q)
    over = max(0.0, Q - fc.v)
    kink = fc.c2 * (Q * over^2 - over^3 / 3)
    fossil_ces(fc) || return fc.c1 / 2 * Q^2 + kink
    return fc.c0 * fc.Q0 * (Q / fc.Q0)^(1 + 1 / fc.eta) / (1 + fc.eta) + kink
end

# land supply functions: land = L0*(r/r0)^ϵ
land_supply = (
    L0=0.11768, # baseline land use
    r0_land=587.71, # baseline land rent
    ϵ_land=0.1  # land supply elasticity
)

supply = (
    fuel=fuel_cost,
    land=land_supply
)

# fixed share of corn to soybeans
ω = 0.54

# κ: fixed costs of climate-smart practice adoption
κ = 19.0  # $ per acre

# exogenously fixed non-soy feedstock price ($/lb)
# Non-soy supply is perfectly elastic at this price up to nonsoy_capacity, then vertical.
const nonsoy_feedstock_price = 0.4

# non-soy feedstock availability ceiling (billion lb)
const nonsoy_capacity = 25.0

# HEFA SAF additional processing cost compared to RD ($/gal)
const hefa_saf_premium = 0.064

# Feedstock slate adjustment f_k = (rho_k/2)(A_k - s_bar_k F_k)^2 / F_k, A_k = non-soy lb and
# F_k = total lb on route k. Without it the soy and non-soy variants are perfect substitutes
# and their split is indeterminate. rho = 0.1 regularizes; it does not calibrate the slate.
ρ_pretreat = Dict(
    :bd => 0.1,
    :rd => 0.1,
    :hefa => 0.1,
)

# Floor on the feedstock denominator (B lb). A route can be idle (HEFA-SAF is zero in the
# status quo), and s = A/F is then 0/0. The floor keeps both the marginal terms and their
# derivatives bounded; at 1e-3 B lb against ~19 B lb of throughput it is numerically invisible.
const PRETREAT_FLOOR = 1e-3

# Floor inside the fractional power of the fossil supply curve (B gal).  d/dQ of Q^(1/eta) is
# unbounded at Q = 0; at 1e-6 against ~20 B gal of throughput the floor is numerically invisible.
const SUPPLY_FLOOR = 1e-6

# One route per fuel, not one per plant. Pooling HEFA-SAF with RD (they share a facility) pins
# the pool's slate but leaves the allocation of non-soy BETWEEN them free, which is the same
# indeterminacy one level up, and it bites under the volumetric RFS, where nothing else
# distinguishes the two. Separate routes make every quantity determinate.
const PRETREAT_ROUTES = Dict(
    :bd => (ns=[:biodiesel_nonsoy],
        soy=[:biodiesel_soy],
        sbar=0.431),
    :rd => (ns=[:rd_nonsoy],
        soy=[:rd_soy],
        sbar=0.731),
    # No base-year slate data for HEFA-SAF. It runs through the same hydrotreater as RD, so it
    # is given RD's slate and RD's ρ.
    :hefa => (ns=[:saf_hefa_nonsoy],
        soy=[:saf_hefa_conv, :saf_hefa_cs],
        sbar=0.731),
)

# coefficient values for the model
coeff = (
    delta=δ,
    gamma=γ,
    r=r,
    beta=β,
    alpha=α,
    theta=θ,
    omega=ω,
    kappa=κ,
    soybean_to_oil=soybean_to_oil,
    soybean_to_meal=soybean_to_meal,
    meal_per_oil=meal_per_oil,
    delta_mj=δ_mj,
    baselineCI=baselineCI,
    nonsoy_feedstock_price=nonsoy_feedstock_price,
    nonsoy_capacity=nonsoy_capacity,
    rho_pretreat=ρ_pretreat,
    hefa_saf_premium=hefa_saf_premium
)

# policy_ci(g, config, delta): the carbon intensity the REGULATOR uses as the tax and credit
# BASE for fuel `g`. This is a policy quantity, not a physical one. Actual emissions always use
# the true delta[g] (see calculate_emissions_detail); nothing here changes them.
#
#   recognize_cs = true   every fuel is priced at its own CI, so a CS pathway is credited for
#                         being cleaner (saf_atj_cs = 0.004674).
#   recognize_cs = false  the regulator cannot verify the climate-smart practice, so a CS
#                         pathway is priced AS ITS CONVENTIONAL COUNTERPART.
policy_ci(g, config, delta) =
    (g == :saf_atj_cs && !config.recognize_cs) ? delta[:saf_atj_conv] :
    (g == :saf_hefa_cs && !config.recognize_cs) ? delta[:saf_hefa_conv] :
    delta[g]

# saf_credit(g, config, coeff): per-gallon tax credit paid to aviation fuel `g`
function saf_credit(g, config, coeff)
    delta = coeff.delta
    g in SAF_GOODS || return 0.0
    ci = policy_ci(g, config, delta)
    (config.use_ci_threshold && ci > 0.5 * delta[:jet_fuel]) && return 0.0
    return config.p * max(0.0, delta[:jet_fuel] - ci)
end

# =====================
# meta
# =====================
meta = Dict(
    :process_labels => Dict(
        :feedstock_corn_n => "Feedstock: Conventional corn",
        :feedstock_corn_cs => "Feedstock: Climate-smart corn",
        :feedstock_soy_n => "Feedstock: Conventional soy",
        :feedstock_soy_cs => "Feedstock: Climate-smart soy",
        :jet_fuel => "Jet fuel",
        :saf_atj_conv => "Conventional ATJ-SAF",
        :saf_atj_cs => "Climate-smart ATJ-SAF",
        :saf_hefa_conv => "Conventional HEFA-SAF",
        :saf_hefa_cs => "Climate-smart HEFA-SAF",
        :saf_hefa_nonsoy => "Non soy HEFA-SAF",
        :gasoline => "Gasoline",
        :ethanol => "Ethanol",
        :diesel => "Diesel",
        :biodiesel_soy => "Soy Biodiesel",
        :biodiesel_nonsoy => "Non soy Biodiesel",
        :rd_soy => "Soy Renewable Diesel",
        :rd_nonsoy => "Non soy Renewable diesel",
        :corn => "Food: Corn",
        :soyoil => "Food: Soy oil"
    ),
    :scenario_labels => Dict(
        :statusquo => "Status quo",
        :carbon_tax => "Carbon tax",
        :rfs => "Volumetric mandate",
        :lcfs => "CI standard",
        :tax_credit => "Tax credit",
    )
)

# put them into a container
sets = (
    processes=GOODS,
    scenarios=SCENARIOS,
    sectors=SECTORS,
    fuel_goods=FUEL_GOODS,
    feedstock_goods=FEEDSTOCK_GOODS,
    food_goods=FOOD_GOODS
)

params = (
    sets=sets,
    meta=meta,
    coeff=coeff,
    demand=demand,
    supply=supply
)

# =================================================================================
# 2. Model Building
# =================================================================================

function build_unified_model(params, config)

    model = Model(PATHSolver.Optimizer)
    set_silent(model)
    # Unpack parameters
    coeff = params.coeff
    demand = params.demand
    supply = params.supply
    meta = params.meta
    r = coeff.r
    beta = coeff.beta
    theta = coeff.theta
    alpha = coeff.alpha
    delta = coeff.delta
    gamma = coeff.gamma
    soybean_to_oil = coeff.soybean_to_oil
    soybean_to_meal = coeff.soybean_to_meal
    meal_per_oil = coeff.meal_per_oil
    fuel_cost = supply.fuel
    land_supply = supply.land
    omega = coeff.omega
    kappa = coeff.kappa
    delta_mj = coeff.delta_mj
    baselineCI = coeff.baselineCI
    nonsoy_feedstock_price = coeff.nonsoy_feedstock_price
    nonsoy_capacity = coeff.nonsoy_capacity
    hefa_saf_premium = coeff.hefa_saf_premium
    rho_pretreat = get(coeff, :rho_pretreat, Dict{Symbol,Float64}())
    L0 = land_supply.L0
    r0_land = land_supply.r0_land
    ϵ_land = land_supply.ϵ_land

    # Product groups    
    FEEDSTOCK_GOODS = [:feedstock_corn_n, :feedstock_corn_cs, :feedstock_soy_n, :feedstock_soy_cs]
    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    BIODIESEL_GOODS = [:biodiesel_soy, :biodiesel_nonsoy]
    RD_GOODS = [:rd_soy, :rd_nonsoy]
    CORN_PRODUCTS = [:saf_atj_conv, :saf_atj_cs, :ethanol, :corn]
    SOY_PRODUCTS = [:saf_hefa_conv, :saf_hefa_cs, :biodiesel_soy, :rd_soy, :soyoil]
    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ALL_GOODS = union(AVIATION_FUELS, [:gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy, :corn, :soyoil])

    # =====================
    # Variables
    # =====================

    # price and quantities
    @variables model begin
        x[s in SECTORS] >= 0.1 # demand quantities
        q[g in FUEL_GOODS] >= 0 # supply quantities
        p_f[f in FEEDSTOCK_GOODS] >= 0 # feedstock price
        p_c[s in [:avi, :gas, :die, :soymeal]] >= 0 # consumer's price on x
        l_n >= 0 # conventional land use
        l_cs >= 0 # climate-smart land use
    end

    # Common dual variables 
    @variable(model, λ_rfs >= 0)
    @variable(model, λ_blendwall_ethanol >= 0)
    @variable(model, λ_blendwall_biodiesel >= 0)
    @variable(model, λ_nonsoy_capacity >= 0)

    # Policy-specific dual variables 
    @variable(model, λ_rfs_avi >= 0)      # RFS aviation
    @variable(model, λ_lcfs >= 0)         # LCFS

    # Land rent
    @variable(model, r_land >= 1)

    # Variable naming
    for g in FUEL_GOODS
        set_name(q[g], meta[:process_labels][g])
    end

    # =====================
    # Consumer's utility max
    # =====================

    # consumer's marginal benefit for each sectors
    @expression(model, marginal_benefit[s in SECTORS],
        demand[s].A * x[s]^(demand[s].k))

    # price mapping
    sector_prices = Dict(
        :avi => p_c[:avi],
        :gas => p_c[:gas],
        :die => p_c[:die],
        :corn => p_f[:feedstock_corn_n], # consumer buys food at the feedstock price
        :soyoil => p_f[:feedstock_soy_n],
        :soymeal => p_c[:soymeal]
    )

    # Consumer's u-max conditions (P=MB)
    @constraint(model, consumer_condition[s in SECTORS],
        sector_prices[s] - marginal_benefit[s] ⟂ x[s]
    )

    # =====================
    # Price per unit of fuels
    # =====================

    @expression(model, price_per_unit[g in GOODS],
        if g == :jet_fuel
            r[:jet_fuel] * p_c[:avi]
        elseif g in SAF_GOODS
            r[:jet_fuel] * beta[(:saf, :jet_fuel)] * p_c[:avi]
        elseif g == :gasoline
            r[:gasoline] * p_c[:gas]
        elseif g == :ethanol
            r[:gasoline] * beta[(:ethanol, :gasoline)] * p_c[:gas]
        elseif g == :diesel
            r[:diesel] * p_c[:die]
        elseif g in BIODIESEL_GOODS
            r[:diesel] * beta[(:biodiesel, :diesel)] * p_c[:die]
        elseif g in RD_GOODS
            r[:diesel] * beta[(:rd, :diesel)] * p_c[:die]
        elseif g == :corn
            p_f[:feedstock_corn_n]
        elseif g == :soyoil
            p_f[:feedstock_soy_n]
        elseif g == :soymeal
            p_c[:soymeal]
        else
            0.0
        end
    )

    # =====================
    # Upstream farmers
    # =====================
    # feedstock production (soybeans to soybean oil crushing conversion =10.71lb/bushel)
    # n: conventional, cs: climate-smart
    @expression(model, q_corn_n, omega * gamma[:feedstock_corn_n] * l_n)
    @expression(model, q_corn_cs, omega * gamma[:feedstock_corn_cs] * l_cs)
    @expression(model, q_soy_n, (1 - omega) * gamma[:feedstock_soy_n] * l_n * soybean_to_oil)
    @expression(model, q_soy_cs, (1 - omega) * gamma[:feedstock_soy_cs] * l_cs * soybean_to_oil)

    # Marginal revenue for two types of farmers
    @expression(model, marginal_revenue_n,
        omega * gamma[:feedstock_corn_n] * p_f[:feedstock_corn_n] +
        (1 - omega) * gamma[:feedstock_soy_n] * (p_f[:feedstock_soy_n] * soybean_to_oil + p_c[:soymeal] * soybean_to_meal)
    )

    @expression(model, marginal_revenue_cs,
        omega * gamma[:feedstock_corn_cs] * p_f[:feedstock_corn_cs] +
        (1 - omega) * gamma[:feedstock_soy_cs] * (p_f[:feedstock_soy_cs] * soybean_to_oil + p_c[:soymeal] * soybean_to_meal)
    )

    # zero profit conditions for two types of farmers
    @constraint(model,
        r_land - marginal_revenue_n ⟂ l_n
    )
    @constraint(model,
        r_land + kappa - marginal_revenue_cs ⟂ l_cs
    )

    # =====================
    # Downstream producers
    # =====================

    # Total SAF production by type for processing cost calculation
    @expression(model, total_saf_atj,
        q[:saf_atj_conv] + q[:saf_atj_cs]
    )
    @expression(model, total_saf_hefa,
        q[:saf_hefa_conv] + q[:saf_hefa_cs] + q[:saf_hefa_nonsoy] +
        q[:rd_soy] + q[:rd_nonsoy]
    ) # HEFA SAF and RD share the same production facility. But HEFA SAF needs additional processing that costs HEFA premium

    @expression(model, total_bd,
        q[:biodiesel_soy] + q[:biodiesel_nonsoy]
    )

    # Processing costs for different fuel types
    @expression(model, process_mc_atj,
        fuel_cost[:saf_atj_shared].c0 +
        fuel_cost[:saf_atj_shared].c1 * total_saf_atj +
        fuel_cost[:saf_atj_shared].c2 * max(0, total_saf_atj - fuel_cost[:saf_atj_shared].v)^2
    )

    @expression(model, process_mc_hefa,
        fuel_cost[:saf_hefa_shared].c0 +
        fuel_cost[:saf_hefa_shared].c1 * total_saf_hefa +
        fuel_cost[:saf_hefa_shared].c2 * max(0, total_saf_hefa - fuel_cost[:saf_hefa_shared].v)^2
    )

    @expression(model, process_mc_biodiesel,
        fuel_cost[:biodiesel_shared].c0 +
        fuel_cost[:biodiesel_shared].c1 * total_bd +
        fuel_cost[:biodiesel_shared].c2 * max(0, total_bd - fuel_cost[:biodiesel_shared].v)^2
    )

    # Marginal costs for other fuels that do not have a shared production facility
    # Jet fuel, gasoline and diesel take the constant-elasticity branch (see fuel_cost above);
    # ethanol keeps the hockey stick.  SUPPLY_FLOOR keeps the fractional power away from zero,
    # where its derivative is unbounded; the three fossil quantities never approach it.
    @expression(model, marginal_costs_fuel[g in (:jet_fuel, :gasoline, :diesel, :ethanol)],
        (fossil_ces(fuel_cost[g]) ?
         fuel_cost[g].c0 * ((q[g] + SUPPLY_FLOOR) / fuel_cost[g].Q0)^(1 / fuel_cost[g].eta) :
         fuel_cost[g].c0 + fuel_cost[g].c1 * q[g]) +
        fuel_cost[g].c2 * max(0, q[g] - fuel_cost[g].v)^2
    )

    # Feedstock slate adjustment (see ρ_pretreat above for the derivation).  Marginal terms of
    # f_k = (ρ_k/2)(s_k - s̄_k)^2 F_k; their difference across the two variants is ρ_k(s_k - s̄_k),
    # which is what pins the split.  Skipped entirely when ρ = 0 so the original model is
    # reproduced exactly rather than approximately.
    pretreat_adj = Dict{Symbol,Any}(g => 0.0 for g in FUEL_GOODS)
    for (k, route) in PRETREAT_ROUTES
        ρ_k = get(rho_pretreat, k, 0.0)
        ρ_k == 0.0 && continue
        A_k = sum(alpha[g] * q[g] for g in route.ns)
        F_k = A_k + sum(alpha[g] * q[g] for g in route.soy)
        s̄ = route.sbar
        d_k = @expression(model, (A_k - s̄ * F_k) / (F_k + PRETREAT_FLOOR))   # s_k - s̄_k
        for g in route.ns
            pretreat_adj[g] = @expression(model,
                alpha[g] * ρ_k * (d_k * (1 - s̄) - d_k^2 / 2))
        end
        for g in route.soy
            pretreat_adj[g] = @expression(model,
                alpha[g] * ρ_k * (-d_k * s̄ - d_k^2 / 2))
        end
    end

    # Policy Coefficients (ALL policies included)
    @expression(model, policy_adjustment[g in FUEL_GOODS],
        # Common contraints      
        # 1. Road RFS D6
        λ_rfs * (
            (g == :gasoline || g == :diesel) ? theta :
            (g == :ethanol) ? -1.0 :
            (g in BIODIESEL_GOODS) ? -1.5 :
            (g in RD_GOODS) ? -1.7 : 0.0
        ) +

        # 2. Blend wall constraints (gasoline, ethanol)
        λ_blendwall_ethanol * (
            (g == :gasoline) ? -0.1 :
            (g == :ethanol) ? 0.9 : 0.0
        ) +

        # 3. Blend wall constraints (diesel, biodiesel)
        λ_blendwall_biodiesel * (
            (g == :diesel) ? -0.05 :
            (g in BIODIESEL_GOODS) ? 0.95 : 0.0
        ) +

        # 4. Non-soy capacity
        λ_nonsoy_capacity * (
            (g == :saf_hefa_nonsoy) ? alpha[:saf_hefa_nonsoy] :
            (g == :biodiesel_nonsoy) ? alpha[:biodiesel_nonsoy] :
            (g == :rd_nonsoy) ? alpha[:rd_nonsoy] : 0.0
        ) +
        # 5. RD BD subsidy
        #(-1) * (g in RD_GOODS ? 1.0 : 0.0) + (-1) * (g in BIODIESEL_GOODS ? 1.0 : 0.0) +

        # Policy-specific adjustments (aviation fuels only)
        # 1. Carbon tax.
        config.t * (g in AVIATION_FUELS ? policy_ci(g, config, delta) : 0.0) +
        #config.t * (
        #    config.carbon_tax_scope == :aviation ? (g in AVIATION_FUELS ? delta[g] : 0.0) :
        #    config.carbon_tax_scope == :all ? (g in ALL_GOODS ? delta[g] : 0.0) : 0.0
        #) +


        # 2. RFS aviation mandate
        λ_rfs_avi * (
            g == :jet_fuel ? config.θ_avi :
            g in SAF_GOODS ? (
                ((!config.use_ci_threshold || delta[g] <= 0.5 * delta[:jet_fuel]) &&
                 (config.recognize_cs || g ∉ [:saf_atj_cs, :saf_hefa_cs])) ? -1.6 : 0.0
            ) : 0.0
        ) +

        # 3. LCFS
        λ_lcfs * (
            g in AVIATION_FUELS ? (
                g == :jet_fuel ? -((1 - config.σ) * delta[:jet_fuel] - delta[g]) :
                ((!config.use_ci_threshold || delta[g] <= 0.5 * delta[:jet_fuel]) &&
                 (config.recognize_cs || g ∉ [:saf_atj_cs, :saf_hefa_cs])) ?
                -((1 - config.σ) * delta[:jet_fuel] - delta[g]) : 0.0
            ) : 0.0
        ) +

        #. 4. Tax credit for SAF
        -saf_credit(g, config, coeff)
    )

    # zero profit conditions: marginal cost + policy adjustments - price ⟂ q
    # Conventional ATJ SAF    
    @constraint(model,
        process_mc_atj +
        alpha[:saf_atj_conv] * p_f[:feedstock_corn_n] +
        policy_adjustment[:saf_atj_conv] -
        0.159 * p_f[:feedstock_corn_n] - # Deduct DDGS value
        price_per_unit[:saf_atj_conv]
        ⟂
        q[:saf_atj_conv]
    )

    # Climate-smart ATJ SAF producers
    @constraint(model,
        process_mc_atj +
        alpha[:saf_atj_cs] * p_f[:feedstock_corn_cs] +
        policy_adjustment[:saf_atj_cs] -
        0.159 * p_f[:feedstock_corn_n] - # Deduct DDGS value
        price_per_unit[:saf_atj_cs]
        ⟂
        q[:saf_atj_cs]
    )

    # Conventional HEFA SAF
    @constraint(model,
        process_mc_hefa +
        alpha[:saf_hefa_conv] * p_f[:feedstock_soy_n] +
        policy_adjustment[:saf_hefa_conv] +
        pretreat_adj[:saf_hefa_conv] -
        price_per_unit[:saf_hefa_conv]
        ⟂
        q[:saf_hefa_conv]
    )

    # Climate-smart HEFA SAF producers
    @constraint(model,
        process_mc_hefa +
        alpha[:saf_hefa_cs] * p_f[:feedstock_soy_cs] +
        policy_adjustment[:saf_hefa_cs] +
        pretreat_adj[:saf_hefa_cs] -
        price_per_unit[:saf_hefa_cs]
        ⟂
        q[:saf_hefa_cs]
    )

    # Non-soy HEFA SAF producers
    @constraint(model,
        process_mc_hefa +
        alpha[:saf_hefa_nonsoy] * nonsoy_feedstock_price +
        policy_adjustment[:saf_hefa_nonsoy] +
        pretreat_adj[:saf_hefa_nonsoy] -
        price_per_unit[:saf_hefa_nonsoy]
        ⟂
        q[:saf_hefa_nonsoy]
    )

    # ethanol producers
    @constraint(model,
        marginal_costs_fuel[:ethanol] +
        alpha[:ethanol] * p_f[:feedstock_corn_n] +
        policy_adjustment[:ethanol] -
        0.092 * p_f[:feedstock_corn_n] - # Deduct DDGS value
        price_per_unit[:ethanol]
        ⟂
        q[:ethanol]
    )

    # Soy biodiesel producers
    @constraint(model,
        process_mc_biodiesel +
        alpha[:biodiesel_soy] * p_f[:feedstock_soy_n] +
        policy_adjustment[:biodiesel_soy] +
        pretreat_adj[:biodiesel_soy] -
        price_per_unit[:biodiesel_soy]
        ⟂
        q[:biodiesel_soy]
    )

    # Non Soy biodiesel producers
    @constraint(model,
        process_mc_biodiesel +
        alpha[:biodiesel_nonsoy] * nonsoy_feedstock_price +
        policy_adjustment[:biodiesel_nonsoy] +
        pretreat_adj[:biodiesel_nonsoy] -
        price_per_unit[:biodiesel_nonsoy]
        ⟂
        q[:biodiesel_nonsoy]
    )
    # Soy renewable diesel producers
    @constraint(model,
        process_mc_hefa - hefa_saf_premium +
        alpha[:rd_soy] * p_f[:feedstock_soy_n] +
        policy_adjustment[:rd_soy] +
        pretreat_adj[:rd_soy] -
        price_per_unit[:rd_soy]
        ⟂
        q[:rd_soy]
    )

    # Non Soy Renewable diesel producers
    @constraint(model,
        process_mc_hefa - hefa_saf_premium +
        alpha[:rd_nonsoy] * nonsoy_feedstock_price +
        policy_adjustment[:rd_nonsoy] +
        pretreat_adj[:rd_nonsoy] -
        price_per_unit[:rd_nonsoy]
        ⟂
        q[:rd_nonsoy]
    )

    # Fuels without feedstock costs (fossil fuels + non-soy biofuels)
    FUELS_NO_FEEDSTOCK = [:jet_fuel, :gasoline, :diesel]

    @constraint(model, [g in FUELS_NO_FEEDSTOCK],
        marginal_costs_fuel[g] +
        policy_adjustment[g] -
        price_per_unit[g]
        ⟂
        q[g]
    )

    # =====================
    # Market Clearing Conditions
    # =====================
    # Land market
    @constraint(model, L0 * (r_land / r0_land)^ϵ_land -
                       (l_n + l_cs) ⟂ r_land
    )

    # Upstream Feedstock market
    # total conventional feedstock corn demand : Fuel (conventional ATJ SAF, Ethanol) + food - DDGS
    @expression(model, total_corn_n_demand,
        sum(alpha[g] * q[g] for g in [:saf_atj_conv, :ethanol]) +
        x[:corn] -
        (0.092 * q[:ethanol] + 0.159 * (q[:saf_atj_conv] + q[:saf_atj_cs]))
    )
    @constraint(model, (q_corn_n - total_corn_n_demand ⟂ p_f[:feedstock_corn_n]))

    # total climate-smart feedstock corn demand: Climate-smart ATJ SAF
    @expression(model, total_corn_cs_demand,
        sum(alpha[g] * q[g] for g in [:saf_atj_cs])
    )
    @constraint(model, (q_corn_cs - total_corn_cs_demand ⟂ p_f[:feedstock_corn_cs]))

    # total conventional feedstock soy demand: Fuel (conventional HEFA SAF, Soy Biodiesel, Soy RD) + food
    @expression(model, total_soy_n_demand,
        sum(alpha[g] * q[g] for g in [:saf_hefa_conv, :biodiesel_soy, :rd_soy]) +
        x[:soyoil]
    )
    @constraint(model, (q_soy_n - total_soy_n_demand ⟂ p_f[:feedstock_soy_n]))

    # total climate-smart feedstock soy demand: Climate-smart HEFA SAF
    @expression(model, total_soy_cs_demand,
        sum(alpha[g] * q[g] for g in [:saf_hefa_cs])
    )
    @constraint(model, (q_soy_cs - total_soy_cs_demand ⟂ p_f[:feedstock_soy_cs]))


    # Downstream markets
    @constraint(model,
        r[:jet_fuel] * (q[:jet_fuel] + beta[(:saf, :jet_fuel)] *
                                       sum(q[g] for g in SAF_GOODS)) - x[:avi]
        ⟂
        p_c[:avi]
    )

    @constraint(model,
        r[:gasoline] * (q[:gasoline] + beta[(:ethanol, :gasoline)] * q[:ethanol]) - x[:gas]
        ⟂
        p_c[:gas]
    )

    @constraint(model,
        r[:diesel] * (q[:diesel] +
                      beta[(:biodiesel, :diesel)] * sum(q[g] for g in BIODIESEL_GOODS) +
                      beta[(:rd, :diesel)] * sum(q[g] for g in RD_GOODS)
        ) - x[:die]
        ⟂
        p_c[:die]
    )

    # soy meal market
    @expression(model, total_soymeal_supply,
        meal_per_oil * (total_soy_n_demand + total_soy_cs_demand)
    )
    @constraint(model, (total_soymeal_supply - x[:soymeal] ⟂ p_c[:soymeal]))


    # =====================
    # Policy Constraints
    # =====================
    # Base RFS D6 (always active)
    @constraint(model,
        q[:ethanol] +
        1.5 * sum(q[g] for g in BIODIESEL_GOODS) +
        1.7 * sum(q[g] for g in RD_GOODS) -
        theta * q[:gasoline] - theta * q[:diesel]
        ⟂
        λ_rfs
    )

    # Blend walls (always active)
    @constraint(model,
        0.1 * q[:gasoline] - 0.9 * q[:ethanol]
        ⟂
        λ_blendwall_ethanol
    )

    @constraint(model,
        0.05 * q[:diesel] - 0.95 * sum(q[g] for g in BIODIESEL_GOODS)
        ⟂
        λ_blendwall_biodiesel
    )

    # Non-soy capacity (always active)
    @constraint(model,
        nonsoy_capacity - (alpha[:biodiesel_nonsoy] * q[:biodiesel_nonsoy] + alpha[:rd_nonsoy] * q[:rd_nonsoy] +
                           alpha[:saf_hefa_nonsoy] * q[:saf_hefa_nonsoy])
        ⟂
        λ_nonsoy_capacity
    )

    # RFS aviation (controlled by θ_avi)
    @constraint(model,
        1.6 * sum(
            q[g] for g in SAF_GOODS if
                     (!config.use_ci_threshold || delta[g] <= 0.5 * delta[:jet_fuel]) &&
                     (config.recognize_cs || g ∉ [:saf_atj_cs, :saf_hefa_cs])
        ) - config.θ_avi * q[:jet_fuel]
        ⟂
        λ_rfs_avi
    )


    # LCFS (controlled by σ)
    @constraint(model,
        (1 - config.σ) * delta[:jet_fuel] * (
            q[:jet_fuel] +
            sum(q[g] for g in SAF_GOODS if
                         (!config.use_ci_threshold || delta[g] <= 0.5 * delta[:jet_fuel]) &&
                         (config.recognize_cs || g ∉ [:saf_atj_cs, :saf_hefa_cs]))
        ) -
        (delta[:jet_fuel] * q[:jet_fuel] +
         sum(delta[g] * q[g] for g in SAF_GOODS if
                                 (!config.use_ci_threshold || delta[g] <= 0.5 * delta[:jet_fuel]) &&
                                 (config.recognize_cs || g ∉ [:saf_atj_cs, :saf_hefa_cs])
        ))
        ⟂
        λ_lcfs
    )

    return model
end

# =================================================================================
# 3. Functions for running, extracting results, and writing tables
# =================================================================================

# =====================
# Run Scenario Function
# =====================

function run_scenario(scenario::Symbol, params, policy_configs)

    # Get policy parameters
    #t, θ_avi, σ, p = policy_matrix[scenario]

    # Build config
    #config = (
    #    t=t,
    #    θ_avi=θ_avi,
    #    σ=σ,
    #    p=p
    #)
    config = getproperty(policy_configs, scenario)

    # Build and solve
    model = build_unified_model(params, config)
    optimize!(model)

    if is_solved_and_feasible(model)
        println("\n✓ $(scenario) solved successfully")
        return model
    else
        println("\n✗ $(scenario) failed to solve")
        return nothing
    end
end


# =====================
# Result Extraction (when there is only 1 policy stringency)
# =====================

function extract_solution(model, scenario)
    if isnothing(model)
        return nothing
    end

    # Extract quantities
    q = value.(model[:q])
    x = value.(model[:x])
    p_c = value.(model[:p_c])
    p_f = value.(model[:p_f])

    # Extract land use
    l_n = value(model[:l_n])
    l_cs = value(model[:l_cs])

    # Extract feedstock production
    q_corn_n = value(model[:q_corn_n])
    q_corn_cs = value(model[:q_corn_cs])
    q_soy_n = value(model[:q_soy_n])
    q_soy_cs = value(model[:q_soy_cs])

    # Calculate derived quantities
    ddgs = 0.092 * q[:ethanol] + 0.159 * (q[:saf_atj_conv] + q[:saf_atj_cs])
    soymeal_produ = value(model[:total_soymeal_supply])

    # Extract duals
    duals = (
        λ_rfs=value(model[:λ_rfs]),
        λ_rfs_avi=value(model[:λ_rfs_avi]),
        λ_lcfs=value(model[:λ_lcfs]),
        r_land=value(model[:r_land]),
        λ_blendwall_ethanol=value(model[:λ_blendwall_ethanol]),
        λ_blendwall_biodiesel=value(model[:λ_blendwall_biodiesel]),
        λ_nonsoy_capacity=value(model[:λ_nonsoy_capacity])
    )

    return (
        scenario=scenario,
        q=q,
        x=x,
        p_f=p_f,
        p_c=p_c,
        l_n=l_n,
        l_cs=l_cs,
        q_feedstock=(
            corn_n=q_corn_n,
            corn_cs=q_corn_cs,
            soy_n=q_soy_n,
            soy_cs=q_soy_cs
        ),
        duals=duals,
        ddgs=ddgs,
        soymeal_produ=soymeal_produ
    )
end

# =====================
# Result Extraction (when there are multiple policy stringencies)
# =====================

function extract_all_solutions(results)
    solutions = Dict()
    for (scenario_name, model) in results
        solutions[scenario_name] = extract_solution(model, scenario_name)
    end
    return solutions
end

export params, saf_credit, policy_ci, build_unified_model, run_scenario, extract_solution, extract_all_solutions
export FOSSIL_GROUPS, fossil_ces, supply_ps
export FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS, SAF_GOODS, AVIATION_FUELS

end # module SAFPolicy
