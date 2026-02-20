using JuMP, PATHSolver
using Pkg
using Printf
using Plots
using DataFrames
Pkg.add("DataFrames")

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
    #:observed2024
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

# =====================
# parameters
# =====================

# delta: Carbon Intensity (kg CO2e per MJ) all fuel and food goods
δ_vec = [
    0.01155398, 0.006545966, 0.005282266, 0.004869036, 0.004237186, 0.002486962,
    0.012051015, 0.003570138, 0.013507512, 0.002547826, 0.002680263,
    0.003202355, 0.002521693, 0.006295, 0.00783
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
    :saf_hefa_conv => 8.0,
    :saf_hefa_cs => 8.0,
    :saf_hefa_nonsoy => 8.0,
    :biodiesel_soy => 7.55,
    :biodiesel_nonsoy => 7.55,
    :rd_soy => 7.55,
    :rd_nonsoy => 7.55
)

# theta: road sector RFS D6 blending share
θ = 0.125

# Soybeans to oil conversion factor (lb oil per bushel soybeans)
soybean_to_oil = 10.71

# meal per oil ratio: (million metric ton of meal per billion lb of oil)
meal_per_oil = 2.0116

# delta_mj for IRA credit calculation
δ_mj_vec = [
    89.0,      # 1. jet_fuel
    56.5998,   # 2. saf_atj_conv
    46.5998,   # 3. saf_atj_cs
    23.9854,   # 4. saf_hefa_conv
    13.9854,   # 5. saf_hefa_cs
    17.46,     # 6. saf_hefa_nonsoy
    100.72,    # 7. gasoline
    40.17,     # 8. ethanol
    104.87,    # 9. diesel
    21.64,     # 10. biodiesel_soy
    17.96,     # 11. biodiesel_nonsoy
    24.61,     # 12. rd_soy
    16.69      # 13. rd_nonsoy
]

δ_mj = Dict(g => v for (g, v) in zip(FUEL_GOODS, δ_mj_vec))

# baseline CI for IRA credit calculation
const baselineCI = 50.0

# demand functions: p(x) = p0*(x/x0)^(1/sigma) = A*x^k where sigma < 0, s = x0*(p_sub/p0)^(-1/sigma)
# sigma: price elasticity of demand
# k = 1 / sigma
# A = p0 * x0^(-1/sigma)
# s: quantity at p_high for integral calculation

function create_demand_params(sigma, p0, x0, p_high)
    A_val = p0 * x0^(-1 / sigma)
    return (k=1 / sigma,
        A=A_val,
        s=(p_high / A_val)^sigma)
end

demand = Dict(
    :avi => create_demand_params(-0.4, 0.04, 1204.79, 10.0),
    :gas => create_demand_params(-0.2, 0.127, 2856.73, 10.0),
    :die => create_demand_params(-0.1, 0.3356, 357.28, 10.0),
    :corn => create_demand_params(-0.23, 4.55, 7.1951 + 1.313879, 20.0), # food+DDGS
    :soyoil => create_demand_params(-0.18, 0.465, 14.164, 5.0),
    :soymeal => create_demand_params(-0.941, 423.41, 56.155, 600.0)
)

# Fuel supply functions: fuel hockey stick = c0 + c1*q + c2*(x-v)^2
# Feedstock to SAF production constant
#LOW
c0_vec = [
    2.338,  # 1. jet_fuel
    0.32,               # 2. saf_atj (shared by conv & cs ATJ SAF)
    0.44,               # 3. saf_hefa (shared by conv, cs, nonsoy HEFA SAF, rd_soy, rd_nonsoy)
    2.7,               # 4. gasoline
    0.23,                # 5. ethanol
    2.435,              # 6. diesel
    1.16,               # 7. biodiesel_soy
    1.16                # 8. biodiesel_nonsoy
]
#Medium
c0_vec = [
    0.0575 * 20.3386,  # 1. jet_fuel
    3.97,               # 2. saf_atj (shared by conv & cs ATJ SAF)
    1.44,               # 3. saf_hefa (shared by conv, cs, nonsoy HEFA SAF, rd_soy, rd_nonsoy)
    1.35,               # 4. gasoline
    0.0,                # 5. ethanol
    1.218,              # 6. diesel
    1.16,               # 7. biodiesel_soy
    1.16                # 8. biodiesel_nonsoy
]
#HIGH
c0_vec = [
    2.338,  # 1. jet_fuel
    4.97,               # 2. saf_atj (shared by conv & cs ATJ SAF)
    1.56,               # 3. saf_hefa (shared by conv, cs, nonsoy HEFA SAF, rd_soy, rd_nonsoy)
    2.7,               # 4. gasoline
    0.23,                # 5. ethanol
    2.435,              # 6. diesel
    1.16,               # 7. biodiesel_soy
    1.16                # 8. biodiesel_nonsoy
]
c1_vec = [
    0.0,  # 1. jet_fuel
    0.0,     # 2. saf_atj (shared)
    0.0,     # 3. saf_hefa (shared)
    0.0,   # 4. gasoline
    0.0,   # 5. ethanol
    0.0,  # 6. diesel
    0.0,     # 7. biodiesel_soy
    0.0      # 8. biodiesel_nonsoy
]


c2_vec = [
    50.0,  # 1. jet_fuel
    5.0,   # 2. saf_atj (shared)
    5.0,   # 3. saf_hefa (shared)
    50.0,  # 4. gasoline
    50.0,  # 5. ethanol
    50.0,  # 6. diesel
    10.0,  # 7. biodiesel_soy
    50.0   # 8. biodiesel_nonsoy
]

v_vec = [
    24050.0,  # 1. jet_fuel
    6.9,      # 2. saf_atj (shared)
    15.0,     # 3. saf_hefa (shared)
    42557.0,  # 4. gasoline
    18.01,    # 5. ethanol
    64.68,    # 6. diesel
    2.092,    # 7. biodiesel_soy
    3.0       # 8. biodiesel_nonsoy
]

# Create mapping for fuel cost parameters
fuel_goods_cost_map = [
    :jet_fuel,
    :saf_atj_shared,      # Used for both saf_atj_conv and saf_atj_cs
    :saf_hefa_shared,     # Used for saf_hefa_conv, saf_hefa_cs, saf_hefa_nonsoy, rd_soy, rd_nonsoy
    :gasoline,
    :ethanol,
    :diesel,
    :biodiesel_soy,
    :biodiesel_nonsoy
]

fuel_cost = Dict()
for (key, c0, c1, c2, v) in zip(fuel_goods_cost_map, c0_vec, c1_vec, c2_vec, v_vec)
    fuel_cost[key] = (c0=c0, c1=c1, c2=c2, v=v)
end


# land supply functions: land = L0*(r/r0)^ϵ
land_supply = (
    L0=0.11768, # baseline land use
    r0_land=586.6, # baseline land rent
    ϵ_land=0.1  # land supply elasticity
)

supply = (
    fuel=fuel_cost,
    #feedstock=feedstock_cost,
    land=land_supply
)

# land corn ratio
ω = 0.54

# κ: fixed costs of climate-smart practice adoption
κ = 19.0  # $ per acre

# Non-soy feedstock price ($/lb)
const nonsoy_feedstock_price = 0.48

# HEFA SAF additional processing cost compared to RD ($/gal)
const hefa_saf_premium = 0.08

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
    meal_per_oil=meal_per_oil,
    delta_mj=δ_mj,
    baselineCI=baselineCI,
    nonsoy_feedstock_price=nonsoy_feedstock_price,
    hefa_saf_premium=hefa_saf_premium
)

# helper function for tax credit calculation
function tax_credit_rate(δ_mj_value, baselineCI, p)
    ci_mmbtu = δ_mj_value * 1055.06 / 1000
    emission_factor = max(0.0, (baselineCI - ci_mmbtu) / baselineCI)
    return p * emission_factor
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
        :rfs => "RFS",
        :lcfs => "LCFS",
        :tax_credit => "Tax credit",
        #:obs2024 => "2024 observed"
    )
)

# =====================
# put them into a container
# =====================
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
    meal_per_oil = coeff.meal_per_oil
    fuel_cost = supply.fuel
    land_supply = supply.land
    omega = coeff.omega
    kappa = coeff.kappa
    delta_mj = coeff.delta_mj
    baselineCI = coeff.baselineCI
    nonsoy_feedstock_price = coeff.nonsoy_feedstock_price
    hefa_saf_premium = coeff.hefa_saf_premium
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

    # =====================
    # Variables
    # =====================

    @variables model begin
        x[s in SECTORS] >= 0.1 # demand quantities
        q[g in FUEL_GOODS] >= 0 # supply quantities
        p_f[f in FEEDSTOCK_GOODS] >= 0 # feedstock price
        p_c[s in [:avi, :gas, :die, :soymeal]] >= 0 # consumer's price on x
        l_n >= 0 # conventional land use
        l_cs >= 0 # climate-smart land use

    end

    # Base dual variables 
    @variable(model, λ_rfs >= 0)
    @variable(model, λ_blendwall_ethanol >= 0)
    @variable(model, λ_blendwall_biodiesel >= 0)
    @variable(model, λ_nonsoy_capacity >= 0)

    # Policy-specific dual variables 
    @variable(model, λ_rfs_avi >= 0)      # RFS aviation
    @variable(model, λ_lcfs >= 0)         # LCFS
    @variable(model, r_land >= 1)    # Land rent

    # Variable naming
    for g in FUEL_GOODS
        set_name(q[g], meta[:process_labels][g])
    end

    # =====================
    # Consumer's utility max
    # =====================

    # marginal benefit
    @expression(model, marginal_benefit[s in SECTORS],
        demand[s].A * x[s]^(demand[s].k))

    # price mapping
    sector_prices = Dict(
        :avi => p_c[:avi],
        :gas => p_c[:gas],
        :die => p_c[:die],
        :corn => p_f[:feedstock_corn_n],
        :soyoil => p_f[:feedstock_soy_n],
        :soymeal => p_c[:soymeal]
    )

    # Consumer's u-max conditions
    @constraint(model, consumer_condition[s in SECTORS],
        sector_prices[s] - marginal_benefit[s] ⟂ x[s]
    )

    # =====================
    # Price per unit
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
    # feedstock production (soybeans to soybean oil conversion =10.71lb/bushel)
    @expression(model, q_corn_n, omega * gamma[:feedstock_corn_n] * l_n)
    @expression(model, q_corn_cs, omega * gamma[:feedstock_corn_cs] * l_cs)
    @expression(model, q_soy_n, (1 - omega) * gamma[:feedstock_soy_n] * l_n * soybean_to_oil)
    @expression(model, q_soy_cs, (1 - omega) * gamma[:feedstock_soy_cs] * l_cs * soybean_to_oil)

    # Marginal revenue: Upstream farmers
    @expression(model, marginal_revenue_n,
        omega * gamma[:feedstock_corn_n] * p_f[:feedstock_corn_n] +
        (1 - omega) * gamma[:feedstock_soy_n] * p_f[:feedstock_soy_n] * soybean_to_oil
    )

    @expression(model, marginal_revenue_cs,
        omega * gamma[:feedstock_corn_cs] * p_f[:feedstock_corn_cs] +
        (1 - omega) * gamma[:feedstock_soy_cs] * p_f[:feedstock_soy_cs] * soybean_to_oil
    )

    # zero profit condition
    # conventional farmer
    @constraint(model,
        r_land - marginal_revenue_n ⟂ l_n
    )
    # climate-smart farmer
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
    )

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

    # Marginal costs for other fuels
    @expression(model, marginal_costs_fuel[g in (:jet_fuel, :gasoline, :diesel, :ethanol,
            :biodiesel_soy, :biodiesel_nonsoy)],
        fuel_cost[g].c0 +
        fuel_cost[g].c1 * q[g] +
        fuel_cost[g].c2 * max(0, q[g] - fuel_cost[g].v)^2)


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

        # Policy-specific adjustments (aviation fuels only)
        # 1. Carbon tax
        config.t * (g in AVIATION_FUELS ? delta[g] : 0.0) +

        # 2. RFS aviation mandate
        λ_rfs_avi * (
            (g == :jet_fuel) ? config.θ_avi :
            #(g in SAF_GOODS) ? -1.6 : 0.0
            (g in SAF_GOODS && delta[g] <= 0.5 * delta[:jet_fuel]) ? -1.6 : 0.0 # 50% CI threshold
        ) +

        # 3. LCFS
        λ_lcfs * (g in AVIATION_FUELS ?
                  -((1 - config.σ) * delta[:jet_fuel] - delta[g]) : 0.0
        ) +

        #. 4. Tax credit for SAF
        -(g in SAF_GOODS && haskey(delta_mj, g) ? tax_credit_rate(delta_mj[g], baselineCI, config.p) : 0.0)
    )

    # zero profit conditions
    # marginal cost + policy adjustments - price ⟂ q
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
        process_mc_hefa + hefa_saf_premium +
        alpha[:saf_hefa_conv] * p_f[:feedstock_soy_n] +
        policy_adjustment[:saf_hefa_conv] -
        price_per_unit[:saf_hefa_conv]
        ⟂
        q[:saf_hefa_conv]
    )

    # Climate-smart HEFA SAF producers
    @constraint(model,
        process_mc_hefa + hefa_saf_premium +
        alpha[:saf_hefa_cs] * p_f[:feedstock_soy_cs] +
        policy_adjustment[:saf_hefa_cs] -
        price_per_unit[:saf_hefa_cs]
        ⟂
        q[:saf_hefa_cs]
    )

    # Non-soy HEFA SAF producers
    @constraint(model,
        process_mc_hefa + hefa_saf_premium +
        alpha[:saf_hefa_nonsoy] * nonsoy_feedstock_price +
        policy_adjustment[:saf_hefa_nonsoy] -
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
        marginal_costs_fuel[:biodiesel_soy] +
        alpha[:biodiesel_soy] * p_f[:feedstock_soy_n] +
        policy_adjustment[:biodiesel_soy] -
        price_per_unit[:biodiesel_soy]
        ⟂
        q[:biodiesel_soy]
    )

    # Non Soy biodiesel producers
    @constraint(model,
        marginal_costs_fuel[:biodiesel_nonsoy] +
        alpha[:biodiesel_nonsoy] * nonsoy_feedstock_price +
        policy_adjustment[:biodiesel_nonsoy] -
        price_per_unit[:biodiesel_nonsoy]
        ⟂
        q[:biodiesel_nonsoy]
    )
    # Soy renewable diesel producers
    @constraint(model,
        process_mc_hefa +
        alpha[:rd_soy] * p_f[:feedstock_soy_n] +
        policy_adjustment[:rd_soy] -
        price_per_unit[:rd_soy]
        ⟂
        q[:rd_soy]
    )

    # Non Soy Renewable diesel producers
    @constraint(model,
        process_mc_hefa +
        alpha[:rd_nonsoy] * nonsoy_feedstock_price +
        policy_adjustment[:rd_nonsoy] -
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
        30 - (alpha[:biodiesel_nonsoy] * q[:biodiesel_nonsoy] + alpha[:rd_nonsoy] * q[:rd_nonsoy] +
              alpha[:saf_hefa_nonsoy] * q[:saf_hefa_nonsoy])
        ⟂
        λ_nonsoy_capacity
    )

    # RFS aviation (controlled by θ_avi)
    @constraint(model,
        1.6 * sum(q[g] for g in SAF_GOODS if delta[g] <= 0.5 * delta[:jet_fuel]) - config.θ_avi * q[:jet_fuel] # 50% CI threshold
        # 1.6 * sum(q[g] for g in SAF_GOODS) - config.θ_avi * q[:jet_fuel]
        ⟂
        λ_rfs_avi
    )

    # LCFS (controlled by σ)
    @constraint(model,
        (1 - config.σ) * delta[:jet_fuel] * sum(q[g] for g in AVIATION_FUELS) -
        sum(delta[g] * q[g] for g in AVIATION_FUELS)
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

function run_scenario(scenario::Symbol, params)

    # Get policy parameters
    t, θ_avi, σ, p = POLICY_MATRIX[scenario]

    # Build config
    config = (
        t=t,
        θ_avi=θ_avi,
        σ=σ,
        p=p
    )

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

# =====================
# Display Comparison Tables and individual Tables
# =====================
function display_comparison_tables(solutions, params, policy_configs;
    scenarios=nothing,
    title="RESULTS")

    println("\n" * "="^130)
    println(title)
    println("="^130)

    # Calculate implicit taxes
    implicit_taxes = calculate_all_implicit_taxes(solutions, params, policy_configs)

    tables = [
        ("Implicit Taxes/Subsidies (\$/gallon)", make_implicit_tax_table(implicit_taxes, params; scenarios=scenarios)),
        ("Production", make_production_table(solutions, params; scenarios=scenarios)),
        ("Demand", make_demand_table(solutions; scenarios=scenarios)),
        ("Prices", make_price_table(solutions; scenarios=scenarios)),
        ("Land Use (Million Acres)", make_land_table(solutions, params; scenarios=scenarios)),
        ("Emissions (MMT CO2e)", make_emissions_table(solutions, params; scenarios=scenarios)),
        ("Dual Variables", make_duals_table(solutions; scenarios=scenarios))
    ]

    for (table_name, table) in tables
        println("\n--- $table_name ---")
        show(table, allrows=true)
    end

    println("\n" * "="^130)
end

# =====================
# 1) Implicit Tax/Subsidy Calculation
# =====================

function calculate_implicit_taxes(solution, params, config)
    if isnothing(solution)
        return nothing
    end

    # Tuple unpacking
    t, θ_avi, σ, p = config

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
        implicit_tax[g][:carbon_tax] = t * delta[g]

        # 2. RFS aviation component
        if g == :jet_fuel
            implicit_tax[g][:rfs_avi] = γ_avi * θ_avi
        else  # SAF goods
            implicit_tax[g][:rfs_avi] = (delta[g] <= 0.5 * delta[:jet_fuel]) ? -γ_avi * 1.6 : 0.0
        end

        # 3. LCFS component
        implicit_tax[g][:lcfs] = -μ * ((1 - σ) * delta[:jet_fuel] - delta[g])

        # 4. Tax credit component (SAF only)
        if g != :jet_fuel
            implicit_tax[g][:tax_credit] = -tax_credit_rate(delta_mj[g], baselineCI, p)
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

"""
Calculate implicit taxes for all solutions
"""
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

"""
Create implicit tax table
"""
function make_implicit_tax_table(implicit_taxes, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(implicit_taxes)) : scenarios
    labels = params.meta[:process_labels]

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    # Define which tax component to show for each scenario
    scenario_tax_type = Dict(
        :statusquo => :total,
        :carbontax => :carbon_tax,
        :rfs => :rfs_avi,
        :lcfs => :lcfs,
        :taxcredit => :tax_credit
    )

    df = DataFrame(Fuel=String[])
    for scenario in scenario_list
        df[!, String(scenario)] = Float64[]
    end

    for g in AVIATION_FUELS
        push!(df.Fuel, labels[g])
        for scenario in scenario_list
            if haskey(implicit_taxes, scenario) && !isnothing(implicit_taxes[scenario])
                tax_type = get(scenario_tax_type, scenario, :total)
                value = implicit_taxes[scenario][g][tax_type]
                push!(df[!, String(scenario)], value)
            else
                push!(df[!, String(scenario)], 0.0)
            end
        end
    end

    return df
end

"""
Display implicit taxes with custom formatting
"""
function display_implicit_taxes(implicit_taxes, params; scenarios=nothing, show_statusquo=false)
    labels = params.meta[:process_labels]

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs,
        :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    scenario_tax_type = Dict(
        :carbontax => :carbon_tax,
        :rfs => :rfs_avi,
        :lcfs => :lcfs,
        :taxcredit => :tax_credit
    )

    scenario_labels = Dict(
        :carbontax => "Carbon Tax",
        :rfs => "RFS",
        :lcfs => "LCFS",
        :taxcredit => "Tax Credit"
    )

    # Determine scenarios to display
    if isnothing(scenarios)
        display_scenarios = collect(keys(implicit_taxes))
    else
        display_scenarios = scenarios
    end

    # Exclude statusquo if requested
    if !show_statusquo
        display_scenarios = filter(s -> s != :statusquo, display_scenarios)
    end

    println("\n" * "="^90)
    println("IMPLICIT TAXES/SUBSIDIES BY POLICY (\$/gallon)")
    println("="^90)

    # Header
    @printf("%-30s", "Fuel")
    for scenario in display_scenarios
        label = get(scenario_labels, scenario, String(scenario))
        @printf("%15s", label)
    end
    println()
    println("-"^90)

    # Data rows
    for g in AVIATION_FUELS
        @printf("%-30s", labels[g])
        for scenario in display_scenarios
            if haskey(implicit_taxes, scenario) && !isnothing(implicit_taxes[scenario])
                tax_type = get(scenario_tax_type, scenario, :total)
                value = implicit_taxes[scenario][g][tax_type]
                @printf("%15.4f", value)
            else
                @printf("%15s", "N/A")
            end
        end
        println()
    end

    println("="^90)
end

# =====================
# 2) Production, Demand, Price, Land Use, Emissions, Duals Tables
# =====================
function make_production_table(solutions, params; scenarios=nothing)
    labels = params.meta[:process_labels]
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(Product=String[])
    for scenario in scenario_list
        df[!, scenario] = Float64[]
    end

    # Fuel goods
    for g in FUEL_GOODS
        push!(df.Product, labels[g])
        for scenario in scenario_list
            push!(df[!, scenario], solutions[scenario].q[g])
        end
    end

    # Feedstock
    feedstock_goods = [:feedstock_corn_n, :feedstock_corn_cs, :feedstock_soy_n, :feedstock_soy_cs]
    feedstock_keys = [:corn_n, :corn_cs, :soy_n, :soy_cs]

    for (g, key) in zip(feedstock_goods, feedstock_keys)
        push!(df.Product, labels[g])
        for scenario in scenario_list
            push!(df[!, scenario], solutions[scenario].q_feedstock[key])
        end
    end

    # DDGS
    push!(df.Product, "DDGS")
    for scenario in scenario_list
        push!(df[!, scenario], solutions[scenario].ddgs)
    end

    # Soymeal
    push!(df.Product, "Soymeal Production (MMT)")
    for scenario in scenario_list
        push!(df[!, scenario], solutions[scenario].soymeal_produ)
    end

    return df
end

function make_demand_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(
        Scenario=String[],
        Aviation=Float64[],
        Gasoline=Float64[],
        Diesel=Float64[],
        Corn=Float64[],
        Corn_excl_DDGS=Float64[],
        Soyoil=Float64[],
        Soymeal=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]
        push!(df, (
            String(scenario),
            sol.x[:avi],
            sol.x[:gas],
            sol.x[:die],
            sol.x[:corn],
            sol.x[:corn] - sol.ddgs,
            sol.x[:soyoil],
            sol.x[:soymeal]
        ))
    end

    return df
end

function make_price_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(
        Scenario=String[],
        Aviation=Float64[],
        Gasoline=Float64[],
        Diesel=Float64[],
        ConvCorn=Float64[],
        CSCorn=Float64[],
        ConvSoy=Float64[],
        CSSoy=Float64[],
        Soymeal=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]
        push!(df, (
            String(scenario),
            sol.p_c[:avi],
            sol.p_c[:gas],
            sol.p_c[:die],
            sol.p_f[:feedstock_corn_n],
            sol.p_f[:feedstock_corn_cs],
            sol.p_f[:feedstock_soy_n],
            sol.p_f[:feedstock_soy_cs],
            sol.p_c[:soymeal]
        ))
    end

    return df
end

function make_land_table(solutions, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    omega = params.coeff.omega

    df = DataFrame(
        Scenario=String[],
        Corn_N=Float64[],
        Corn_CS=Float64[],
        Corn_Total=Float64[],
        Soy_N=Float64[],
        Soy_CS=Float64[],
        Soy_Total=Float64[],
        Total_N=Float64[],
        Total_CS=Float64[],
        Grand_Total=Float64[],
        Land_Rent=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]

        corn_land_n = omega * sol.l_n
        soy_land_n = (1 - omega) * sol.l_n
        corn_land_cs = omega * sol.l_cs
        soy_land_cs = (1 - omega) * sol.l_cs

        total_corn = corn_land_n + corn_land_cs
        total_soy = soy_land_n + soy_land_cs
        total_n = sol.l_n
        total_cs = sol.l_cs
        total = sol.l_n + sol.l_cs

        push!(df, (
            String(scenario),
            1000 * corn_land_n,
            1000 * corn_land_cs,
            1000 * total_corn,
            1000 * soy_land_n,
            1000 * soy_land_cs,
            1000 * total_soy,
            1000 * total_n,
            1000 * total_cs,
            1000 * total,
            sol.duals.r_land
        ))
    end

    return df
end

function make_emissions_table(solutions, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    FOOD_GOODS = [:corn, :soyoil]

    delta = params.coeff.delta

    df = DataFrame(
        Scenario=String[],
        Aviation=Float64[],
        Road=Float64[],
        Food=Float64[],
        Total=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]

        avi_emission = sum(delta[g] * sol.q[g] for g in AVIATION_FUELS)
        road_emission = sum(delta[g] * sol.q[g] for g in ROAD_FUELS)
        food_emission = sum(delta[g] * sol.x[g] for g in FOOD_GOODS)

        push!(df, (
            String(scenario),
            avi_emission * 1000,
            road_emission * 1000,
            food_emission * 1000,
            (avi_emission + road_emission + food_emission) * 1000
        ))
    end

    return df
end

function make_duals_table(solutions; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios

    df = DataFrame(
        Scenario=String[],
        λ_rfs=Float64[],
        λ_rfs_avi=Float64[],
        λ_lcfs=Float64[],
        r_land=Float64[],
        λ_blendwall_ethanol=Float64[],
        λ_blendwall_biodiesel=Float64[],
        λ_nonsoy_capacity=Float64[]
    )

    for scenario in scenario_list
        sol = solutions[scenario]
        push!(df, (
            String(scenario),
            sol.duals.λ_rfs,
            sol.duals.λ_rfs_avi,
            sol.duals.λ_lcfs,
            sol.duals.r_land,
            sol.duals.λ_blendwall_ethanol,
            sol.duals.λ_blendwall_biodiesel,
            sol.duals.λ_nonsoy_capacity
        ))
    end

    return df
end

"""
Display all comparison tables
"""
function display_equivalent_policy_results(equivalent_solutions, params, target_saf, equivalent_policies)

    policy_labels = Dict(
        :carbontax => ("Carbon Tax", "Carbon Tax (\$/ton CO2e)"),
        :rfs => ("RFS Aviation", "Mandate Share"),
        :lcfs => ("LCFS", "CI Reduction (σ)"),
        :taxcredit => ("Tax Credit", "Rate (\$/gal)")
    )

    println("\n" * "="^130)
    println("EQUIVALENT POLICY COMPARISON (Target SAF = $(target_saf) billion gallons)")
    println("="^130)

    # Policy Parameters
    println("\n--- Policy Parameters ---")
    param_df = DataFrame(
        Policy=String[],
        Parameter_Name=String[],
        Parameter_Value=Float64[],
        Actual_SAF=Float64[]
    )

    for policy_type in [:carbontax, :rfs, :lcfs, :taxcredit]
        result = equivalent_policies[policy_type]
        config = result.config
        label, param_name = policy_labels[policy_type]

        param_value = if policy_type == :carbontax
            config.t
        elseif policy_type == :rfs
            config.θ_avi
        elseif policy_type == :lcfs
            config.σ
        else
            config.p
        end

        push!(param_df, (label, param_name, param_value, result.actual_saf))
    end

    show(param_df, allrows=true)

    # Production
    println("\n\n--- Production ---")
    prod_table = make_production_table(equivalent_solutions, params;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit])
    show(prod_table, allrows=true)

    # Demand
    println("\n\n--- Demand ---")
    demand_table = make_demand_table(equivalent_solutions;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit])
    show(demand_table, allrows=true)

    # Prices
    println("\n\n--- Prices ---")
    price_table = make_price_table(equivalent_solutions;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit])
    show(price_table, allrows=true)

    # Land Use
    println("\n\n--- Land Use (Million Acres) ---")
    land_table = make_land_table(equivalent_solutions, params;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit])
    show(land_table, allrows=true)

    # Emissions
    println("\n\n--- Emissions (MMT CO2e) ---")
    emissions_table = make_emissions_table(equivalent_solutions, params;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit])
    show(emissions_table, allrows=true)

    # Duals
    println("\n\n--- Dual Variables ---")
    duals_table = make_duals_table(equivalent_solutions;
        scenarios=[:carbontax, :rfs, :lcfs, :taxcredit])
    show(duals_table, allrows=true)

    println("\n" * "="^130)
end

# =================================================================================
# 4. Run Base Scenarios
# =================================================================================

# Define policy stringency
const POLICY_MATRIX = (
    #             t     θ_avi  σ    p
    statusquo=(0.0, 0.0, 0.0, 0.0),
    carbontax=(250.0, 0.0, 0.0, 0.0),
    rfs=(0.0, 0.3, 0.0, 0.0),
    lcfs=(0.0, 0.0, 0.03, 0.0),
    taxcredit=(0.0, 0.0, 0.0, 10.0)
)

# Run all scenarios
results = Dict()
for scenario in SCENARIOS
    results[scenario] = run_scenario(scenario, params)
end

# Extract solutions
solutions = extract_all_solutions(results)

# Display all results
println("\n" * "="^80)
println("POLICY STRINGENCY")
println("="^80)

for scenario in SCENARIOS
    t, θ_avi, σ, p = POLICY_MATRIX[scenario]

    stringency = if scenario == :statusquo
        "Status Quo (no policy)"
    elseif scenario == :carbontax
        "t = $t"
    elseif scenario == :rfs
        "θ_avi = $θ_avi"
    elseif scenario == :lcfs
        "σ = $σ"
    elseif scenario == :taxcredit
        "p = $p"
    else
        ""
    end

    println("$scenario: $stringency")
end

println("="^80)

display_comparison_tables(solutions, params, POLICY_MATRIX;
    scenarios=SCENARIOS,
    title="BASE SCENARIO COMPARISON")

# =================================================================================
# 5. Extended Policy Parameter Grid
# =================================================================================

# Define policy ranges
const POLICY_RANGES = (
    t=0:100:1500,           # Carbon tax: 0, 10, 20, ..., 400
    θ_avi=0:0.05:1.0,     # RFS aviation: 0, 0.05, 0.1, ..., 0.5
    σ=0.0:0.05:1.0,      # LCFS: 1.0, 0.99, 0.98, ..., 0.5
    p=0:15:450             # Tax credit: 0, 1, 2, ..., 100
)

# Create policy scenarios based on which parameter varies
function create_policy_scenarios()
    scenarios = Dict()

    # 1. Carbon tax scenarios (varying t)
    for t in POLICY_RANGES.t
        scenarios[Symbol("carbontax_$(Int(t))")] = (t=Float64(t), θ_avi=0.0, σ=0.0, p=0.0)
    end

    # 2. RFS aviation scenarios (varying θ_avi)
    for θ in POLICY_RANGES.θ_avi
        # Use round to avoid floating point precision issues
        θ_int = round(Int, θ * 100)
        scenarios[Symbol("rfs_$θ_int")] = (t=0.0, θ_avi=Float64(θ), σ=0.0, p=0.0)
    end

    # 3. LCFS scenarios (varying σ)
    for σ in POLICY_RANGES.σ
        # Use round to avoid floating point precision issues
        σ_int = round(Int, σ * 100)
        scenarios[Symbol("lcfs_$σ_int")] = (t=0.0, θ_avi=0.0, σ=Float64(σ), p=0.0)
    end

    # 4. Tax credit scenarios (varying p)
    for p in POLICY_RANGES.p
        scenarios[Symbol("taxcredit_$(Int(p))")] = (t=0.0, θ_avi=0.0, σ=0.0, p=Float64(p))
    end

    # Add status quo
    scenarios[:statusquo] = (t=0.0, θ_avi=0.0, σ=0.0, p=0.0)
    return scenarios
end

const EXTENDED_POLICY_MATRIX = create_policy_scenarios()

# =====================
# Run the extended grid scenarios
# =====================

function run_all_scenarios(params; verbose=true)
    results = Dict()
    solutions = Dict()

    total_scenarios = length(EXTENDED_POLICY_MATRIX)
    current = 0

    for (scenario_name, config) in EXTENDED_POLICY_MATRIX
        current += 1

        if verbose
            println("\n[$current/$total_scenarios] Running $scenario_name...")
        end

        try
            model = build_unified_model(params, config)
            optimize!(model)

            if is_solved_and_feasible(model)
                results[scenario_name] = model
                solutions[scenario_name] = extract_solution(model, scenario_name)

                if verbose
                    println("  ✓ Solved successfully")
                end
            else
                if verbose
                    println("  ✗ Failed to solve")
                end
                results[scenario_name] = nothing
                solutions[scenario_name] = nothing
            end
        catch e
            if verbose
                println("  ✗ Error: $e")
            end
            results[scenario_name] = nothing
            solutions[scenario_name] = nothing
        end
    end

    return results, solutions
end

# Run all scenarios
println("\nRunning extended policy analysis...")
all_results, all_solutions = run_all_scenarios(params, verbose=true)


# =====================
# Plot the extended grid results
# =====================

# Convert solutions to DataFrame
function solutions_to_dataframe(all_solutions, params)

    AVIATION_FUELS = [:jet_fuel, :saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]
    ROAD_FUELS = [:gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy]
    FOOD_GOODS = [:corn, :soyoil]
    delta = params.coeff.delta

    results_df = DataFrame(
        scenario=String[],
        policy_type=String[],
        t=Float64[],
        θ_avi=Float64[],
        σ=Float64[],
        p=Float64[]
    )

    # Add fuel quantity columns
    for fuel in AVIATION_FUELS
        results_df[!, Symbol("q_$fuel")] = Float64[]
    end

    for fuel in ROAD_FUELS
        results_df[!, Symbol("q_$fuel")] = Float64[]
    end

    # Add demand columns
    for sector in [:avi, :gas, :die, :corn, :soyoil, :soymeal]
        results_df[!, Symbol("x_$sector")] = Float64[]
    end

    # Add price columns
    results_df[!, :p_avi] = Float64[]
    results_df[!, :p_gas] = Float64[]
    results_df[!, :p_die] = Float64[]
    results_df[!, :r_land] = Float64[]

    # Add emission columns
    results_df[!, :emission_avi] = Float64[]
    results_df[!, :emission_road] = Float64[]
    results_df[!, :emission_food] = Float64[]
    results_df[!, :emission_total] = Float64[]

    # Fill DataFrame
    for (scenario_name, sol) in all_solutions
        if isnothing(sol)
            continue
        end

        scenario_str = String(scenario_name)
        config = EXTENDED_POLICY_MATRIX[scenario_name]

        # Determine policy type
        if occursin("carbontax", scenario_str)
            policy_type = "carbontax"
        elseif occursin("rfs", scenario_str)
            policy_type = "rfs"
        elseif occursin("lcfs", scenario_str)
            policy_type = "lcfs"
        elseif occursin("taxcredit", scenario_str)
            policy_type = "taxcredit"
        else
            policy_type = "statusquo"
        end

        row = [
            scenario_str,
            policy_type,
            config.t,
            config.θ_avi,
            config.σ,
            config.p
        ]

        # Aviation fuels
        for fuel in AVIATION_FUELS
            push!(row, sol.q[fuel])
        end

        # Road fuels
        for fuel in ROAD_FUELS
            push!(row, sol.q[fuel])
        end

        # Demand
        for sector in [:avi, :gas, :die, :corn, :soyoil, :soymeal]
            push!(row, sol.x[sector])
        end

        # Prices
        push!(row, sol.p_c[:avi])
        push!(row, sol.p_c[:gas])
        push!(row, sol.p_c[:die])
        push!(row, sol.duals.r_land)

        # Emissions
        avi_emission = sum(delta[g] * sol.q[g] for g in AVIATION_FUELS)
        road_emission = sum(delta[g] * sol.q[g] for g in ROAD_FUELS)
        food_emission = sum(delta[g] * sol.x[g] for g in FOOD_GOODS)

        push!(row, avi_emission * 1000)
        push!(row, road_emission * 1000)
        push!(row, food_emission * 1000)
        push!(row, (avi_emission + road_emission + food_emission) * 1000)

        push!(results_df, row)
    end

    return results_df
end

# Plot fuel production by policy grid
function plot_fuel_production(results_df, fuel_config)

    policies = [
        ("carbontax", :t, "Carbon Tax (\$ / ton CO2e)", "Carbon Tax Policy", false),
        ("rfs", :θ_avi, "RFS Aviation Mandate (θ_avi)", "RFS Aviation Policy", false),
        ("lcfs", :σ, "LCFS Standard (σ)", "LCFS Policy", true),
        ("taxcredit", :p, "Tax Credit Rate (\$ / gallon)", "Tax Credit Policy", false)
    ]

    plots = []

    for (i, (policy_type, xcol, xlabel, title, reverse_sort)) in enumerate(policies)
        df = filter(row -> row.policy_type == policy_type, results_df)
        sort!(df, xcol, rev=reverse_sort)

        x_min = minimum(df[!, xcol])
        x_max = maximum(df[!, xcol])

        # Main fuel plot
        p_main = plot(
            xlabel="",
            ylabel="$(fuel_config.main_fuel_label)\n(billion gallons)",
            title=title,
            titlefontsize=16,
            titlefontweight=:bold,
            legend=false,
            grid=true,
            xlims=(x_min, x_max),
            xticks=:none,
            bottom_margin=-1Plots.mm,
            top_margin=10Plots.mm,
            left_margin=12Plots.mm
        )

        plot!(p_main, df[!, xcol], df[!, fuel_config.main_fuel],
            linewidth=2, marker=:circle, markersize=3, color=:black, markerstrokewidth=0
        )

        # Biofuel plot
        p_bio = plot(
            xlabel=xlabel,
            ylabel="$(fuel_config.biofuel_ylabel)\n(billion gallons)",
            legend=false,
            grid=true,
            xlims=(x_min, x_max),
            top_margin=-1Plots.mm,
            bottom_margin=45Plots.mm,
            left_margin=12Plots.mm,
            guidefontsize=11
        )

        for (col, label, color) in fuel_config.biofuel_types
            plot!(p_bio, df[!, xcol], df[!, col],
                linewidth=2, marker=:circle, markersize=3, color=color, markerstrokewidth=0
            )
        end

        p_combined = plot(p_main, p_bio,
            layout=grid(2, 1, heights=[0.4, 0.6]),
            link=:x
        )

        push!(plots, p_combined)
    end

    # Legend
    p_legend = plot(
        legend=:top,
        legendcolumns=fuel_config.legendcolumns,
        grid=false, showaxis=false, ticks=false,
        xlims=(0, 1), ylims=(0, 1),
        framestyle=:none,
        legendfontsize=10,
        size=(2200, 80)
    )

    plot!(p_legend, [NaN], [NaN],
        label=fuel_config.main_fuel_label,
        linewidth=2, color=:black, marker=:circle)

    for (col, label, color) in fuel_config.biofuel_types
        plot!(p_legend, [NaN], [NaN], label=label, linewidth=2, color=color, marker=:circle)
    end

    final_plot = plot(
        plot(plots..., layout=(2, 2)),
        p_legend,
        layout=grid(2, 1, heights=[0.95, 0.05]),
        size=(2500, 1800),
        plot_title=fuel_config.plot_title,
        plot_titlefontsize=22,
        plot_titlefontweight=:bold,
        margin=10Plots.mm
    )

    return final_plot
end

# =====================
# Run and Plot the extended grid results
# =====================
println("\nRunning extended policy analysis...")
all_results, all_solutions = run_all_scenarios(params, verbose=true)

results_df = solutions_to_dataframe(all_solutions, params)

println("\nGenerating figures...")

# Aviation Fuel
aviation_config = (
    main_fuel=:q_jet_fuel,
    main_fuel_label="Jet Fuel",
    biofuel_types=[
        (:q_saf_atj_conv, "Conventional ATJ-SAF", :blue),
        (:q_saf_atj_cs, "Climate-Smart ATJ-SAF", :red),
        (:q_saf_hefa_conv, "Conventional HEFA-SAF", :green),
        (:q_saf_hefa_cs, "Climate-Smart HEFA-SAF", :orange),
        (:q_saf_hefa_nonsoy, "Non-soy HEFA-SAF", :purple)
    ],
    biofuel_ylabel="SAF Production",
    plot_title="Aviation Fuel Production by Policy Stringency",
    filename="aviation_fuel_tight_layout.png",
    legendcolumns=6
)

aviation_plot = plot_fuel_production(results_df, aviation_config)
#savefig(aviation_plot, aviation_config.filename)
#display(aviation_plot)

# Road Gasoline
gasoline_config = (
    main_fuel=:q_gasoline,
    main_fuel_label="Gasoline",
    biofuel_types=[
        (:q_ethanol, "Ethanol", :red)
    ],
    biofuel_ylabel="Ethanol Production",
    plot_title="Road Gasoline Fuel Production by Policy Stringency",
    filename="road_gasoline_fuel_tight_layout.png",
    legendcolumns=2
)

gasoline_plot = plot_fuel_production(results_df, gasoline_config)
#savefig(gasoline_plot, gasoline_config.filename)
#display(gasoline_plot)

# Diesel
diesel_config = (
    main_fuel=:q_diesel,
    main_fuel_label="Diesel",
    biofuel_types=[
        (:q_rd_soy, "Soy Renewable Diesel", :red),
        (:q_rd_nonsoy, "Non-soy Renewable Diesel", :orange),
        (:q_biodiesel_soy, "Soy Biodiesel", :blue),
        (:q_biodiesel_nonsoy, "Non-soy Biodiesel", :purple)
    ],
    biofuel_ylabel="Renewable Diesel & Biodiesel",
    plot_title="Diesel Fuel Production by Policy Stringency",
    filename="diesel_fuel_tight_layout.png",
    legendcolumns=5
)

diesel_plot = plot_fuel_production(results_df, diesel_config)
#savefig(diesel_plot, diesel_config.filename)
#display(diesel_plot)

#println("\nFigures saved!")

# =================================================================================
# 6. Target SAF Analysis
# =================================================================================

# =====================
# Find policy stringency for target SAF production
# =====================
function find_policy_for_target_saf(target_saf, params, policy_type; tolerance=0.001)

    SAF_GOODS = [:saf_atj_conv, :saf_atj_cs, :saf_hefa_conv, :saf_hefa_cs, :saf_hefa_nonsoy]

    search_ranges = Dict(
        :carbontax => (0.0, 1500.0, :t),
        :rfs => (0.0, 1.0, :θ_avi),
        :lcfs => (0.0, 1.0, :σ),
        :taxcredit => (0.0, 450.0, :p)
    )

    low, high, param_name = search_ranges[policy_type]

    max_iterations = 100
    iteration = 0

    while iteration < max_iterations && (high - low) > 0.0001
        iteration += 1
        mid = (low + high) / 2.0

        config = if policy_type == :carbontax
            (t=mid, θ_avi=0.0, σ=0.0, p=0.0)
        elseif policy_type == :rfs
            (t=0.0, θ_avi=mid, σ=0.0, p=0.0)
        elseif policy_type == :lcfs
            (t=0.0, θ_avi=0.0, σ=mid, p=0.0)
        else
            (t=0.0, θ_avi=0.0, σ=0.0, p=mid)
        end

        model = build_unified_model(params, config)
        optimize!(model)

        if !is_solved_and_feasible(model)
            println("  Warning: Model not solved at $param_name = $mid")
            high = mid
            continue
        end

        total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)

        println("  Iteration $iteration: $param_name = $(round(mid, digits=3)), SAF = $(round(total_saf, digits=3))")

        if abs(total_saf - target_saf) < tolerance
            println("  ✓ Found solution: $param_name = $(round(mid, digits=3))")
            return (policy_value=mid, model=model, actual_saf=total_saf, config=config)
        end

        if total_saf < target_saf
            low = mid
        else
            high = mid
        end
    end

    println("  ✗ Max iterations reached. Returning best solution found.")
    mid = (low + high) / 2.0
    config = if policy_type == :carbontax
        (t=mid, θ_avi=0.0, σ=0.0, p=0.0)
    elseif policy_type == :rfs
        (t=0.0, θ_avi=mid, σ=0.0, p=0.0)
    elseif policy_type == :lcfs
        (t=0.0, θ_avi=0.0, σ=mid, p=0.0)
    else
        (t=0.0, θ_avi=0.0, σ=0.0, p=mid)
    end

    model = build_unified_model(params, config)
    optimize!(model)
    total_saf = sum(value(model[:q][g]) for g in SAF_GOODS)

    return (policy_value=mid, model=model, actual_saf=total_saf, config=config)
end


# =====================
# TARGET SAF ANALYSIS (3billion gallons)
# =====================

target_saf = 3.0
policy_types = [:carbontax, :rfs, :lcfs, :taxcredit]

println("\n" * "="^80)
println("FINDING POLICY STRINGENCY FOR TARGET SAF = $target_saf billion gallons")
println("="^80)

equivalent_policies = Dict()

for policy_type in policy_types
    println("\n--- Finding $policy_type ---")
    result = find_policy_for_target_saf(target_saf, params, policy_type)
    equivalent_policies[policy_type] = result
end

# Extract solutions
equivalent_solutions = Dict(
    :carbontax => extract_solution(equivalent_policies[:carbontax].model, :carbontax),
    :rfs => extract_solution(equivalent_policies[:rfs].model, :rfs),
    :lcfs => extract_solution(equivalent_policies[:lcfs].model, :lcfs),
    :taxcredit => extract_solution(equivalent_policies[:taxcredit].model, :taxcredit)
)

# Display tables
display_equivalent_policy_results(equivalent_solutions, params, target_saf, equivalent_policies)


# =====================
# Welfare Analysis Functions
# =====================

# Consumer Surplus Change Calculations
clean_small(val, threshold=1e-10) = abs(val) < threshold ? 0.0 : val

"""
Calculate consumer surplus change for a single good
"""
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

# Base analysis에서 status quo 저장
solution_sq = solutions[:statusquo]

# Base analysis
solution_sq = solutions[:statusquo]
cs_changes_base = calculate_cs_changes(solutions, solution_sq, params; scenarios=SCENARIOS)
display_cs_changes(cs_changes_base; scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="CONSUMER SURPLUS CHANGES FOR BASE POLICIES (billion \$)")

# Equivalent policies
cs_changes_equiv = calculate_cs_changes(equivalent_solutions, solution_sq, params)
display_cs_changes(cs_changes_equiv; scenarios=[:carbontax, :rfs, :lcfs, :taxcredit],
    title="CONSUMER SURPLUS CHANGES FOR EQUIVALENT POLICIES (billion \$)")

# Extended grid
#cs_changes_grid = calculate_cs_changes(all_solutions, solution_sq, params)




# Producer Surplus Change Calculations
"""
Calculate land producer surplus change
"""
function calc_land_ps_change(L_policy, r_policy, L_sq, r_sq, land_supply)
    r0_land = land_supply.r0_land
    ϵ_land = land_supply.ϵ_land
    L0 = land_supply.L0

    # Revenue change
    revenue_change = r_policy * L_policy - r_sq * L_sq

    # Cost change (Result term)
    exponent = (1 + ϵ_land) / ϵ_land
    cost_change = (r0_land * ϵ_land) / (L0^(1 / ϵ_land) * (1 + ϵ_land)) *
                  (L_policy^exponent - L_sq^exponent)

    return revenue_change - cost_change
end

"""
Calculate fossil fuel producer surplus change for a single fuel
"""
function calc_fossil_ps_change(Q_policy, P_policy, Q_sq, P_sq, fuel_params)
    c0 = fuel_params.c0
    c2 = fuel_params.c2
    v = fuel_params.v

    # PS_policy = P*Q - ∫[0 to Q] MC dq
    #           = P*Q - [c0*Q + c2*max(0,Q-v)^3/3]

    # Policy scenario PS
    if Q_policy > v
        cost_policy = c0 * Q_policy + c2 * (Q_policy - v)^3 / 3
    else
        cost_policy = c0 * Q_policy
    end
    ps_policy = P_policy * Q_policy - cost_policy

    # Status quo PS
    if Q_sq > v
        cost_sq = c0 * Q_sq + c2 * (Q_sq - v)^3 / 3
    else
        cost_sq = c0 * Q_sq
    end
    ps_sq = P_sq * Q_sq - cost_sq

    return ps_policy - ps_sq
end

"""
Calculate producer surplus changes for all scenarios
"""
function calculate_ps_changes(solutions, solution_sq, params; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(solutions)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    land_supply = params.supply.land

    # Status quo values
    L_sq = solution_sq.l_n + solution_sq.l_cs
    r_sq = solution_sq.duals.r_land

    ps_changes_all = Dict()

    for scenario in scenario_list
        solution_policy = solutions[scenario]

        # Policy values
        L_policy = solution_policy.l_n + solution_policy.l_cs
        r_policy = solution_policy.duals.r_land

        # Land PS change
        ps_land = calc_land_ps_change(L_policy, r_policy, L_sq, r_sq, land_supply)

        # Fossil fuels: Perfect competition → P = MC → PS = 0 always
        ps_jet = 0.0
        ps_gasoline = 0.0
        ps_diesel = 0.0

        ps_fossil_total = 0.0
        ps_total = ps_land  # Only land has PS

        ps_changes_all[scenario] = Dict(
            :land => ps_land,
            :jet_fuel => ps_jet,
            :gasoline => ps_gasoline,
            :diesel => ps_diesel,
            :fossil_total => ps_fossil_total,
            :total => ps_total
        )
    end

    return ps_changes_all
end
"""
Create producer surplus change table
"""
function make_ps_change_table(ps_changes_all; scenarios=nothing)
    scenario_list = isnothing(scenarios) ? collect(keys(ps_changes_all)) : scenarios
    scenario_list = filter(s -> s != :statusquo, scenario_list)

    df = DataFrame(Producer=String[])
    for scenario in scenario_list
        df[!, String(scenario)] = Float64[]
    end

    producers = [
        (:land, "Land Owners"),
        (:jet_fuel, "Jet Fuel Producers"),
        (:gasoline, "Gasoline Producers"),
        (:diesel, "Diesel Producers"),
        (:fossil_total, "Fossil Fuel Total"),
        (:total, "TOTAL")
    ]

    for (prod_key, prod_label) in producers
        push!(df.Producer, prod_label)
        for scenario in scenario_list
            value = clean_small(ps_changes_all[scenario][prod_key])
            push!(df[!, String(scenario)], value)
        end
    end

    return df
end

"""
Display producer surplus changes
"""
function display_ps_changes(ps_changes_all; scenarios=nothing, title="PRODUCER SURPLUS CHANGES (billion \$)")
    println("\n" * "="^130)
    println(title)
    println("="^130)

    ps_table = make_ps_change_table(ps_changes_all; scenarios=scenarios)
    show(ps_table, allrows=true)

    println("\n" * "="^130)
end

# Base analysis
solution_sq = solutions[:statusquo]

# Calculate PS changes
ps_changes_base = calculate_ps_changes(solutions, solution_sq, params; scenarios=SCENARIOS)

# Display
display_ps_changes(ps_changes_base; scenarios=[:carbontax, :rfs, :lcfs, :taxcredit])

ps_changes_equiv = calculate_ps_changes(equivalent_solutions, solution_sq, params)
display_ps_changes(ps_changes_equiv; scenarios=[:carbontax, :rfs, :lcfs, :taxcredit])

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