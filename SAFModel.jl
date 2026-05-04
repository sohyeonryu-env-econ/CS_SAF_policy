module SAFModel

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

# delta: Carbon Intensity (ton CO2e per gallon) all fuel and food goods
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
meal_per_oil = 2.22

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
    :soymeal => create_demand_params(-0.941, 423.41, 62.27, 600.0)
)

# Fuel supply functions: fuel hockey stick = c0 + c1*q + c2*(x-v)^2
# Feedstock to SAF production constant


#LOW
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
    24050.0,  # 1. jet_fuel
    6.9,      # 2. saf_atj (shared)
    15.0,     # 3. saf_hefa (shared)
    42557.0,  # 4. gasoline
    18.01,    # 5. ethanol
    64.68,    # 6. diesel
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

fuel_cost = Dict()
for (key, c0, c1, c2, v) in zip(fuel_goods_cost_map, c0_vec, c1_vec, c2_vec, v_vec)
    fuel_cost[key] = (c0=c0, c1=c1, c2=c2, v=v)
end


# land supply functions: land = L0*(r/r0)^ϵ
land_supply = (
    L0=0.11768, # baseline land use
    r0_land=587.71, # baseline land rent
    ϵ_land=0.1  # land supply elasticity
)

supply = (
    fuel=fuel_cost,
    #feedstock=feedstock_cost,
    land=land_supply
)

# land corn ratio
ω = 0.54
σ_cet = 0.0 # elasticity of transformation between corn and soybeans

# κ: fixed costs of climate-smart practice adoption
κ = 19.0  # $ per acre

# Non-soy feedstock price ($/lb)
const nonsoy_feedstock_price = 0.49

# HEFA SAF additional processing cost compared to RD ($/gal)
const hefa_saf_premium = 0.064

coeff = (
    delta=δ,
    gamma=γ,
    r=r,
    beta=β,
    alpha=α,
    theta=θ,
    omega=ω,
    σ_cet=σ_cet,
    kappa=κ,
    soybean_to_oil=soybean_to_oil,
    soybean_to_meal=soybean_to_meal,
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

function build_unified_model(params, config; warm_start=nothing)

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
    σ_cet = coeff.σ_cet
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
    ALL_GOODS = union(AVIATION_FUELS, [:gasoline, :ethanol, :diesel, :biodiesel_soy, :biodiesel_nonsoy, :rd_soy, :rd_nonsoy, :corn, :soyoil])

    # =====================
    # Variables
    # =====================

    @variables model begin
        x[s in SECTORS] >= 0.1 # demand quantities
        q[g in FUEL_GOODS] >= 0 # supply quantities
        p_f[f in FEEDSTOCK_GOODS] >= 0 # feedstock price
        p_c[s in [:avi, :gas, :die, :soymeal]] >= 0 # consumer's price on x
        l_n >= 0      # conventional farmer 총 토지
        l_cs >= 0     # climate-smart farmer 총 토지
        r_corn >= 500   # corn 토지 임대료
        r_soy >= 500    # soy 토지 임대료

    end

    # Base dual variables 
    @variable(model, λ_rfs >= 0)
    @variable(model, λ_blendwall_ethanol >= 0)
    @variable(model, λ_blendwall_biodiesel >= 0)
    @variable(model, λ_nonsoy_capacity >= 0)

    # Policy-specific dual variables 
    @variable(model, λ_rfs_avi >= 0)      # RFS aviation
    @variable(model, λ_lcfs >= 0)         # LCFS
    #@variable(model, r_land )    # Land rent

    # Variable naming
    for g in FUEL_GOODS
        set_name(q[g], meta[:process_labels][g])
    end

    # I need initial values for the non linear solver to converge. (CET)
    #if !isnothing(warm_start)
    #    set_start_value(l_n, warm_start.l_n)
    #    set_start_value(l_cs, warm_start.l_cs)
    #    set_start_value(r_corn, warm_start.duals.r_corn)
    #    set_start_value(r_soy, warm_start.duals.r_soy)
    #    for g in FUEL_GOODS
    #        set_start_value(q[g], warm_start.q[g])
    #    end
    #    for s in SECTORS
    #        set_start_value(x[s], warm_start.x[s])
    #    end
    #    for f in FEEDSTOCK_GOODS
    #        set_start_value(p_f[f], warm_start.p_f[f])
    #    end
    #    for s in [:avi, :gas, :die, :soymeal]
    #        set_start_value(p_c[s], warm_start.p_c[s])
    #    end
    #end

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
    # Landowner
    # =====================
    # CET price index (r_land)
    @expression(model, r_land,
        (omega * r_corn^(1 + σ_cet) + (1 - omega) * r_soy^(1 + σ_cet))^(1 / (1 + σ_cet))
    )

    # Total land supply: responds to r_land
    @expression(model, L_total, L0 * (r_land / r0_land)^ϵ_land)

    # CET land supply by crop
    # σ_cet = 0 이면 l_corn_supply = L_total * omega, l_soy_supply = L_total * (1-omega) 로 고정
    @expression(model, l_corn_supply, L_total * omega * (r_corn / r_land)^σ_cet)
    @expression(model, l_soy_supply, L_total * (1 - omega) * (r_soy / r_land)^σ_cet)


    # =====================
    # Upstream farmers
    # =====================

    # 같은 농부가 corn/soy를 land owner CET 배분 비율로 동시에 경작
    @expression(model, l_corn_n, l_n * (l_corn_supply / L_total))
    @expression(model, l_corn_cs, l_cs * (l_corn_supply / L_total))
    @expression(model, l_soy_n, l_n * (l_soy_supply / L_total))
    @expression(model, l_soy_cs, l_cs * (l_soy_supply / L_total))


    # feedstock production quantity
    @expression(model, q_corn_n, gamma[:feedstock_corn_n] * l_corn_n)
    @expression(model, q_corn_cs, gamma[:feedstock_corn_cs] * l_corn_cs)
    @expression(model, q_soy_n, gamma[:feedstock_soy_n] * l_soy_n * soybean_to_oil)
    @expression(model, q_soy_cs, gamma[:feedstock_soy_cs] * l_soy_cs * soybean_to_oil)

    # feedstock production (soybeans to soybean oil conversion =10.71lb/bushel)
    #@expression(model, q_corn_n, omega * gamma[:feedstock_corn_n] * l_n)
    #@expression(model, q_corn_cs, omega * gamma[:feedstock_corn_cs] * l_cs)
    #@expression(model, q_soy_n, (1 - omega) * gamma[:feedstock_soy_n] * l_n * soybean_to_oil)
    # oil and meal are jointly produced, so soybeans quantity = soybean oil * soybean_to_oil = soybean meal * soybean_to_meal. gamma here is for soybeans before crushing.
    #@expression(model, q_soy_cs, (1 - omega) * gamma[:feedstock_soy_cs] * l_cs * soybean_to_oil)

    # Marginal revenue: Upstream farmers
    #@expression(model, marginal_revenue_n,
    #    omega * gamma[:feedstock_corn_n] * p_f[:feedstock_corn_n] +
    #    (1 - omega) * gamma[:feedstock_soy_n] * (p_f[:feedstock_soy_n] * soybean_to_oil + p_c[:soymeal] * soybean_to_meal)
    #)

    #@expression(model, marginal_revenue_cs,
    #    omega * gamma[:feedstock_corn_cs] * p_f[:feedstock_corn_cs] +
    #    (1 - omega) * gamma[:feedstock_soy_cs] * (p_f[:feedstock_soy_cs] * soybean_to_oil + p_c[:soymeal] * soybean_to_meal)
    #)

    # Marginal revenue: Upstream farmers
    @expression(model, marginal_revenue_n,
        (l_corn_supply / L_total) * gamma[:feedstock_corn_n] * p_f[:feedstock_corn_n] +
        (l_soy_supply / L_total) * gamma[:feedstock_soy_n] * (p_f[:feedstock_soy_n] * soybean_to_oil +
                                                              p_c[:soymeal] * soybean_to_meal)
    )

    @expression(model, marginal_revenue_cs,
        (l_corn_supply / L_total) * gamma[:feedstock_corn_cs] * p_f[:feedstock_corn_cs] +
        (l_soy_supply / L_total) * gamma[:feedstock_soy_cs] * (p_f[:feedstock_soy_cs] * soybean_to_oil +
                                                               p_c[:soymeal] * soybean_to_meal)
    )

    # zero profit condition
    # conventional farmer
    @constraint(model, r_land - marginal_revenue_n ⟂ l_n)
    @constraint(model, r_land + kappa - marginal_revenue_cs ⟂ l_cs)

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

    @expression(model, process_mc_biodiesel,
        fuel_cost[:biodiesel_shared].c0 +
        fuel_cost[:biodiesel_shared].c1 * (q[:biodiesel_soy] + q[:biodiesel_nonsoy]) +
        fuel_cost[:biodiesel_shared].c2 * max(0, (q[:biodiesel_soy] + q[:biodiesel_nonsoy]) - fuel_cost[:biodiesel_shared].v)^2
    )

    # Marginal costs for other fuels
    @expression(model, marginal_costs_fuel[g in (:jet_fuel, :gasoline, :diesel, :ethanol)],
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
        # config.t * (g in AVIATION_FUELS ? delta[g] : 0.0) +
        config.t * (
            config.carbon_tax_scope == :aviation ? (g in AVIATION_FUELS ? delta[g] : 0.0) :
            config.carbon_tax_scope == :all ? (g in ALL_GOODS ? delta[g] : 0.0) : 0.0
        ) +

        # 2. RFS aviation mandate
        λ_rfs_avi * (
            (g == :jet_fuel) ? config.θ_avi :
            #(g in SAF_GOODS) ? -1.6 : 0.0 # without 50% CI threshold
            (g in SAF_GOODS && delta[g] <= 0.5 * delta[:jet_fuel]) ? -1.6 : 0.0 # 50% CI threshold
        ) +

        # 3. LCFS
        λ_lcfs * (g in AVIATION_FUELS ?
                  -((1 - config.σ) * delta[:jet_fuel] - delta[g]) : 0.0
        ) +

        #. 4. Tax credit for SAF
        -(g in SAF_GOODS && haskey(delta_mj, g) ? tax_credit_rate(delta_mj[g], baselineCI, config.p) : 0.0)
        # If the tax credit is applied to all biofuels
        #-(g in union(SAF_GOODS, BIODIESEL_GOODS, RD_GOODS, [:ethanol]) && haskey(delta_mj, g) ?
        #  tax_credit_rate(delta_mj[g], baselineCI, config.p) : 0.0)
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
        process_mc_hefa +
        alpha[:saf_hefa_conv] * p_f[:feedstock_soy_n] +
        policy_adjustment[:saf_hefa_conv] -
        price_per_unit[:saf_hefa_conv]
        ⟂
        q[:saf_hefa_conv]
    )

    # Climate-smart HEFA SAF producers
    @constraint(model,
        process_mc_hefa +
        alpha[:saf_hefa_cs] * p_f[:feedstock_soy_cs] +
        policy_adjustment[:saf_hefa_cs] -
        price_per_unit[:saf_hefa_cs]
        ⟂
        q[:saf_hefa_cs]
    )

    # Non-soy HEFA SAF producers
    @constraint(model,
        process_mc_hefa +
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
        process_mc_biodiesel +
        alpha[:biodiesel_soy] * p_f[:feedstock_soy_n] +
        policy_adjustment[:biodiesel_soy] -
        price_per_unit[:biodiesel_soy]
        ⟂
        q[:biodiesel_soy]
    )

    # Non Soy biodiesel producers
    @constraint(model,
        process_mc_biodiesel +
        alpha[:biodiesel_nonsoy] * nonsoy_feedstock_price +
        policy_adjustment[:biodiesel_nonsoy] -
        price_per_unit[:biodiesel_nonsoy]
        ⟂
        q[:biodiesel_nonsoy]
    )
    # Soy renewable diesel producers
    @constraint(model,
        process_mc_hefa - hefa_saf_premium +
        alpha[:rd_soy] * p_f[:feedstock_soy_n] +
        policy_adjustment[:rd_soy] -
        price_per_unit[:rd_soy]
        ⟂
        q[:rd_soy]
    )

    # Non Soy Renewable diesel producers
    @constraint(model,
        process_mc_hefa - hefa_saf_premium +
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
    @constraint(model, l_corn_supply - (l_corn_n + l_corn_cs) ⟂ r_corn)
    @constraint(model, l_soy_supply - (l_soy_n + l_soy_cs) ⟂ r_soy)

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
        #1.6 * sum(q[g] for g in SAF_GOODS) - config.θ_avi * q[:jet_fuel] # without 50% CI threshold
        1.6 * sum(q[g] for g in SAF_GOODS if delta[g] <= 0.5 * delta[:jet_fuel]) - config.θ_avi * q[:jet_fuel] # 50% CI threshold
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

function run_scenario(scenario::Symbol, params, policy_configs; warm_start=nothing)
    config = getproperty(policy_configs, scenario)
    model = build_unified_model(params, config; warm_start=warm_start)
    optimize!(model)
    status = termination_status(model)
    println("  Status: $status")
    if is_solved_and_feasible(model)
        println("✓ $(scenario) solved")
        return model
    else
        println("✗ $(scenario) failed: $status")
        return nothing
    end
end

function extract_solution(model, scenario)
    if isnothing(model)
        return nothing
    end

    q = value.(model[:q])
    x = value.(model[:x])
    p_c = value.(model[:p_c])
    p_f = value.(model[:p_f])

    l_n = value(model[:l_n])
    l_cs = value(model[:l_cs])
    r_corn = value(model[:r_corn])
    r_soy = value(model[:r_soy])
    r_land = value(model[:r_land])

    # Derived land use
    l_corn_n = value(model[:l_corn_n])
    l_corn_cs = value(model[:l_corn_cs])
    l_soy_n = value(model[:l_soy_n])
    l_soy_cs = value(model[:l_soy_cs])

    q_corn_n = value(model[:q_corn_n])
    q_corn_cs = value(model[:q_corn_cs])
    q_soy_n = value(model[:q_soy_n])
    q_soy_cs = value(model[:q_soy_cs])

    ddgs = 0.092 * q[:ethanol] + 0.159 * (q[:saf_atj_conv] + q[:saf_atj_cs])
    soymeal_produ = value(model[:total_soymeal_supply])

    duals = (
        λ_rfs=value(model[:λ_rfs]),
        λ_rfs_avi=value(model[:λ_rfs_avi]),
        λ_lcfs=value(model[:λ_lcfs]),
        r_land=r_land,
        r_corn=r_corn,
        r_soy=r_soy,
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
        l_corn_n=l_corn_n,
        l_corn_cs=l_corn_cs,
        l_soy_n=l_soy_n,
        l_soy_cs=l_soy_cs,
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

function extract_all_solutions(results)
    solutions = Dict()
    for (scenario_name, model) in results
        solutions[scenario_name] = extract_solution(model, scenario_name)
    end
    return solutions
end

export params, tax_credit_rate, build_unified_model, run_scenario, extract_solution, extract_all_solutions
export FUEL_GOODS, FEEDSTOCK_GOODS, FOOD_GOODS
export is_solved_and_feasible

end # module SAFModel