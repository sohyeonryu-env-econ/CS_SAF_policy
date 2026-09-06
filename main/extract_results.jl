# extract_results.jl
cd(@__DIR__)
println("Working directory: ", pwd())

using JLD2
using DataFrames
using CSV
using JuMP

# The validation table (at the bottom) reads the conversion factors r, beta and the land
# share omega straight from params. Copying them as constants means a change in ModelMkt
# would not reach the table.
include(joinpath(@__DIR__, "scenarios.jl"))
using .Scenarios
include(joinpath(@__DIR__, "model_mkt.jl"))
import .ModelMkt: params

# Output is reported in metric; the model stays in US units. See main/units.jl for why,
# and flip Units.METRIC there to reproduce the old US-unit CSV byte for byte.
include(joinpath(@__DIR__, "units.jl"))
using .Units

include(joinpath(@__DIR__, "paths.jl"))
using .Paths
Paths.setup()

const DATA_DIR   = Paths.DATA_DIR
const TABLE_DIR  = Paths.TABLE_DIR
const FIGURE_DIR = Paths.FIGURE_DIR

# =================================================================================
# 1. Load results
# =================================================================================

# RESULT_SET decides which solver output becomes the tables.
#   :unified  -> unified_benchmark.jl. Every scenario is tuned to one common GHG
#                abatement (the abatement of an RFS with no CS crediting and no CI
#                threshold producing 3B gal of SAF), so cases are comparable.
#   :fourcase -> the older 4-case tables, where each case had its own target. The runner
#                that produced them is gone, so this only works off an old saved jld2.
const RESULT_SET = :unified

const INPUT_FILE, OUTPUT_CSV = RESULT_SET == :unified ?
                               ("results_unified_benchmark.jld2", "results_unified_benchmark.csv") :
                               ("results_all_cases.jld2", "results_comprehensive.csv")

# all_case_results: per case, the solutions (results), welfare, target_emissions and target_reduction
# base_results: the statusquo and firstbest solutions (emissions already included)
@load joinpath(DATA_DIR, INPUT_FILE) all_case_results base_results base_configs

println("Result set: $(RESULT_SET)  (input: $(INPUT_FILE))")

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

# CANONICAL CASE NUMBERING, must match unified_benchmark.jl:
#   case1 = no CS, no 50% CI threshold
#   case2 = no CS, 50% CI threshold
#   case3 = CS,    no 50% CI threshold
#   case4 = CS,    50% CI threshold

# SCEN_TAGGED: the reported scenario set, (a) through (i).
#
# This is the SAME list 3B_outcomes.jl publishes, deliberately. A letter has to mean the
# same thing in the CSV and in the paper tables, and the only way to guarantee that is to
# write it once. Columns are ordered by letter, not by case.
#
# Each entry carries the two policy-design flags explicitly rather than hiding them behind
# a case number, so the CSV states what each scenario is without a lookup table:
#   cs   = does the regulator credit climate-smart practice?
#   thr  = is the 50% CI-reduction eligibility threshold applied?
#
# Canonical case numbering (see below): case1 = no CS/no thr, case2 = no CS/thr,
# case3 = CS/no thr, case4 = CS/thr.
#
# The first-best aviation carbon tax (:firstbest) is NOT reported. It is a welfare
# benchmark, not one of the instruments being compared, and mixing it into the same table
# invited reading it as a tenth policy scenario.
# SCEN from scenarios.jl is used as is. This file reads name / cs / thr; the figure
# fields label / short are ignored.
SCEN_TAGGED = copy(Scenarios.SCEN)

# Drop anything the solver did not produce, rather than erroring deep inside a table.
filter!(sc -> sc.case === :base ||
              (haskey(all_case_results, sc.case) &&
               haskey(all_case_results[sc.case].results, sc.policy)), SCEN_TAGGED)

cs_text(sc) = isnothing(sc.cs) ? "" : (sc.cs ? "CS" : "no CS")
thr_text(sc) = isnothing(sc.thr) ? "" : (sc.thr ? "CI threshold" : "no CI threshold")

# Column header: letter + policy, with just enough of the design flags to stay unique
# ((c), (d) and (e) are all the RFS). The full description goes in the Scenario rows the
# table opens with, so the header can stay short.
function header_of(sc)
    isnothing(sc.cs) && return "$(sc.tag) $(sc.name)"
    suffix = sc.cs ? (sc.thr ? "CS + CI thr" : "CS") : "no CS"
    return "$(sc.tag) $(sc.name) $(suffix)"
end

policy_order = [(sc.policy, sc.case) for sc in SCEN_TAGGED if sc.case !== :base]

# The BENCHMARK scenario (unified only), flagged separately in the tables. = (c)
const BENCH_KEY = RESULT_SET == :unified ? (:rfs, :case1) : nothing

scenario_order = [
    (:observed, :observed),
    (:validation, :validation),
    [(sc.policy, sc.case) for sc in SCEN_TAGGED]...,
]

scenario_labels = Dict{Any,String}(
    (:observed, :observed) => "Observed",
    (:validation, :validation) => "Validation",
)
for sc in SCEN_TAGGED
    scenario_labels[(sc.policy, sc.case)] = header_of(sc)
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
# 7. Scenario description and policy stringency
# =================================================================================

# The column header carries only the letter and the policy, so these four rows say in
# full what each scenario is. Reading a column then needs no external key.
scen_rows = [
    ("Label",          sc -> sc.tag),
    ("Policy",         sc -> sc.name),
    ("Climate-smart",  cs_text),
    ("CI threshold",   thr_text),
]

for (row_name, f) in scen_rows
    d = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation)
            d[(scenario, group)] = ""
        else
            sc = SCEN_TAGGED[findfirst(x -> x.policy === scenario && x.case === group, SCEN_TAGGED)]
            d[(scenario, group)] = f(sc)
        end
    end
    add_row!(df, "Scenario", row_name, "-", d)
end

stringency_dict = Dict()
for (scenario, group) in scenario_order
    if group in (:observed, :validation)
        stringency_dict[(scenario, group)] = missing
    elseif group == :base
        stringency_dict[(scenario, group)] = 0.0     # status quo; firstbest is not reported
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

# Abatement actually delivered and the gap to target. A one-line check that the
# equivalence held. statusquo/firstbest live in base_results, policy scenarios in all_case_results.
const SQ_TOTAL_EMISSIONS = base_results[:statusquo].emissions.total

achieved_dict = Dict()
gap_dict = Dict()
bench_dict = Dict()
for (scenario, group) in scenario_order
    key = (scenario, group)
    if group in (:observed, :validation)
        achieved_dict[key] = missing
        gap_dict[key] = missing
        bench_dict[key] = missing
    elseif group == :base
        achieved_dict[key] = SQ_TOTAL_EMISSIONS - base_results[scenario].emissions.total
        gap_dict[key] = missing
        bench_dict[key] = 0
    else
        achieved = SQ_TOTAL_EMISSIONS - all_case_results[group].results[scenario].emissions.total
        achieved_dict[key] = achieved
        gap_dict[key] = achieved - all_case_results[group].target_reduction
        bench_dict[key] = key == BENCH_KEY ? 1 : 0
    end
end
add_row!(df, "Target", "Achieved_Reduction_BtonCO2e", "B ton CO2e", achieved_dict)
add_row!(df, "Target", "Reduction_Gap_BtonCO2e", "B ton CO2e", gap_dict)
add_row!(df, "Target", "Is_Benchmark", "0/1", bench_dict)

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

# Travel demand is a SERVICE quantity, not a fuel volume: the market-clearing conditions in
# model_mkt.jl read r[:jet_fuel]*(q_jet + beta*sum SAF) - x[:avi], with r in miles per gallon.
# So x[:avi] is billion revenue passenger-miles and x[:gas] / x[:die] are billion vehicle-miles.
# These rows used to be labelled "B gal", which is off by the fuel economy factor r.
demand_sectors = [
    (:avi, "B RPM"),
    (:gas, "B VMT"),
    (:die, "B VMT"),
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
    (:avi, "\$/RPM"),
    (:gas, "\$/VMT"),
    (:die, "\$/VMT"),
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

# Every multiplier inherits the unit of ITS OWN constraint, so they do not share a denominator:
# the RFS and blend-wall constraints are written in B gal, land in B acre, the non-soy capacity
# in B lb, and the LCFS in B ton CO2e. They used to all be labelled a bare "\$".
# Row names follow the display names, not the solution field names: lambda_rfs is the
# ROAD RFS D6 (always active background, theta = 0.125), lambda_rfs_avi is the aviation
# instrument now called the volumetric mandate. Naming them apart stops the two from
# being read as the same policy at two stringencies.
dual_vars = [
    (:λ_rfs, "lambda_rfs_d6_road", "\$/gal"),
    (:λ_rfs_avi, "lambda_vol_mandate", "\$/gal"),
    (:λ_lcfs, "lambda_ci_standard", "\$/ton CO2e"),
    (:r_land, "r_land", "\$/acre"),
    (:λ_blendwall_ethanol, "lambda_blendwall_ethanol", "\$/gal"),
    (:λ_blendwall_biodiesel, "lambda_blendwall_biodiesel", "\$/gal"),
    (:λ_nonsoy_capacity, "lambda_nonsoy_capacity", "\$/lb"),
]

for (dv, dv_label, dv_unit) in dual_vars
    dv_dict = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation)
            dv_dict[(scenario, group)] = missing
        else
            sol = get_solution(scenario, group)
            dv_dict[(scenario, group)] = safe_get_value(sol.duals, dv)
        end
    end
    add_row!(df, "Shadow prices", dv_label, dv_unit, dv_dict)
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
add_row!(df, "Welfare", "PS_change_land", "B\$", ps_dict)

# PS non-soy feedstock (scarcity rent = λ_nonsoy_capacity * feedstock used)
ps_ns_dict = Dict()
for (scenario, group) in scenario_order
    if group in (:observed, :validation, :base)
        ps_ns_dict[(scenario, group)] = missing
    else
        ps_ns_dict[(scenario, group)] = all_case_results[group].welfare.ps_nonsoy[scenario].ps_change
    end
end
add_row!(df, "Welfare", "PS_change_nonsoy", "B\$", ps_ns_dict)

# --- Fossil refining rent -----------------------------------------------------
# Jet fuel, gasoline and diesel now face constant-elasticity supply, so displacing them
# destroys refiner rent.  Identically zero under the perfectly elastic benchmark.
ps_fo_dict = Dict()
for (scenario, group) in scenario_order
    if group in (:observed, :validation, :base)
        ps_fo_dict[(scenario, group)] = missing
    else
        ps_fo_dict[(scenario, group)] = all_case_results[group].welfare.ps_fossil[scenario].ps_change
    end
end
add_row!(df, "Welfare", "PS_change_fossil", "B\$", ps_fo_dict)

for g in (:jet_fuel, :gasoline, :diesel)
    d = Dict()
    for (scenario, group) in scenario_order
        d[(scenario, group)] = group in (:observed, :validation, :base) ? missing :
            all_case_results[group].welfare.ps_fossil[scenario].by_fuel[g]
    end
    add_row!(df, "Welfare", "PS_change_fossil_$(g)", "B\$", d)
end

# PS total (land + non-soy + fossil), this is what enters private surplus
ps_tot_dict = Dict()
for (scenario, group) in scenario_order
    if group in (:observed, :validation, :base)
        ps_tot_dict[(scenario, group)] = missing
    else
        ps_tot_dict[(scenario, group)] =
            all_case_results[group].welfare.welfare_summary[scenario].ps_total_change
    end
end
add_row!(df, "Welfare", "PS_change", "B\$", ps_tot_dict)

# --- Non-soy feedstock diagnostics -------------------------------------------
# PS_change_nonsoy is only non-zero when the capacity constraint binds. These rows
# make that visible: if feedstock_use hits capacity (25 B lb, ModelMkt.nonsoy_capacity)
# then λ > 0 and the
# effective price paid by refiners is c + λ rather than the exogenous c.
# All pulled from the ps_nonsoy NamedTuple, so no model rebuild is needed here.
nonsoy_diagnostics = [
    (:use_policy, "feedstock_use", "B lb"),
    (:price_policy, "effective_price_c_plus_lambda", "\$/lb"),
    (:rent_policy, "rent_level", "B\$"),
    (:capacity, "capacity", "B lb"),
]

for (field, label, unit) in nonsoy_diagnostics
    ns_dict = Dict()
    for (scenario, group) in scenario_order
        if group in (:observed, :validation, :base)
            ns_dict[(scenario, group)] = missing
        else
            ns_dict[(scenario, group)] =
                getproperty(all_case_results[group].welfare.ps_nonsoy[scenario], field)
        end
    end
    add_row!(df, "Non-soy feedstock", label, unit, ns_dict)
end

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
# cs_change / ps_total_change / gr_change / env_benefit are already written above under their
# own names (CS_change_total, PS_change, GovRevenue, EnvBenefit_total_benefit), so only the two
# aggregates go here.
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
# 18. Unit conversion, Rounding and Save
# =================================================================================

# US customary -> metric, driven by the per-row `Unit` column. Every numeric cell is
# converted and the label rewritten in one pass; an unregistered unit string raises
# rather than passing through unconverted. Must run AFTER every add_row! and BEFORE
# rounding, so the rounding rules below apply to the reported magnitudes.
Units.convert_df!(df)
println("Units: converted to metric (Units.METRIC = $(Units.METRIC))")

function round_value_custom(x, category)
    ismissing(x) && return missing
    !(x isa Number) && return x
    if category in ("Price_consumer", "Price")
        # Significant digits, not decimal places. In metric the consumer prices are
        # small ($/passenger-km ~ 0.0247, $/L ~ 0.618), and a fixed 3 decimals would leave two
        # significant figures or fewer -- the precision loss already flagged in the
        # validation-table note below.
        return round(x, sigdigits=6)
    elseif category in ("Emissions", "Welfare", "Non-soy feedstock")
        return round(x, digits=5)
    elseif category == "Target"
        # Reduction_Gap sits at about 1e-6, so keep more digits
        return round(x, digits=8)
    elseif category == "Policy"
        # theta_avi and sigma lose the differences between cases if cut to 2 decimals
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

output_file = joinpath(TABLE_DIR, OUTPUT_CSV)
CSV.write(output_file, df)

println("\n" * "="^80)
println("✓ CSV saved: $output_file")
println("  Rows: $(nrow(df)), Columns: $(ncol(df))")
println("="^80)
println("\nFirst 20 rows:")
show(first(df, 20), allrows=true, allcols=false)


# make_validation_table.jl
# Builds the model validation LaTeX table by reading observed_data and
# base_results[:statusquo] directly. The same table comes out regardless of RESULT_SET.
#
# Why not go through the CSV: Price_consumer there is rounded to 3 decimals, and
# p_c[:avi] is about 0.0397, leaving only two significant digits. Multiplied by r = 58.96
# the jet price prints as 2.36 instead of 2.34, and observed (0.0396) prints as the same
# 2.36, making the difference read as 0.0%. Use the unrounded solution instead.
#
# Mapping:
#   Consumption          -> solution.q (fuels), observed_data["Production"]
#   Corn/Soyoil for food -> solution.x, observed_data["Demand"]
#   Fuel prices ($/gal)  -> p_c ($/mile) * r  (ethanol converts once more through beta)
#   Corn/Soy prices      -> solution.p_f, observed_data["Price"]
#   Land                 -> model cropland splits between corn and soy at a fixed omega
#                           (ModelMkt: q_corn = omega*gamma*l, q_soy = (1-omega)*gamma*l)

using Printf

const TEX_OUT = joinpath(TABLE_DIR, "validation_table.tex")

const SQ_SOL = base_results[:statusquo]
const R_MILE = params.coeff.r            # gallon -> mile
const BETA_E = params.coeff.beta         # relative energy content across fuels
const OMEGA = params.coeff.omega         # corn share of cropland (the rest is soy)

# Observed cropland: the USDA figure net of export acreage (footnote 1). This is exogenous
# data rather than model output, so it stays a constant. The StatusQuo side is computed.
const LAND_CORN_OBS = 70.15
const LAND_SOY_OBS = 50.15

num(v) = (ismissing(v) || !(v isa Number)) ? NaN : float(v)

obs(category, variable) = num(get(observed_data, (category, string(variable)), missing))

sq_q(g) = num(safe_get_value(SQ_SOL.q, g))
sq_x(g) = num(safe_get_value(SQ_SOL.x, g))
sq_pf(g) = num(safe_get_value(SQ_SOL.p_f, g))
sq_pc(g) = num(safe_get_value(SQ_SOL.p_c, g))

# $/mile -> $/gallon
jet_pg(p) = R_MILE[:jet_fuel] * p
gas_pg(p) = R_MILE[:gasoline] * p
eth_pg(p) = R_MILE[:gasoline] * BETA_E[(:ethanol, :gasoline)] * p
die_pg(p) = R_MILE[:diesel] * p

sq_land_total = (SQ_SOL.l_n + SQ_SOL.l_cs) * 1000   # B acre -> M acre
sq_land_corn = OMEGA * sq_land_total
sq_land_soy = (1 - OMEGA) * sq_land_total

# This table is built from the raw solution, NOT from `df`, so Units.convert_df! above
# does not touch it. Each value converts at the point it is read.
#
#   fuel volume   B gal          -> B liters    prices  $/gal    -> $/L
#   corn          B bushel       -> MMT                 $/bushel -> $/tonne
#   soy oil       B lb           -> MMT                 $/lb     -> $/tonne
#   land          M acres        -> M ha
#
# $/L runs around 0.6 where $/gal ran around 2.3, so the fuel price block takes an extra
# decimal in metric; everything else keeps two.
const MU = Units.METRIC
qgal(x) = Units.gal_to_L(x)
qbu(x) = Units.bu_to_Mt(x)
qlb(x) = Units.lb_to_Mt(x)
qac(x) = Units.acre_to_ha(x)
ppg(x) = Units.price_gal_to_L(x)
ppb(x) = Units.price_bu_to_t(x)
ppl(x) = Units.price_lb_to_t(x)

const H_VOL = MU ? "Consumption (billion liters)" : "Consumption (billion gallons)"
const H_CORN_Q = MU ? "Corn for food (million tonnes)" : "Corn for food (billion bushels)"
const H_SOY_Q = MU ? "Soybean oil for food (million tonnes)" : "Soybean oil for food (billion lbs)"
const H_PRICE = MU ? "Price (\\\$/liter)" : "Price (\\\$/gallon)"
const H_CORN_P = MU ? "Corn for food (\\\$/tonne)" : "Corn for food (\\\$/bushel)"
const H_SOY_P = MU ? "Soybean oil for food (\\\$/tonne)" : "Soybean oil for food (\\\$/lb)"
const H_LAND = MU ? "Land use\\tnote{1} (million hectares)" : "Land use\\tnote{1} (million acres)"
const D_FUEL_P = MU ? 3 : 2     # $/L needs one more decimal than $/gal

# Table layout: (label, observed value, statusquo value, decimals). Section headers are :header.
rows = [
    (:header, H_VOL),
    ("Jet Fuel", qgal(obs("Production", :jet_fuel)), qgal(sq_q(:jet_fuel)), 2),
    ("Gasoline", qgal(obs("Production", :gasoline)), qgal(sq_q(:gasoline)), 2),
    ("Ethanol", qgal(obs("Production", :ethanol)), qgal(sq_q(:ethanol)), 2),
    ("Diesel", qgal(obs("Production", :diesel)), qgal(sq_q(:diesel)), 2),
    (H_CORN_Q, qbu(obs("Demand", :corn)), qbu(sq_x(:corn)), 2),
    (H_SOY_Q, qlb(obs("Demand", :soyoil)), qlb(sq_x(:soyoil)), 2),
    (:header, H_PRICE),
    ("Jet", ppg(jet_pg(obs("Price_consumer", :avi))), ppg(jet_pg(sq_pc(:avi))), D_FUEL_P),
    ("Gasoline", ppg(gas_pg(obs("Price_consumer", :gas))), ppg(gas_pg(sq_pc(:gas))), D_FUEL_P),
    ("Ethanol", ppg(eth_pg(obs("Price_consumer", :gas))), ppg(eth_pg(sq_pc(:gas))), D_FUEL_P),
    ("Diesel", ppg(die_pg(obs("Price_consumer", :die))), ppg(die_pg(sq_pc(:die))), D_FUEL_P),
    (H_CORN_P, ppb(obs("Price", :feedstock_corn_n)), ppb(sq_pf(:feedstock_corn_n)), 2),
    (H_SOY_P, ppl(obs("Price", :feedstock_soy_n)), ppl(sq_pf(:feedstock_soy_n)), 2),
    (:header, H_LAND),
    ("Land used for corn", qac(LAND_CORN_OBS), qac(sq_land_corn), 2),
    ("Land used for soybean", qac(LAND_SOY_OBS), qac(sq_land_soy), 2),
]

fmt(x, d) = isnan(x) ? "" : (d == 1 ? @sprintf("%.1f", x) :
                             d == 2 ? @sprintf("%.2f", x) :
                             d == 3 ? @sprintf("%.3f", x) : @sprintf("%.0f", x))

function diffpct(obs, sq)
    (isnan(obs) || isnan(sq) || abs(obs) < 1e-12) && return ""
    @sprintf("%.1f", (sq - obs) / obs * 100)
end

io = IOBuffer()
println(io, "\\begin{table}[h]")
println(io, "\\centering")
println(io, "\\caption{Model validation with observed data in 2024}")
println(io, "\\begin{adjustbox}{max width=\\textwidth}")
println(io, "\\begin{threeparttable}")
println(io, "\\begin{tabular}{lccc}")
println(io, "\\toprule")
println(io, "\\textbf{Variable} & \\textbf{Observed 2024} & \\textbf{Status Quo (No policy)} & \\textbf{Difference (\\%)} \\\\")
println(io, "\\midrule")

for row in rows
    if row[1] === :header
        println(io, "\\textbf{$(row[2])} & & & \\\\")
    else
        label, ov, sv, d = row
        println(io, "$(label) & $(fmt(ov, d)) & $(fmt(sv, d)) & $(diffpct(ov, sv)) \\\\")
    end
end

println(io, "\\bottomrule")
println(io, "\\end{tabular}")
println(io, "\\begin{tablenotes}")
println(io, "\\footnotesize")
# The footnote quotes the USDA source, which publishes acres. Give the hectare figures
# the table body uses, with the source's own acres kept in parentheses so the citation
# stays checkable against USDA 2025a without a conversion in the reader's head.
const NOTE_HA = let ha(x) = @sprintf("%.1f", Units.acre_to_ha(x))
    MU ?
    "US corn and soybean cropland that were harvested in 2024 total $(ha(82.9)) and " *
    "$(ha(86.1)) million hectares (82.9 and 86.1 million acres, USDA 2025a). Excluding " *
    "$(ha(12.75)) and $(ha(35.95)) million hectares (12.75 and 35.95 million acres) used " *
    "for crop production for exports, which are not included in this study, this yields " *
    "$(ha(70.15)) and $(ha(50.15)) million hectares allocated to biofuel production and food." :
    "US corn and soybean cropland that were harvested in 2024 total 82.9 and 86.1 million " *
    "acres (USDA 2025a). Excluding 12.75 and 35.95 million acres used for crop production " *
    "for exports, which are not included in this study, this yields 70.15 and 50.15 million " *
    "acres allocated to biofuel production and food."
end
println(io, "\\item[1] " * NOTE_HA)
println(io, "\\end{tablenotes}")
println(io, "\\end{threeparttable}")
println(io, "\\end{adjustbox}")
println(io, "\\end{table}")

tex = String(take!(io))
write(TEX_OUT, tex)
println("✓ LaTeX table written to: ", TEX_OUT)
println("\n", "="^70, "\n")
println(tex)
