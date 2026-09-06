module Units

# =================================================================================
# US customary -> metric conversion
# =================================================================================

using DataFrames

# =================================================================================
# 1. Exact definitions
# =================================================================================

const KG_PER_LB = 0.45359237
const L_PER_GAL = 3.785411784
const HA_PER_ACRE = 0.40468564224
const KM_PER_MILE = 1.609344

const KG_PER_BU_CORN = 56 * KG_PER_LB   # 25.40117272
const KG_PER_BU_SOY = 60 * KG_PER_LB   # 27.21554220

# =================================================================================
# 2. Master switch
# =================================================================================
# Set to false to reproduce the old US-unit output byte for byte
const METRIC = true

# =================================================================================
# 3. Conversion table
# =================================================================================
# A factor of 1.0 means "already metric, label only".

const CONV = Dict{String,Tuple{Float64,String}}(

    # --- fuel volume -------------------------------------------------------------
    "B gal" => (L_PER_GAL, "B liters"),
    "\$/gal" => (1 / L_PER_GAL, "\$/L"),

    # --- crop mass ------------------
    "B bushel" => (KG_PER_BU_CORN, "MMT"),
    "\$/bushel" => (1000 / KG_PER_BU_CORN, "\$/tonne"),

    # --- oil / feedstock mass ----------------------------------------------------
    "B lb" => (KG_PER_LB, "MMT"),
    "\$/lb" => (1000 / KG_PER_LB, "\$/tonne"),

    # --- land --------------------------------------------------------------------
    "M acres" => (HA_PER_ACRE, "M ha"),
    "\$/acre" => (1 / HA_PER_ACRE, "\$/ha"),

    # --- travel services ---------------------------------------------------------
    "B RPM" => (KM_PER_MILE, "B passenger-km"),
    "\$/RPM" => (1 / KM_PER_MILE, "\$/passenger-km"),
    "B VMT" => (KM_PER_MILE, "B vehicle-km"),
    "\$/VMT" => (1 / KM_PER_MILE, "\$/vehicle-km"),

    # --- already metric: relabel only -------------------------------------------
    "B ton CO2e" => (1.0, "B tonne CO2e"),
    "\$/ton CO2e" => (1.0, "\$/tonne CO2e"),
    "\$/tonCO2" => (1.0, "\$/tonne CO2e"),
    "\$/metric ton" => (1.0, "\$/tonne"),
    "MMT" => (1.0, "MMT"),

    # --- dimensionless -----------------------------------------------------------
    "B\$" => (1.0, "B\$"),
    "Various" => (1.0, "Various"),
    "0/1" => (1.0, "0/1"),
    "-" => (1.0, "-"),
)

# =================================================================================
# 4. Scalar API
# =================================================================================

# factor(unit): multiplicative factor taking a value in `unit` to its metric counterpart.
# Returns 1.0 when METRIC is off. Throws on an unrecognised unit string.
function factor(unit::AbstractString)
    METRIC || return 1.0
    haskey(CONV, unit) || error(
        "Units: no conversion registered for unit \"$unit\". " *
        "Add it to CONV in main/units.jl rather than letting the row through unconverted.")
    return CONV[unit][1]
end

# label(unit): the metric label replacing `unit`. Returns `unit` unchanged when METRIC is off.
function label(unit::AbstractString)
    METRIC || return String(unit)
    haskey(CONV, unit) || error("Units: no conversion registered for unit \"$unit\".")
    return CONV[unit][2]
end

# convert_value(x, unit): convert one value. `missing` and non-numeric entries pass
# through untouched, which is what the Observed / Validation columns need.
convert_value(x, unit::AbstractString) =
    (ismissing(x) || !(x isa Number)) ? x : x * factor(unit)

# =================================================================================
# 5. DataFrame API
# =================================================================================

# convert_df!(df; unit_col, skip_cols): convert every numeric cell of `df` in place using
# the per-row unit in `unit_col`, then rewrite that column with the metric labels. This is
# the path used by main/extract_results.jl, whose Unit column already carries one label
# per row. Returns `df`.
function convert_df!(df::DataFrame;
    unit_col::Symbol=:Unit,
    skip_cols=(:Category, :Variable, :Unit))
    METRIC || return df
    hasproperty(df, unit_col) ||
        error("Units.convert_df!: no column $(unit_col) in the DataFrame.")

    value_cols = [c for c in propertynames(df) if !(c in skip_cols)]

    for i in 1:nrow(df)
        u = string(df[i, unit_col])
        f = factor(u)
        if f != 1.0
            for c in value_cols
                df[i, c] = convert_value(df[i, c], u)
            end
        end
        df[i, unit_col] = label(u)
    end
    return df
end

# convert_cols!(df, mapping): convert selected columns of a wide DataFrame, where
# `mapping` gives `column => unit` with the same unit strings as CONV. This is the path
# for the figure scripts, whose frames are one row per scenario and one column per
# variable, so there is no per-row Unit column to drive convert_df!.
# Columns absent from `df` are skipped, so one mapping can serve several scripts. An
# unregistered unit string still raises. Returns `df`.
function convert_cols!(df::DataFrame, mapping)
    METRIC || return df
    for (col, unit) in mapping
        c = Symbol(col)
        hasproperty(df, c) || continue
        f = factor(unit)
        f == 1.0 && continue
        df[!, c] = [convert_value(v, unit) for v in df[!, c]]
    end
    return df
end

# =================================================================================
# 6. Direct helpers
# =================================================================================
# For figure and table code that has no Unit column and converts a known quantity.
# Named so the direction is unmistakable at the call site.

gal_to_L(x) = METRIC ? x * L_PER_GAL : x   # B gal    -> B liters
bu_to_Mt(x) = METRIC ? x * KG_PER_BU_CORN : x   # B bu     -> MMT (corn)
lb_to_Mt(x) = METRIC ? x * KG_PER_LB : x   # B lb     -> MMT
acre_to_ha(x) = METRIC ? x * HA_PER_ACRE : x   # M acres  -> M ha
mile_to_km(x) = METRIC ? x * KM_PER_MILE : x   # B miles  -> B km

price_gal_to_L(p) = METRIC ? p / L_PER_GAL : p   # $/gal    -> $/L
price_bu_to_t(p) = METRIC ? p * 1000 / KG_PER_BU_CORN : p # $/bu   -> $/tonne (corn)
price_lb_to_t(p) = METRIC ? p * 1000 / KG_PER_LB : p   # $/lb    -> $/tonne
price_acre_to_ha(p) = METRIC ? p / HA_PER_ACRE : p   # $/acre  -> $/ha
price_mile_to_km(p) = METRIC ? p / KM_PER_MILE : p   # $/mile  -> $/km

# =================================================================================
# 7. Axis label helper
# =================================================================================
# Figure axes spell the unit out ("billion liters") the same way the CSV labels do,
# matching how they read today ("billion gallons").  Keeping both spellings in one place
# stops the CSV and the figures from drifting apart.

const AXIS = Dict{String,String}(
    "billion gallons" => "billion liters",
    "B gal" => "B liters",
    "billion bushels" => "million tonnes",
    "billion lbs" => "million tonnes",
    "million acres" => "million hectares",
    "Million Acres" => "Million Hectares",
    "billion miles" => "billion km",
    "billion RPM" => "billion passenger-km",
    "\$/gallon" => "\$/liter",
    "\$/bushel" => "\$/tonne",
    "\$/lb" => "\$/tonne",
    "\$/acre" => "\$/hectare",
)

# axis(text): rewrite the unit words inside a free-form axis label. Longest key first,
# so "billion gallons" matches before "B gal" could interfere.
function axis(text::AbstractString)
    METRIC || return String(text)
    out = String(text)
    for k in sort!(collect(keys(AXIS)), by=length, rev=true)
        out = replace(out, k => AXIS[k])
    end
    return out
end

export METRIC, factor, label, convert_value, convert_df!, convert_cols!, axis,
    gal_to_L, bu_to_Mt, lb_to_Mt, acre_to_ha, mile_to_km,
    price_gal_to_L, price_bu_to_t, price_lb_to_t, price_acre_to_ha, price_mile_to_km

end # module
