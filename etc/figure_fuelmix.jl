# plot_fuel_mix.jl
cd(@__DIR__)

using CSV
using DataFrames
using PyCall

const OUTPUT_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/results"
const FIG_DIR = "/Users/sohyeonserenryu/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/CS SAF policy/output/figures"

# =================================================================================
# 1. Load CSV
# =================================================================================

df = CSV.read(joinpath(OUTPUT_DIR, "results_comprehensive.csv"), DataFrame)

function get_val(df, variable, col)
    rows = filter(r -> r.Variable == variable, df)
    if nrow(rows) == 0 || ismissing(rows[1, col])
        return 0.0
    end
    return Float64(rows[1, col])
end

# =================================================================================
# 2. Scenarios
# =================================================================================

scenario_cols = ["Status Quo", "EquivEmission_CarbonTax", "EquivEmission_RFS",
    "EquivEmission_LCFS", "EquivEmission_TaxCredit"]
scenario_labels = ["Status Quo", "Carbon Tax", "RFS", "LCFS", "Tax Credit"]
n = length(scenario_cols)

# =================================================================================
# 3. Component definitions
# =================================================================================

# Jet fuel panel (fossil jet fuel only)
jet_vars = ["jet_fuel"]
jet_labels = ["Jet Fuel"]
jet_colors = ["#4A4A4A"]
jet_hatches = [""]
jet_ec = ["#4A4A4A"]

# Aviation SAF panel
avi_vars = ["saf_hefa_conv", "saf_hefa_cs", "saf_hefa_nonsoy", "saf_atj_conv", "saf_atj_cs"]
avi_labels = ["HEFA (conv)", "HEFA (CS)", "HEFA (non-soy)", "ATJ (conv)", "ATJ (CS)"]
avi_colors = ["#BA8127", "#BA8127", "#BA8127", "#E4BC7D", "#E4BC7D"]
avi_hatches = ["", "////", "....", "", "////"]
avi_ec = ["#9A7D30", "black", "black", "#5A4010", "black"]

# Diesel biofuels
die_vars = ["rd_soy", "rd_nonsoy", "biodiesel_soy", "biodiesel_nonsoy"]
die_labels = ["RD (soy)", "RD (non-soy)", "Biodiesel (soy)", "Biodiesel (non-soy)"]
die_colors = ["#8B0000", "#8B0000", "#E8A0A0", "#E8A0A0"]
die_hatches = ["", "....", "", "...."]
die_ec = ["#6B0000", "black", "#C07070", "black"]

# Gasoline biofuels
gas_vars = ["ethanol"]
gas_labels = ["Ethanol"]
gas_colors = ["#1B4F72"]
gas_hatches = [""]
gas_ec = ["#1B4F72"]

# =================================================================================
# 4. Build data matrices
# =================================================================================

function build_matrix(vars, cols)
    mat = zeros(length(vars), length(cols))
    for (j, col) in enumerate(cols)
        for (i, var) in enumerate(vars)
            mat[i, j] = get_val(df, var, col)
        end
    end
    return mat
end

jet_data = build_matrix(jet_vars, scenario_cols)
avi_data = build_matrix(avi_vars, scenario_cols)
die_data = build_matrix(die_vars, scenario_cols)
gas_data = build_matrix(gas_vars, scenario_cols)

# =================================================================================
# 5. Draw panel function
# =================================================================================

function draw_panel!(ax, data_mat, colors, hatches, edge_colors, comp_labels;
    show_yticks=true, title="", ytick_labels=nothing, xlim_right=nothing)
    n_comp, n_scen = size(data_mat)
    y = collect(0:(n_scen-1))

    for i in 1:n_comp
        offsets = vec(sum(data_mat[1:i-1, :], dims=1))
        vals = data_mat[i, :]
        ax.barh(y, vals,
            left=offsets,
            height=0.6,
            color=colors[i],
            hatch=hatches[i],
            edgecolor=length(hatches[i]) == 0 ? "white" : edge_colors[i],
            linewidth=0.5,
            label=comp_labels[i]
        )
    end

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("bil. gallons", fontsize=9)
    ax.invert_yaxis()

    if show_yticks && !isnothing(ytick_labels)
        ax.set_yticks(y)
        ax.set_yticklabels(ytick_labels, fontsize=10)
    else
        ax.set_yticks(y)
        ax.set_yticklabels(fill("", n_scen))
    end

    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0, xlim_right)
end

# =================================================================================
# 6. Backend + plot
# =================================================================================

mpl = pyimport("matplotlib")
mpl.use("MacOSX")
mplt = pyimport("matplotlib.pyplot")
patches = pyimport("matplotlib.patches")

mpl.rcParams["hatch.linewidth"] = 5.0

fig, axes = mplt.subplots(1, 4,
    figsize=(17, 5.0),
    gridspec_kw=Dict("wspace" => 0.06)
)

draw_panel!(axes[1], jet_data, jet_colors, jet_hatches, jet_ec, jet_labels;
    show_yticks=true,
    title="Jet Fuel (bil. gallons)",
    ytick_labels=scenario_labels,
    xlim_right=21)

draw_panel!(axes[2], avi_data, avi_colors, avi_hatches, avi_ec, avi_labels;
    show_yticks=false,
    title="SAF (bil. gallons)",
    xlim_right=4)

draw_panel!(axes[3], die_data, die_colors, die_hatches, die_ec, die_labels;
    show_yticks=false,
    title="Diesel Biofuels (bil. gallons)",
    xlim_right=6)

draw_panel!(axes[4], gas_data, gas_colors, gas_hatches, gas_ec, gas_labels;
    show_yticks=false,
    title="Ethanol (bil. gallons)",
    xlim_right=16)

# =================================================================================
# 7. Two-row legend
# =================================================================================

row1_labels = ["Jet Fuel", "HEFA", "ATJ", "RD", "BD", "Ethanol"]
row1_colors = ["#4A4A4A", "#BA8127", "#E4BC7D", "#8B0000", "#E8A0A0", "#1B4F72"]
row1_handles = [patches.Patch(facecolor=c, edgecolor=c, label=l)
                for (c, l) in zip(row1_colors, row1_labels)]

row2_labels = ["Conventional", "Climate-Smart", "Non-soy"]
row2_hatches = ["", "////", "...."]
row2_handles = [patches.Patch(facecolor="white", edgecolor="black",
    hatch=h, linewidth=0.8, label=l)
                for (h, l) in zip(row2_hatches, row2_labels)]

leg1 = fig.legend(row1_handles, row1_labels,
    loc="lower center",
    ncol=3,
    fontsize=9,
    frameon=false,
    bbox_to_anchor=(0.5, 0.13),
    handlelength=2.0,
    handleheight=1.2,
)

fig.add_artist(leg1)
leg2 = fig.legend(row2_handles, row2_labels,
    loc="lower center",
    ncol=3,
    fontsize=9,
    frameon=false,
    bbox_to_anchor=(0.5, 0.01),
    handlelength=2.0,
    handleheight=1.2,
)

mplt.subplots_adjust(bottom=0.30, top=0.93, left=0.09, right=0.98)

# =================================================================================
# 8. Save + show
# =================================================================================

mplt.savefig(joinpath(FIGURE_DIR, "fig_fuel_mix.png"), bbox_inches="tight", dpi=300)
println("✓ Saved fig_fuel_mix.pdf / .png")

mplt.show()