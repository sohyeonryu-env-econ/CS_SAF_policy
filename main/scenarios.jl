module Scenarios

# =================================================================================
# The single definition point for cases and scenarios.
# =================================================================================
#
# This list used to be copied across six files (unified_benchmark, 3B_outcomes,
# extract_results, fig_stacked_paths, mac_7scenarios, fig_land_use). Editing one
# label meant editing six places, and missing one made a table and a figure
# disagree silently.
#
# Two layers live here:
#
#   Layer 1  CASES  what gets solved. Used by unified_benchmark.jl; becomes the jld2 keys.
#   Layer 2  SCEN   what gets reported. Table columns and figure axes.
#
# Layer 2 points at layer 1 through (policy, case), so dropping a case visibly
# requires fixing SCEN in the same file.

# =================================================================================
# Layer 1. Cases = (credit climate-smart practice) x (50% CI threshold)
# =================================================================================
#
# The carbon tax is dropped in the threshold cases (case2, case4): use_ci_threshold
# only touches saf_credit and RFS / LCFS eligibility, so it never reaches the tax.
# Keeping it there would solve the same problem twice.

const CASES = [
    (name=:case1, recognize_cs=false, use_ci_threshold=false, policies=[:carbontax, :rfs, :lcfs, :taxcredit]),
    (name=:case2, recognize_cs=false, use_ci_threshold=true, policies=[:rfs, :lcfs, :taxcredit]),
    (name=:case3, recognize_cs=true, use_ci_threshold=false, policies=[:carbontax, :rfs, :lcfs, :taxcredit]),
    (name=:case4, recognize_cs=true, use_ci_threshold=true, policies=[:rfs, :lcfs, :taxcredit]),
]

# Case name -> (recognize_cs, use_ci_threshold). Used to build configs in grid sweeps.
const CASE_FLAGS = Dict(c.name => (recognize_cs=c.recognize_cs,
    use_ci_threshold=c.use_ci_threshold) for c in CASES)

# =================================================================================
# Layer 2. Reported scenarios (a) to (i)
# =================================================================================
#
# Three separate text fields, because they are consumed in different places:
#
#   label   figure titles and legends. "\n" is a line break; one_line() flattens it.
#   short   bar chart tick labels, broken up more finely to fit a narrow slot.
#   name    policy name in the validation CSV. CS and threshold status go in the
#           cs / thr columns instead, so they stay out of the name.
#
# cs / thr of nothing means "not applicable". The (b) carbon tax produces no SAF at
# all, so there is no CS pathway to credit; writing "no CS" would advertise a design
# choice that never binds.

const SCEN = [
    (tag="(a)", policy=:statusquo, case=:base,
        label="Status quo", short="Status\nquo",
        name="Status quo", cs=nothing, thr=nothing),
    (tag="(b)", policy=:carbontax, case=:case1,
        label="Carbon tax", short="Carbon\ntax",
        name="Carbon tax", cs=nothing, thr=nothing),
    (tag="(c)", policy=:rfs, case=:case1,
        label="Volumetric mandate:\nwithout crediting CS", short="Vol.\nmandate\nno CS",
        name="Volumetric mandate", cs=false, thr=false),
    (tag="(d)", policy=:rfs, case=:case3,
        label="Volumetric mandate:\ncrediting CS", short="Vol.\nmandate\nCS",
        name="Volumetric mandate", cs=true, thr=false),
    (tag="(e)", policy=:rfs, case=:case4,
        label="Volumetric mandate:\ncrediting CS + CI threshold", short="Vol.\nmandate\nCS+thr",
        name="Volumetric mandate", cs=true, thr=true),
    (tag="(f)", policy=:lcfs, case=:case1,
        label="CI standard:\nwithout crediting CS", short="CI\nstandard\nno CS",
        name="CI standard", cs=false, thr=false),
    (tag="(g)", policy=:lcfs, case=:case3,
        label="CI standard:\ncrediting CS", short="CI\nstandard\nCS",
        name="CI standard", cs=true, thr=false),
    (tag="(h)", policy=:taxcredit, case=:case1,
        label="Tax credit:\nwithout crediting CS", short="Tax\ncredit\nno CS",
        name="Tax credit", cs=false, thr=false),
    (tag="(i)", policy=:taxcredit, case=:case3,
        label="Tax credit:\ncrediting CS", short="Tax\ncredit\nCS",
        name="Tax credit", cs=true, thr=false),
]

# =================================================================================
# Helpers
# =================================================================================

# Flatten label line breaks into spaces, for tables and console output.
one_line(s) = replace(s.label, "\n" => " ")

# Is this the status quo? Only this one has case :base, read from a different jld2 slot.
is_sq(s) = s.case === :base

# Scenarios minus the given tags, e.g. except(SCEN, "(a)", "(d)").
except(list, tags...) = filter(s -> !(s.tag in tags), list)

# Pick one by tag, to attach file-specific fields (row/col, grid, color).
by_tag(list, tag) = list[findfirst(s -> s.tag == tag, list)]

# Attach file-specific fields to SCEN entries: decorate([("(b)", (row=1, col=1)), ...]).
# Things that only mean something in one file (panel layout, sweep grids) stay in that
# file rather than here.
decorate(pairs) = [merge(by_tag(SCEN, t), extra) for (t, extra) in pairs]

end # module Scenarios
