module Paths

# =================================================================================
# Where output goes
# =================================================================================

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

# Root of the output tree: SAF_OUTPUT if set, otherwise output/ inside the repo.
const BASE = get(ENV, "SAF_OUTPUT", joinpath(REPO_ROOT, "output"))

const DATA_DIR = joinpath(BASE, "data")
const TABLE_DIR = joinpath(BASE, "tables")
const FIGURE_DIR = joinpath(BASE, "figures")

# setup(): create the three directories if missing and print where output actually
# lands. Call once at the top of a script. The printed path is the safeguard: if you
# think SAF_OUTPUT is set and it is not, results drop into the repo silently.
function setup()
    for d in (DATA_DIR, TABLE_DIR, FIGURE_DIR)
        isdir(d) || mkpath(d)
    end
    src = haskey(ENV, "SAF_OUTPUT") ? "ENV[\"SAF_OUTPUT\"]" : "default (in repo)"
    println("Output base: ", BASE, "   [", src, "]")
    return nothing
end

# variant(name): output directories for a robustness branch, one level below each
# kind rather than above it.
#
#     v = Paths.variant("cet_land")
#     v.data / v.tables / v.figures  ->  <BASE>/{data,tables,figures}/cet_land
#
# Nesting this way keeps one classification: open figures/ to see every figure, and
# branch output still never mixes with main output. Creates the three directories if
# missing and returns them as a NamedTuple.
function variant(name::AbstractString)
    dirs = (data=joinpath(DATA_DIR, name),
        tables=joinpath(TABLE_DIR, name),
        figures=joinpath(FIGURE_DIR, name))
    for d in dirs
        isdir(d) || mkpath(d)
    end
    src = haskey(ENV, "SAF_OUTPUT") ? "ENV[\"SAF_OUTPUT\"]" : "default (in repo)"
    println("Output base: ", BASE, "   [", src, "]   variant: ", name)
    return dirs
end

end # module Paths
