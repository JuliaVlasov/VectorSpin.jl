module VectorSpin

using DocStringExtensions
using GenericFFT
using .Threads
import DispersionRelations: fit_complex_frequency

export fit_complex_frequency

include("mesh.jl")
include("initialfields.jl")
include("initialfunction.jl")
include("diagnostics.jl")

abstract type AbstractAdvection end

include("psm.jl")
include("bspline.jl")

include("h2fh.jl")
include("he.jl")
include("haa.jl")
include("h3fh.jl")
include("h1f.jl")
include("h1fh.jl")
include("hv.jl")

end
