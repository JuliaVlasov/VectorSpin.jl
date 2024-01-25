module VectorSpin

using DocStringExtensions
using FFTW
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

include("h2.jl")
include("h2fh.jl")
include("he.jl")
include("haa.jl")
include("h3.jl")
include("h3fh.jl")
include("hp.jl")
include("h1fh.jl")
include("hv.jl")

end
