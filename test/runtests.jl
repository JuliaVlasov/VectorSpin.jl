using Test
using VectorSpin

@testset "Spin Ions" begin

    include("ions.jl")

end


include("vlasov-maxwell.jl")
