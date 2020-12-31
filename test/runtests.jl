using MultiUAVDelivery
using POMDPs
using MultiAgentPOMDPs
using Random
using Test

@testset "MultiUAVDelivery.jl" begin
    
    mdp = FirstOrderMultiUAVDelivery()
    @test mdp isa MultiUAVDeliveryMDP
    s = rand(initialstate(mdp))
    a = [rand(agent_actions(mdp, i, s[i])) for i in 1:n_agents(mdp)]
    @inferred @gen(:sp, :r)(mdp, s, a, MersenneTwister(42))
end
