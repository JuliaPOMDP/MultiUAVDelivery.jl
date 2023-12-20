module MultiUAVDelivery

using LinearAlgebra
using Random
using Distributions
using StaticArrays
using Graphs

using POMDPs
using POMDPTools: ImplicitDistribution
using MultiAgentPOMDPs

const MAPOMDPs = MultiAgentPOMDPs

include("uav_dynamics.jl")

"""
Each UAV's action is a dynamics action or a boarding action.
"""
Base.@kwdef struct UAVGeneralAction{UA <: UAVAction}
    dyn_action::UA
    to_board::Bool
    no_op::Bool
end

## Helper functions and structs for defining the environment

# The subregions that drones have to reach - centre, radius, capacity
const CircularGoalRegion = NamedTuple{(:cen, :rad, :cap), Tuple{SVector{2,Float64}, Float64, Int64}}
const GridCoords = SVector{2,Int64} # (0, 0) means boarded

Base.@kwdef struct UAVGeneralState
    coords::GridCoords
    boarded::Bool
end

function gc_norm(params::UAVParameters, gc1::GridCoords, gc2::GridCoords)
    xy1 = [grid_coords_to_xy(params, gc1)...]
    xy2 = [grid_coords_to_xy(params, gc2)...]
    return norm(xy1 - xy2)
end

is_in_region(params::UAVParameters, reg::CircularGoalRegion, gc::GridCoords) = (norm(reg.cen - [grid_coords_to_xy(params, gc)...]) <= reg.rad)

"""
Discretize continuous x-y state to lattice coords (x, y)
"""
function xy_to_grid_coords(params::UAVParameters, cont_state::UAVState)

    # We ADD XY_LIM because we're subtracting -XY_LIM
    xcoord = convert(Int64, ceil((cont_state.x + params.XY_LIM)/params.XY_AXIS_RES))
    ycoord = convert(Int64, ceil((cont_state.y + params.XY_LIM)/params.XY_AXIS_RES))

    # Corner case when xcoord == 0 || ycoord == 0
    xcoord = (xcoord == 0) ? 1 : xcoord
    ycoord = (ycoord == 0) ? 1 : ycoord

    @assert xcoord*ycoord != 0 "State $(cont_state) maps to coords $((xcoord,ycoord))"

    return GridCoords(xcoord, ycoord)
end

"""
NOTE: (x,y) returned as a Float64 Tuple. It must then be used according to UAVState type
"""
function grid_coords_to_xy(params::UAVParameters, coords::GridCoords)

    x = -params.XY_LIM + params.XY_AXIS_RES*((2*coords[1] - 1)/2.0)
    y = -params.XY_LIM + params.XY_AXIS_RES*((2*coords[2] - 1)/2.0)

    return (x, y)
end


# To define full set of actions per UAV
function get_per_uav_actions(dynamics::UAVDynamics)

    uav_ctl_actions = get_uav_control_actions(dynamics)
    atype = eltype(uav_ctl_actions)

    per_uav_actions = UAVGeneralAction{atype}[]

    # First fill up with control actions
    for uca in uav_ctl_actions
        push!(per_uav_actions, UAVGeneralAction(uca, false, false))
    end

    # Now push BOARD
    push!(per_uav_actions, UAVGeneralAction(atype(), true, false))

    # Then push NO-OP
    push!(per_uav_actions, UAVGeneralAction(atype(), false, true))
end


# Templated on state-action-dynamics types
struct MultiUAVDeliveryMDP{UA <: UAVAction, UDM <: UAVDynamics} <: JointMDP{Vector{UAVGeneralState},Vector{UAVGeneralAction{UA}}}
    nagents::Int64
    grid_side::Int64 # Computed as convert(Int64, div(2*p.dynamics.params.XY_LIM, p.dynamics.params.XY_AXIS_RES)
    dynamics::UDM
    per_uav_actions::Vector{UAVGeneralAction{UA}} # Last action is always boarding
    discount::Float64 # Finite horizon problem....
    goal_regions::Vector{CircularGoalRegion} # Must be mutually exclusive
    region_to_uavids::Vector{Set{Int64}} # Maps region index to set of uav indices assigned to it
    uav_to_region_map::Vector{Int64} # Maps agent index to region index
    constant_cg_adj_mat::Matrix{Int64} # Should be populated at the start with adjmat
    # Reward terms here?
    reach_goal_bonus::Float64
    proximity_penalty_scaling::Float64 # Multiplied by 1/(proximity) if closer than thresh
    repulsion_penalty::Float64 # Everytime two agents target the same cell or try to board at the same time
    dynamics_cost_scaling::Float64
end


# Constructor for MDP that takes minimal required info and does the rest in place
function MultiUAVDeliveryMDP(;nagents::Int64, dynamics::UAVDynamics, discount::Float64=1.0,
                             goal_regions::Vector{CircularGoalRegion}, region_to_uavids::Vector{Set{Int64}},
                             goal_bonus::Float64, prox_scaling::Float64, repul_pen::Float64, dyn_scaling::Float64)

    per_uav_actions = get_per_uav_actions(dynamics)
    grid_side = convert(Int64, floor((2*dynamics.params.XY_LIM)/dynamics.params.XY_AXIS_RES))

    # Set up reverse map of UAV to goal region
    uav_to_region_map = Vector{Int64}(undef, nagents)
    for (reg_idx, uavset) in enumerate(region_to_uavids)
        for uavid in uavset
            uav_to_region_map[uavid] = reg_idx
        end
    end

    # set up constant CG based on regions
    cg_adj_mat = zeros(Int64, nagents, nagents)
    for reg_uavset in region_to_uavids
        for i in reg_uavset
            for j in reg_uavset
                if i != j
                    cg_adj_mat[i, j] = 1
                    cg_adj_mat[j, i] = 1
                end
            end
        end
    end

    return MultiUAVDeliveryMDP{eltype(get_uav_control_actions(dynamics)),typeof(dynamics)}(nagents, grid_side, dynamics, per_uav_actions, discount,
                               goal_regions, region_to_uavids, uav_to_region_map, cg_adj_mat,
                               goal_bonus, prox_scaling, repul_pen, dyn_scaling)
end

## POMDPs.MDP definitions and implementations
POMDPs.discount(p::MultiUAVDeliveryMDP) = p.discount

MAPOMDPs.n_agents(p::MultiUAVDeliveryMDP) = p.nagents
MAPOMDPs.agent_actions(p::MultiUAVDeliveryMDP, idx::Int64) = p.per_uav_actions

# Actions are [dynamics ... BOARD NO-OP]
function MAPOMDPs.agent_actions(p::MultiUAVDeliveryMDP, idx::Int64, s::UAVGeneralState)

    # If boarded, only no-op action can be taken
    if s.boarded
        return p.per_uav_actions[end:end]
    end

    # Otherwise, loop over regions and check if drone is in it
    for (reg_idx, reg) in enumerate(p.goal_regions)
        if is_in_region(p.dynamics.params, reg, s.coords) && p.uav_to_region_map[idx] == reg_idx
            # Then can do dynamics as well as board; only exclude NO-OP
            return p.per_uav_actions[1:end-1]
        end
    end

    # Otherwise, only dynamics; exclude BOARD as well as NO-OP
    return p.per_uav_actions[1:end-2]
end

MAPOMDPs.agent_actionindex(p::MultiUAVDeliveryMDP, idx::Int64, a) = findfirst(isequal(a), p.per_uav_actions)

POMDPs.actions(p::MultiUAVDeliveryMDP) = vec(map(collect, Iterators.product((p.per_uav_actions for i in 1:n_agents(p))...)))
POMDPs.actionindex(p::MultiUAVDeliveryMDP, a, c) = findfirst(isequal(a), p.per_uav_actions)
POMDPs.actionindex(p::MultiUAVDeliveryMDP, a) = findfirst(isequal(a), p.per_uav_actions)

# Look up generate_start_state of the UAV Dynamics model
# Also reject states that start inside a goal region
function POMDPs.initialstate(p::MultiUAVDeliveryMDP)
    ImplicitDistribution() do rng
        initstate = UAVGeneralState[]

        # Using while loop because we'll do rejection sampling
        i = 1
        while i <= n_agents(p)
            rand_gc = xy_to_grid_coords(p.dynamics.params, generate_start_state(p.dynamics, rng))

            invalid_coords = true
            for reg in p.goal_regions
                if is_in_region(p.dynamics.params, reg, rand_gc) || (UAVGeneralState(rand_gc, false) in initstate)
                    invalid_coords = false
                    break
                end
            end

            if invalid_coords
                push!(initstate, UAVGeneralState(rand_gc, false))
                i += 1
            end
        end
        return initstate
    end
end

# NOTE: Needs to be y first so that LinearIndices[x,y] maps correctly
function MAPOMDPs.agent_states(p::MultiUAVDeliveryMDP, idx::Int64)
    coordsset = vec(GridCoords[GridCoords(x, y) for y = 1:p.grid_side for x = 1:p.grid_side])

    stateset = Vector{UAVGeneralState}(undef, 2*length(coordsset))
    for (i, coords) in enumerate(coordsset)
        stateset[2*i-1] = UAVGeneralState(coords, false)
        stateset[2*i] = UAVGeneralState(coords, true)
    end

    return stateset
end

function MAPOMDPs.agent_stateindex(p::MultiUAVDeliveryMDP, idx::Int64, s::UAVGeneralState)

    # First look up index only based on coords
    # 2*coords_idx - 1 for false; 2*coords_idx for true
    coord_idx = LinearIndices((1:p.grid_side, 1:p.grid_side))[s.coords...]

    if s.boarded
        return 2*coord_idx
    else
        return 2*coord_idx - 1
    end
end

# Terminal state when all UAVs have boarded
# NOTE: This doesn't strictly need the state since it looks up the MDP....
POMDPs.isterminal(p::MultiUAVDeliveryMDP, s::AbstractVector{UAVGeneralState}) = all(x -> x.boarded, s)


coord_graph_adj_mat(p::MultiUAVDeliveryMDP) = p.constant_cg_adj_mat

function coord_graph_adj_mat(p::MultiUAVDeliveryMDP, s::AbstractVector{UAVGeneralState})

    state_cg_mat = deepcopy(p.constant_cg_adj_mat)

    # Iterate over states and if both are in different cliques AND Closer than threshold, add edge
    for (i, si) in enumerate(s)
        for (j, sj) in enumerate(s)
            if i != j && p.uav_to_region_map[i] != p.uav_to_region_map[j] &&
                gc_norm(p.dynamics.params, si.coords, sj.coords) <= p.dynamics.params.CG_PROXIMITY_THRESH
                state_cg_mat[i, j] = 1
                state_cg_mat[j, i] = 1
            end
        end
    end

    return state_cg_mat
end

function MAPOMDPs.coordination_graph(p::MultiUAVDeliveryMDP, s)
    SimpleGraph(coord_graph_adj_mat(p, s))
end
function MAPOMDPs.coordination_graph(p::MultiUAVDeliveryMDP)
    SimpleGraph(coord_graph_adj_mat(p))
end

function POMDPs.gen(p::MultiUAVDeliveryMDP, s, a, rng)
    nagents = n_agents(p)
    coordgraph = coordination_graph(p, s)
    sp_vec = Vector{UAVGeneralState}(undef, nagents)

    # NOTE: Local reward vector, different for each agent
    r_vec = Vector{Float64}(undef, nagents)

    # Simple way of doing this:
    # Consider every agent at a time (so focusing on its reward and next state)
    # Consider each of its neighbors in the CG
    # If it tries to board, check that no neighbor is in threshold and if successful, get boarding bonus (only that Agent?), else boarding penalty?
    # If it tries to move, check if any neighbor prevents it (repulsion penalty), otherwise dynamics cost
    # If it has boarded, it should only be no-op and not change it's indiv state (0,0)
    n_prox = 0
    n_coll = 0
    for idx = 1:nagents

        s_idx = s[idx]
        a_idx = a[idx]

        nbrs = neighbors(coordgraph, idx)

        # Branch on the type of action
        if a_idx.no_op

            # Nothing here, just copy over state (which should be 0,0)
            @assert s_idx.boarded == true "No-op action taken by agent $(idx) in state $((s_idx.coords, s_idx.boarded))"
            sp_vec[idx] = s_idx
            r_vec[idx] = 0.0

        elseif a_idx.to_board

            @assert (s_idx.boarded == false && is_in_region(p.dynamics.params, p.goal_regions[p.uav_to_region_map[idx]],
                s_idx.coords)) "Board action taken by
                agent $(idx) of state $(s_idx.coords) when not in region $(p.goal_regions[p.uav_to_region_map[idx]].cen) !"

            any_nbr_boarding = false
            for n in nbrs
                # If neighbor in same goal is boarding, prevent and penalize
                if a[n].to_board && p.uav_to_region_map[n] == p.uav_to_region_map[idx]
                    any_nbr_boarding = true
                    break
                end
            end

            if any_nbr_boarding
                # Repulsion penalty; stay in-place
                sp_vec[idx] = s_idx
                r_vec[idx] = -p.repulsion_penalty
                n_coll += 1
            else
                # Successful boarding; same location and board true
                sp_vec[idx] = UAVGeneralState(s_idx.coords, true)
                r_vec[idx] = p.reach_goal_bonus
            end

        else
            # Dynamics action
            temp_new_uav_state = next_uav_state(p.dynamics, FirstOrderUAVState(grid_coords_to_xy(p.dynamics.params, s_idx.coords)...),
                                                a_idx.dyn_action, rng)
            new_coords = xy_to_grid_coords(p.dynamics.params, temp_new_uav_state)
            sp_vec[idx] = UAVGeneralState(new_coords, false)
            r_vec[idx] = -p.dynamics_cost_scaling*dynamics_cost(p.dynamics, a_idx.dyn_action)
        end
    end # idx in 1:n_agents


    # Do another loop over cells and if any two in same cell, move all of them back to their original cell
    any_clashing_positions = true
    while any_clashing_positions
        clashing_positions = Dict{Int64,Set{Int64}}()
        for (i, si) in enumerate(sp_vec)
            si_coord_idx = LinearIndices((1:p.grid_side, 1:p.grid_side))[si.coords...]
            if ~(haskey(clashing_positions, si_coord_idx))
                clashing_positions[si_coord_idx] = Set{Int64}(i)
            else
                push!(clashing_positions[si_coord_idx], i)
            end
        end

        any_clashing_positions=false
        # Now loop through and set back states that are in the same cell
        # And assign them the repulsion penalty
        for (coord_idx, agents) in clashing_positions
            if length(agents) > 1
                # Need to do at least one more round
                any_clashing_positions = true
                for agt in agents
                    sp_vec[agt] = s[agt]
                    r_vec[agt] = -p.repulsion_penalty
                    n_coll += 1
                end
            end
        end
    end

    # Now we are guaranteed no clashing, loop over states and add reward if closer to goal
    for (idx, (si, spi)) in enumerate(zip(s, sp_vec))
        idx_goal = p.goal_regions[p.uav_to_region_map[idx]]
        rel_dist = norm([grid_coords_to_xy(p.dynamics.params, si.coords)...] - idx_goal.cen) -
                norm([grid_coords_to_xy(p.dynamics.params, spi.coords)...] - idx_goal.cen)

        if abs(rel_dist) > p.dynamics.params.XY_AXIS_RES/2
            r_vec[idx] += p.proximity_penalty_scaling*(1.0/rel_dist)
            n_prox += 1
        end
    end

    # Finally, loop through sp and add a penalty for any pair too close to each other
    for i = 1:p.nagents-1
        if sp_vec[i].boarded == false
            for j = i+1:p.nagents
                if sp_vec[j].boarded == false
                    dist = gc_norm(p.dynamics.params, sp_vec[j].coords, sp_vec[i].coords)

                    @assert dist > p.dynamics.params.XY_AXIS_RES/2.0 "Dist between $((i, j)) of $((sp_vec[i].coords, sp_vec[i].boarded)) and $((sp_vec[j].coords, sp_vec[j].boarded)) is 0!"

                    if dist <= p.dynamics.params.PROXIMITY_THRESH
                        r_vec[i] -= p.proximity_penalty_scaling*(1.0/dist)
                        r_vec[j] -= p.proximity_penalty_scaling*(1.0/dist)
                        n_prox += 2
                    end
                end
            end
        end
    end

    return (sp=sp_vec, r=r_vec, info=(proximity=n_prox, collisions=n_coll))
end

include("utils.jl")

const PSET = [(nagents=8, XY_AXIS_RES=0.2, XYDOT_LIM=0.2, XYDOT_STEP=0.2, NOISESTD=0.1),
              (nagents=16, XY_AXIS_RES=0.1, XYDOT_LIM=0.1, XYDOT_STEP=0.1, NOISESTD=0.05),
              (nagents=32, XY_AXIS_RES=0.08, XYDOT_LIM=0.08, XYDOT_STEP=0.08, NOISESTD=0.05),
              (nagents=48, XY_AXIS_RES=0.05, XYDOT_LIM=0.05, XYDOT_STEP=0.05, NOISESTD=0.02)]

function FirstOrderMultiUAVDelivery(; pset=PSET[1], rewset=(goal_bonus=1000.0, prox_scaling = 1.0, repul_pen=10.0, dynamics_scaling=10.0), seed=7)
    uavparams = UAVParameters(XY_AXIS_RES=pset.XY_AXIS_RES,
                              XYDOT_LIM=pset.XYDOT_LIM,
                              XYDOT_STEP=pset.XYDOT_STEP,
                              PROXIMITY_THRESH=1.5*pset.XY_AXIS_RES,
                              CG_PROXIMITY_THRESH=3.0*pset.XY_AXIS_RES)

    dynamics = FirstOrderUAVDynamics(timestep=1.0,
                                    noise=Distributions.MvNormal(LinearAlgebra.Diagonal(map(abs2, [pset.NOISESTD, pset.NOISESTD]))),
                                     params=uavparams)

    goal_regions, region_to_uavids = get_quadrant_goal_regions(pset.nagents,
                                                               pset.XY_AXIS_RES,
                                                               MersenneTwister(seed))

    mdp = MultiUAVDeliveryMDP(nagents=pset.nagents, dynamics=dynamics,
                              goal_regions=goal_regions, region_to_uavids=region_to_uavids,
                              goal_bonus=rewset.goal_bonus, prox_scaling=rewset.prox_scaling,
                              repul_pen=rewset.repul_pen, dyn_scaling=rewset.dynamics_scaling)

    return mdp
end

export MultiUAVDeliveryMDP, FirstOrderMultiUAVDelivery

end
