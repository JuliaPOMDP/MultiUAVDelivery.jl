## Utils to parameterically generate a problem
# NOTE: Everything below assumes XY_LIM = 1.0

# Computes a goal region radius appropriate for its capacity and resolution
# Basically we want it such that the inscribed square in the circle has enough
# slots to accommodate the capacity and then some. We don't want a tiling puzzle
# to have to be solved.
function get_goal_radius(capacity::Int64, axis_res::Float64)

    rp = sqrt((capacity*axis_res^2)/2.0)
    r = (ceil(2*rp/axis_res)+1.0)*axis_res/2.0

    return min(r, 0.5)
end # function get_goal_radius

# Hardcode the goal regions at the centre of each quadrant
# The capacity is randomly sampled by dividing the agents into four chunks
# and sampling around them.
function get_quadrant_goal_regions(nagents::Int64, axis_res::Float64, rng::AbstractRNG)

    @assert mod(nagents, 8) == 0 "Num. agents $(nagents) is not a multiple of 4!"

    # Divide nagents into 4 chunks and sample around them
    half_nagents = div(nagents, 2)
    oct_nagents = div(nagents, 8)

    # Doing this explicitly instead of as a vector for more clarity
    # Each next cap is necessarily less than half_nagents
    cap_quad1 = rand(rng, oct_nagents:3*oct_nagents)
    cap_quad2 = rand(rng, oct_nagents:3*oct_nagents)

    # Opposing quadrants add up to the same nagents/2
    cap_quad3 = half_nagents - cap_quad1
    cap_quad4 = half_nagents - cap_quad2

    # Set up the four goal regions explicitly
    reg_quad1 = CircularGoalRegion((cen=[0.5, 0.5], rad=get_goal_radius(cap_quad1, axis_res), cap=cap_quad1))
    reg_quad2 = CircularGoalRegion((cen=[-0.5, 0.5], rad=get_goal_radius(cap_quad2, axis_res), cap=cap_quad2))
    reg_quad3 = CircularGoalRegion((cen=[-0.5, -0.5], rad=get_goal_radius(cap_quad3, axis_res), cap=cap_quad3))
    reg_quad4 = CircularGoalRegion((cen=[0.5, -0.5], rad=get_goal_radius(cap_quad4, axis_res), cap=cap_quad4))
    goal_regs = [reg_quad1, reg_quad2, reg_quad3, reg_quad4]

    reg_to_uavid = Set{Int64}[]
    idx = 1
    for gr in goal_regs
        push!(reg_to_uavid, Set{Int64}(idx:idx+gr.cap-1))
        idx = idx+gr.cap
    end

    return goal_regs, reg_to_uavid
end # function get_quadrant_goal_regions
