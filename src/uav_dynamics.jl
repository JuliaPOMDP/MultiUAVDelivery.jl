abstract type UAVDynamics end
abstract type UAVState end
abstract type UAVAction end

# Should have a superset of all params of interest for any UAV model
# NOTE: Please ensure XY_LIM divisible by XY_AXIS_RES
# and XYDOT_LIM divisible by XYDOT_STEP
Base.@kwdef struct UAVParameters
    XY_LIM::Float64 = 1.0 # Only default param; always [-XY_LIM,-XY_LIM] to  [XY_LIM,XY_LIM]
    XY_AXIS_RES::Float64
    XYDOT_LIM::Float64
    XYDOT_STEP::Float64
    PROXIMITY_THRESH::Float64
    CG_PROXIMITY_THRESH::Float64
end


Base.@kwdef struct FirstOrderUAVState <: UAVState
    x::Float64  = 0.0
    y::Float64  = 0.0
end

Base.@kwdef struct FirstOrderUAVAction <: UAVAction
    xdot::Float64   = 0.0
    ydot::Float64   = 0.0
end

Base.@kwdef struct FirstOrderUAVDynamics <: UAVDynamics
    timestep::Float64
    noise::Distribution # Calling rand should return a 2-length vector
    params::UAVParameters
end

# Will be called by MDP etc.
function get_uav_control_actions(dynamics::FirstOrderUAVDynamics)

    fo_actions = FirstOrderUAVAction[]
    vel_vals = range(-dynamics.params.XYDOT_LIM, dynamics.params.XYDOT_LIM, step=dynamics.params.XYDOT_STEP)

    for xdot in vel_vals
        for ydot in vel_vals
            push!(fo_actions, FirstOrderUAVAction(xdot, ydot))
        end
    end

    return fo_actions
end

# Samples independently
function generate_start_state(dynamics::FirstOrderUAVDynamics, rng::AbstractRNG)
    dist = Distributions.Uniform(-dynamics.params.XY_LIM, dynamics.params.XY_LIM)
    x = rand(rng, dist)
    y = rand(rng, dist)

    return FirstOrderUAVState(x, y)
end

function get_relative_state_to_goal(goal_pos::SVector{2,Float64}, state::FirstOrderUAVState)
    return FirstOrderUAVState(state.x - goal_pos[1], state.y - goal_pos[2])
end

# Per-drone dynamics (when needed)
function next_uav_state(dynamics::FirstOrderUAVDynamics, state::FirstOrderUAVState,
                        action::FirstOrderUAVAction, rng::AbstractRNG)

    noiseval = rand(rng, dynamics.noise)

    xp = clamp(state.x + action.xdot*dynamics.timestep, -dynamics.params.XY_LIM, dynamics.params.XY_LIM)
    yp = clamp(state.y + action.ydot*dynamics.timestep, -dynamics.params.XY_LIM, dynamics.params.XY_LIM)

    return FirstOrderUAVState(xp, yp)
end

# Dynamics cost for 1st-order is just the velocity cost
# Will be scaled by higher-level reward function appropriately
function dynamics_cost(dynamics::FirstOrderUAVDynamics, a::FirstOrderUAVAction)
    return (a.xdot^2 + a.ydot^2)
end