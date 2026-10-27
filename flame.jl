# Real-time 2D flow past a draggable cylinder and two fixed cylinders,
# with a live chart of the forces on the moving cylinder.
#
# Run with several threads for speed:   julia -t auto realtime_cylinder.jl
# Packages needed (once):  ] add WaterLily GLMakie StaticArrays
#
# Everything is wrapped in a module, so the script can be re-run in the same
# Julia session without "redefinition of constant" errors.

module FlowDemo

using WaterLily, GLMakie, StaticArrays

# ---------------- settings ----------------
const n, m = 256, 128          # grid cells (x, y)  - lower for more speed
const R    = m / 16f0          # cylinder radius (cells)
const D    = 2R                # diameter
const U    = 1f0               # free-stream velocity
const Re   = 200               # Reynolds number based on radius
const steps_per_frame = 3      # more = faster flow, choppier drag
const vmax  = 2f0 * U          # speed limit for the moving cylinder
const nhist = 400              # number of points kept in the force chart

# ---------------- moving body ----------------
# The centre moves with a piecewise-constant velocity so WaterLily
# (via ForwardDiff on the map) knows the body velocity -> correct boundary condition.
const pos0   = Ref(SA[m/2f0, m/2f0])   # centre at time t0
const vel    = Ref(SA[0f0, 0f0])       # current body velocity
const t0     = Ref(0f0)
const target = Ref(pos0[])             # where the mouse wants it

# ---------------- fixed bodies ----------------
const fixed_pos = (SA[n/2f0, m/2f0],   # add or move centres here
                   SA[n/2f0, m/3f0])
const rc = 2R + 2f0                    # closest allowed distance between centres

cyl_sdf(x, t)  = sqrt(sum(abs2, x)) - R
cyl_move(x, t) = x - (pos0[] + vel[] * (t - t0[]))

body1 = AutoBody(cyl_sdf, cyl_move)                         # moving
body2 = AutoBody(cyl_sdf, (x, t) -> x - fixed_pos[1])       # fixed
body3 = AutoBody(cyl_sdf, (x, t) -> x - fixed_pos[2])       # fixed

sim = Simulation((n, m), (U, 0f0), R; ν = U * R / Re,
                 body = body1 + body2 + body3, T = Float32)

# ---------------- collision handling ----------------
# keep the target point outside the fixed cylinders
function push_out(q)
    for p in fixed_pos
        d = q - p
        r = sqrt(sum(abs2, d))
        r < rc && (q = p + d * (rc / max(r, 1f-3)))
    end
    return q
end

# stop the moving cylinder from passing through on its way
function avoid(c, v, dt)
    for p in fixed_pos
        d   = c - p
        r   = sqrt(sum(abs2, d))
        nrm = d / max(r, 1f-3)               # unit vector away from the fixed cylinder
        vn  = sum(v .* nrm)                  # velocity component along that direction
        if vn < 0 && r + vn * dt < rc        # would touch it during this frame
            v = v - vn * nrm                 # slide along the surface
        end
        r < rc && (v = v + 0.5f0 * (rc - r) / dt * nrm)   # push out if too close
    end
    return v
end

# ---------------- forces on the moving cylinder ----------------
function force_coeffs(sim, body)
    F = -(WaterLily.pressure_force(sim.flow, body) .+
        WaterLily.viscous_force(sim.flow, body) ) # pressure + friction, per unit span
    q = 0.5f0 * sim.U^2 * D                      # ½ ρ U² D  (ρ = 1)
    return Float32(F[1] / q), Float32(sqrt(sum(abs2, F)) / q)  # drag C_D, total |C_F|
end

# ---------------- vorticity field ----------------
ω = zeros(Float32, size(sim.flow.p))
function vorticity!(ω, sim)
    s = sim.L / sim.U
    @inside ω[I] = WaterLily.curl(3, I, sim.flow.u) * s
    return ω
end

# ---------------- figure: flow ----------------
fig = Figure(size = (1100, 820))
ax  = Axis(fig[1, 1], aspect = DataAspect(),
           title = "Drag the cylinder with the left mouse button")
ωobs = Observable(copy(ω))
hm = heatmap!(ax, ωobs, colormap = :seismic, colorrange = (-5, 5))
translate!(hm, -1.5f0, -1.5f0, 0)      # cell I sits at I - 1.5 in WaterLily coordinates

cobs = Observable(Point2f(pos0[]...))
poly!(ax, lift(c -> Circle(c, R), cobs), color = :black)
for p in fixed_pos
    poly!(ax, Circle(Point2f(p...), R), color = :gray20)
end
limits!(ax, 0, n, 0, m)
deregister_interaction!(ax, :rectanglezoom)   # so dragging doesn't zoom

# ---------------- figure: force chart ----------------
ax2 = Axis(fig[2, 1], xlabel = "time  tU/R", ylabel = "coefficient",
           title = "Forces on the moving cylinder")
rowsize!(fig.layout, 2, Relative(0.3))

th, cdh, cfh = Float32[], Float32[], Float32[]   # history buffers
cd_obs = Observable([Point2f(0, 0)])
cf_obs = Observable([Point2f(0, 0)])
lines!(ax2, cd_obs, color = :dodgerblue, label = "drag  C_D")
lines!(ax2, cf_obs, color = :orangered,  label = "total  |C_F|")
axislegend(ax2, position = :lt)

# ---------------- mouse ----------------
function set_target!()
    p = mouseposition(ax.scene)
    q = SA[clamp(Float32(p[1]), R + 2, n - R - 2),
           clamp(Float32(p[2]), R + 2, m - R - 2)]
    target[] = push_out(q)
end

on(events(ax.scene).mousebutton) do ev            # click
    (ev.button == Mouse.left && ev.action == Mouse.press) && set_target!()
    return Consume(false)
end
on(events(ax.scene).mouseposition) do _           # drag
    ispressed(ax.scene, Mouse.left) && set_target!()
    return Consume(false)
end

display(fig)

# ---------------- main loop ----------------
function run!()
    while isopen(fig.scene)
        t = Float32(WaterLily.time(sim))
        c = pos0[] + vel[] * (t - t0[])            # where the body is now
        v = 0.05f0 * (target[] - c)                # steer towards the mouse
        s = sqrt(sum(abs2, v))
        s > vmax && (v = v * (vmax / s))
        v = avoid(c, v, steps_per_frame * sim.flow.Δt[end])
        pos0[], vel[], t0[] = c, v, t

        for _ in 1:steps_per_frame
            sim_step!(sim; remeasure = true)
        end

        # flow field
        vorticity!(ω, sim)
        ωobs[] = copy(ω)
        cobs[] = Point2f(c...)

        # forces
        cd, cf = force_coeffs(sim, body1)
        push!(th, Float32(WaterLily.sim_time(sim))); push!(cdh, cd); push!(cfh, cf)
        if length(th) > nhist
            popfirst!(th); popfirst!(cdh); popfirst!(cfh)
        end
        cd_obs[] = Point2f.(th, cdh)               # new arrays, so Makie redraws
        cf_obs[] = Point2f.(th, cfh)
        autolimits!(ax2)

        sleep(0.001)                               # let the window process events
    end
end

run!()

end # module