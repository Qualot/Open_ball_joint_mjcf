import mujoco
import numpy as np

# 1. Spec object initialization
spec = mujoco.MjSpec()
spec.modelname = "ball_joint_ligaments_all_to_all"

# 2. Global settings
# spec.compiler.angle = "degree"
# spec.compiler.coordinate = "local"

spec.default.site.size = [0.002, 0.002, 0.002]
spec.default.site.rgba = [1, 0, 0, 1]
spec.default.tendon.width = 0.0005
spec.default.tendon.rgba = [0.9, 0.9, 0.9, 0.25]

# 3. Worldbody and lighting
world = spec.worldbody
world.add_light(diffuse=[.5, .5, .5], pos=[0, 0, 1], dir=[0, 0, -1])
world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], rgba=[.9, .9, .9, 1])

# 4. Base Plate
base_plate = world.add_body(name="base_plate", pos=[0, 0, 0.005])
base_plate.add_geom(name="ligament_origin_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
                   size=[0.05, 0.005], rgba=[.3, .3, .3, 1])

sites_origin_body = base_plate.add_body(name="sites_origin", pos=[0, 0, 0.005])

# 5. Link parts
link = base_plate.add_body(name="link", pos=[0, 0, 0.05])
link.add_joint(name="ball_joint", type=mujoco.mjtJoint.mjJNT_BALL, damping=0.05)
link.add_geom(name="ligament_insertion_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
              pos=[0, 0, 0.05], size=[0.025, 0.005], rgba=[.3, .3, .3, 1])
link.add_geom(name="sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.04], rgba=[0, .7, .7, 0.5])
link.add_geom(name="capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, 0.3], size=[0.01], rgba=[0.7, 0.7, 0.7, 1])

sites_ins_body = link.add_body(name="sites_insertion", pos=[0, 0, 0.055])
sites_relay_body = link.add_body(name="sites_relay")

# 6. Site and Tendon procedural generation
num_sites = 12
r_origin = 0.05
r_ins = 0.025

origin_sites = []
ins_sites = []

for i in range(num_sites):
    angle = 2 * np.pi * i / num_sites
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # Origin sites
    s_orig = sites_origin_body.add_site(
        name=f"origin_{i}", 
        pos=[r_origin * cos_a, r_origin * sin_a, 0]
    )
    origin_sites.append(s_orig)
    
    # Insertion sites
    s_ins = sites_ins_body.add_site(
        name=f"ins_{i}", 
        pos=[r_ins * cos_a, r_ins * sin_a, 0]
    )
    ins_sites.append(s_ins)
    
    # Relay sites Green for visualization
    sites_relay_body.add_site(
        name=f"relay_{i}", 
        pos=[r_origin * cos_a, r_origin * sin_a, 0],
        rgba=[0, 1, 0, 1]
    )

# Tendon generation: All-to-All connection
# for i in range(num_sites):
#     for j in range(num_sites):
#         spatial = spec.add_tendon(name=f"lig_{i}_{j}")
#         spatial.wrap_site(f"origin_{i}")
#         spatial.wrap_geom("sphere", f"relay_{i}")
#         spatial.wrap_site(f"ins_{j}")

# Tendon generation: Hall connection
for i in range(num_sites):
    for j in np.arange(-num_sites/4,num_sites/4+1,1): # Connect each origin to 3-4 insertions around the circle
        j = j + i
        if j < 0:
            j = j + num_sites
        if j >= num_sites:
            j = j % num_sites
        j = int(j)
        spatial = spec.add_tendon(name=f"lig_{i}_{j}")
        spatial.wrap_site(f"origin_{i}")
        spatial.wrap_geom("sphere", f"relay_{i}")
        spatial.wrap_site(f"ins_{j}")
        # enable limited range for the tendon
        spatial.limited = True
        # range = [min, max] in meters, representing the length limits of the tendon
        spatial.range = [0, 0.2]


# 7. Compile the model
model = spec.compile()
print(spec.to_xml()) # XML output for verification