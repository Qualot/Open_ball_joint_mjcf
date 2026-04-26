import mujoco
import numpy as np
from dataclasses import dataclass, field

@dataclass
class HipConfig:
    """Configuration parameters for the hip model"""
    model_name: str = "hip_ligamentous_joint_actuated"
    assets_dir: str = "assets"
    pelvis_name: str = "pelvis_0"
    socket_name: str = "socket_0"

    base_link_pos: list = field(default_factory=lambda: [0, 0, 0.7])
    frame_size: float = 0.03
    
    # Ligament parameters
    num_sites: int = 12
    r_lig_origin: float = 0.05
    r_lig_ins: float = 0.025
    r_relay: float = 0.05
    ligament_friction: float = 0.05
    ligament_range: list = field(default_factory=lambda: [0, 0.2])

    #Tendon parameters
    num_tendon_origins: int = 8
    num_tendon_insertions: int = 4
    r_tendon_ins: float = 0.025
    tendon_origin_points = np.array([[0.05, 0.05, -0.11], [0.05, 0.115, -0.045], 
                              [0.03, 0.13+frame_size, -frame_size], [-0.03, 0.13+frame_size, -frame_size], 
                              [-0.05, 0.115, -0.045], [-0.05, 0.05, -0.11], 
                              [-0.03, frame_size, -0.13-frame_size], [0.03, frame_size, -0.13-frame_size]])


class LigamentousHipBuilder:
    def __init__(self):
        self.spec = mujoco.MjSpec()
        self.config = HipConfig()
        self.spec.modelname = self.config.model_name
        # Storage for reuse
        self.relay_sites = []
        self.relay_container = None
        
    def build(self):
        """Main build pipeline"""
        self._set_defaults()
        self._setup_world()
        
        # Build hierarchy
        base_link = self.spec.worldbody.add_body(name="base_link", pos=self.config.base_link_pos)
        pelvis_frame = self._add_pelvis_frame(base_link)
        socket_bottom = self._add_socket_bottom(pelvis_frame)
        self._add_convex_socket(socket_bottom)

        sites_origin_body = self._add_origin_container(socket_bottom)
        
        link = self._add_link_parts()
        
        # Procedural generation
        # 1. First, generate relay sites (to be shared)
        self._add_relay_sites(link)
        
        # 2. Then, use them for ligaments
        self._generate_ligaments(sites_origin_body, link)
        
        # 3. (Optional) Use them for motor tendons later
        self._add_motor_tendons(pelvis_frame, link)
        return self.spec

    def _set_defaults(self):
        """Set global defaults for sites and tendons"""
        self.spec.default.site.size = [0.002, 0.002, 0.002]
        self.spec.default.site.rgba = [0.5, 0.5, 0.5, 1]
        self.spec.default.tendon.width = 0.0005
        self.spec.default.tendon.rgba = [0.9, 0.9, 0.9, 0.25]

    def _setup_world(self):
        """Add lighting and floor"""
        world = self.spec.worldbody
        world.add_light(diffuse=[.5, .5, .5], pos=[0, 0, 1], dir=[90, 0, -1])
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], rgba=[.9, .9, .9, 1])

    def _add_pelvis_frame(self, parent):
        """Create the static pelvis frame structure"""
        frame = parent.add_body(name="pelvis_frame", pos=[0, 0, 0], euler=[0, 0, 0])
        # Add frame geoms (boxes)
        f_size = self.config.frame_size
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.05-f_size/2, f_size/2, -0.1/2-f_size/2], size=[f_size/2, f_size/2, 0.1/2], rgba=[.5, .5, .5, 1])
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.05+f_size/2, f_size/2, -0.1/2-f_size/2], size=[f_size/2, f_size/2, 0.1/2], rgba=[.5, .5, .5, 1])
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.05-f_size/2, 0.13/2, 0.0], size=[f_size/2, 0.13/2, f_size/2], rgba=[.5, .5, .5, 1])
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.05+f_size/2, 0.13/2, 0.0], size=[f_size/2, 0.13/2, f_size/2], rgba=[.5, .5, .5, 1])
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.13+f_size/2, 0.0], size=[0.05, f_size/2, f_size/2], rgba=[.5, .5, .5, 1])
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, f_size/2, -0.1-f_size/2], size=[0.05, f_size/2, f_size/2], rgba=[.5, .5, .5, 1])

        # 1. Load the external spec
        pelvis_file = f"../{self.config.assets_dir}/{self.config.pelvis_name}.xml"
        pelvis_spec = mujoco.MjSpec.from_file(pelvis_file)
        
        # 2. Get the root body from the external spec
        pelvis_source = pelvis_spec.body(f"{self.config.pelvis_name}")  # Assuming the body in the XML is named 'pelvis_0'
        pelvis_pos = parent.add_frame(name="pelvis_frame", pos=[0, 0, 0], euler=[0, 0, 0])
        
        # 3. Create a new body in our current spec and copy EVERYTHING from the source
        # This will copy geoms, child bodies, sites, etc.
        pelvis_pos.attach_body(pelvis_source, 'L_pelvis_', '')  # Attach the new body to the frame for correct positioning
        
        # Optional: If you want to change its position after copying
        # new_pelvis_body.pos = [0, 0, 0] 

        return frame

    def _add_socket_bottom(self, parent):
        """Add the base plate and ligament origin cylinder"""
        base = parent.add_body(name="socket_bottom", pos=[0, 0.08, -0.08], euler=[-135, 0, 0])
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.005], size=[0.05, 0.05, 0.005], rgba=[.3, .3, .3, 1])
        return base

    def _add_origin_container(self, parent):
        """Virtual body to hold origin sites"""
        return parent.add_body(name="sites_origin", pos=[0, 0, 0.01], euler=[180, 0, 0])

    def _add_convex_socket(self, parent_body):
        """Load an external MJCF socket and attach it to the parent body"""
        # 1. Load the external spec
        socket_file = f"../{self.config.assets_dir}/{self.config.socket_name}.xml"
        socket_spec = mujoco.MjSpec.from_file(socket_file)
        
        # 2. Get the root body from the external spec
        # (Usually socket_spec.worldbody.bodies[0] is the body defined in the XML)
        socket_source = socket_spec.body(f"{self.config.socket_name}")  # Assuming the body in the XML is named 'socket_0'
        socket_pos = parent_body.add_frame(name="socket_frame", pos=[0, 0, 0.02], euler=[0, 0, 0])
        
        # 3. Create a new body in our current spec and copy EVERYTHING from the source
        # This will copy geoms, child bodies, sites, etc.
        socket_pos.attach_body(socket_source, 'L_hip_', '')  # Attach the new body to the frame for correct positioning
        
        # Optional: If you want to change its position after copying
        # new_socket_body.pos = [0, 0, 0] 
        
        return socket_pos

    def _add_link_parts(self):
        """Create the leg/link body with its joint and geoms"""
        r_ten_ins = self.config.r_tendon_ins

        # Calculate position relative to world
        #pos = [0, self.config.base_link_pos[1] + 0.07 + 0.045, self.config.base_link_pos[2] - 0.05 - 0.045]
        pos = [0, self.config.base_link_pos[1] + 0.12455, self.config.base_link_pos[2] - 0.12455]
        
        link = self.spec.worldbody.add_body(name="link", pos=pos)
        link.add_joint(name="ball_joint", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.05)
        
        # Link geoms
        link.add_geom(name="ligament_insertion_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
                      pos=[0, 0, -0.05], size=[0.025, 0.005], rgba=[.3, .3, .3, 1])
        link.add_geom(name="tendon_insertion_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
                      pos=[0, 0, -0.1], size=[r_ten_ins, 0.005], rgba=[.3, .3, .3, 1])
        link.add_geom(name="sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.04], 
                      rgba=[0, .7, .7, 0.5], friction=[0.001, 0.001, 0.001])
        link.add_geom(name="cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
                      fromto=[0, 0, 0, 0, 0, -0.3], size=[0.01], rgba=[0.7, 0.7, 0.7, 1])
        link.add_geom(name="weight", type=mujoco.mjtGeom.mjGEOM_SPHERE, 
                      pos=[0, 0, -0.3], size=[0.08], mass=5, rgba=[.2, .2, .2, 1])
        return link

    def _add_relay_sites(self, parent_body):
        """Creates relay sites and stores them for multiple uses (ligaments/motors)"""
        # Create a container for organization
        self.relay_container = parent_body.add_body(name="sites_relay")
        num = self.config.num_sites
        
        for i in range(num):
            angle = 2 * np.pi * i / num
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            s_relay = self.relay_container.add_site(
                name=f"relay_{i}", 
                pos=[self.config.r_relay * cos_a, self.config.r_relay * sin_a, 0],
                rgba=[0, 1, 0, 1] if i != 0 else [0, 0, 1, 1]
            )
            self.relay_sites.append(s_relay)


    def _generate_ligaments(self, origin_body, link_body):
        """Create sites and ligaments using pre-generated relay sites"""
        ins_container = link_body.add_body(name="sites_ligament_insertion", pos=[0, 0, -0.045])
        num = self.config.num_sites
        
        for i in range(num):
            angle = 2 * np.pi * i / num
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Origin sites
            origin_body.add_site(name=f"lig_origin_{i}", 
                                pos=[self.config.r_lig_origin * cos_a, self.config.r_lig_origin * sin_a, 0],
                                rgba=[0.5, 0.5, 0.5, 1] if i != 0 else [0, 0, 1, 1]
                                )
            
            # Insertion sites
            ins_container.add_site(name=f"lig_insertion_{i}", 
                                   pos=[self.config.r_lig_ins * cos_a, self.config.r_lig_ins * sin_a, 0],
                                   rgba=[0.5, 0.5, 0.5, 1] if i != 0 else [0, 0, 1, 1])

        # Tendon connection logic
        for i in range(num):
            for offset in [-1, 0, 1]:
                j = (i + offset) % num
                spatial = self.spec.add_tendon(name=f"lig_{i}_{j}")
                spatial.wrap_site(f"lig_origin_{i}")
                # Reusing the relay site name (or object)
                spatial.wrap_geom("sphere", f"relay_{i}") 
                spatial.wrap_site(f"lig_insertion_{j}")
                
                spatial.frictionloss = self.config.ligament_friction
                spatial.limited = True
                spatial.range = self.config.ligament_range

    def _add_motor_tendons(self, origin_body, link_body):
        """Create sites and tendons using pre-generated relay sites"""
        ins_container = link_body.add_body(name="sites_tendon_insertion", pos=[0, 0, -0.1])
        num_origins = self.config.num_tendon_origins
        num_insertions = self.config.num_tendon_insertions
        
        for i in range(num_origins):
            angle = 2 * np.pi * i / num_origins
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Origin sites
            origin_body.add_site(name=f"tendon_origin_{i}", 
                                pos=self.config.tendon_origin_points[i],
                                rgba=[1, 0, 0, 1] if i != 0 else [0, 0, 1, 1]
                                )
            
        for i in range(num_insertions):
            angle = 2 * np.pi * i / num_insertions
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Insertion sites
            ins_container.add_site(name=f"tendon_insertion_{i}", 
                                   pos=[self.config.r_tendon_ins * cos_a, self.config.r_tendon_ins * sin_a, 0],
                                   rgba=[1, 0, 0, 1] if i != 0 else [0, 0, 1, 1])

        # Tendon connection logic
        for i in range(num_origins):
            j = i//2
            k = int(i/2*3)
            spatial = self.spec.add_tendon(name=f"tendon_{i}_{j}", width=0.002, rgba=[1, 0, 0, 0.5])
            spatial.wrap_site(f"tendon_origin_{i}")
            # Reusing the relay site name (or object)
            spatial.wrap_geom("sphere", f"relay_{k}") 
            spatial.wrap_site(f"tendon_insertion_{j}")

def main():
    # --- Execution ---
    builder = LigamentousHipBuilder()
    spec = builder.build()
    spec.compiler.meshdir = f"{builder.config.assets_dir}"  # Set the mesh directory for the compiler
    model = spec.compile()

    # Verification
    print(spec.to_xml())


if __name__ == "__main__":
    main()