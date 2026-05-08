import mujoco
import numpy as np
from dataclasses import dataclass, field
import yaml
from absl import app, flags

#flags
FLAGS = flags.FLAGS
flags.DEFINE_string('yaml_path', '../config/ligaments_max_pitch.yaml', 'Path to the ligament config yaml')
flags.DEFINE_bool('hide_ligament', False, 'Hide the ligaments')
flags.DEFINE_bool('hide_tendon', False, 'Hide the tendons')
flags.DEFINE_bool('hide_relay', False, 'Hide the relay sites')
flags.DEFINE_bool('hide_geom', False, 'Hide the geoms')
flags.DEFINE_bool('hide_background', False, 'Hide the background')



@dataclass
class HipConfig:
    """Configuration parameters for the hip model"""
    model_name: str = "hip_ligamentous_joint_actuated"
    assets_dir: str = "assets"
    pelvis_name: str = "pelvis_0"
    socket_name: str = "socket_dia60_cov180"

    base_link_pos: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.7]))
    frame_size: float = 0.03

    # Socket parameters
    socket_bottom_pos: np.ndarray = field(default_factory=lambda: np.array([0, 0.08, -0.08]))
    socket_bottom_thickness: float = 0.01
    socket_sensor_thickness: float = 0.01
    socket_thickness_min: float = 0.003

    # Femoral head ball parameters
    ball_diameter: float = 0.06
    
    # Ligament parameters
    num_sites: int = 12
    r_lig_origin: float = 0.05
    r_lig_ins: float = 0.025
    r_relay: float = 0.05
    ligament_friction: float = 0.05
    ligament_range: np.ndarray = field(default_factory=lambda: np.array([0, 0.2]))

    #Tendon parameters
    num_tendon_origins: int = 15
    num_tendon_insertions: int = 5
    r_tendon_ins: float = 0.025
    tendon_origin_points: np.ndarray = field(default_factory=lambda: np.array([]))

class LigamentousHipBuilder:
    def __init__(self, yaml_path=None, 
                 hide_ligament=False, 
                 hide_tendon=False, 
                 hide_relay=False, 
                 hide_geom=False, 
                 hide_background=False):
        self.spec = mujoco.MjSpec.from_string(self._arena_xml(hide_background))
        self.config = HipConfig()
        self.hide_ligament = hide_ligament
        self.hide_tendon = hide_tendon
        self.hide_relay = hide_relay
        self.hide_geom = hide_geom
        self.spec.modelname = self.config.model_name
        # Storage for reuse
        self.relay_sites = []
        self.relay_container = None

        self.ligament_data = {}
        if yaml_path:
            with open(yaml_path, 'r') as f:
                self.ligament_data = yaml.safe_load(f)


    def build(self):
        """Main build pipeline"""
        self._set_defaults()
        #self._setup_world()
        
        # Build hierarchy
        base_link = self.spec.worldbody.add_body(name="base_link", pos=self.config.base_link_pos)
        pelvis_frame = self._add_pelvis_frame(base_link)
        socket_bottom = self._add_socket_bottom(pelvis_frame)
        self._add_socket_sensors(socket_bottom)
        self._add_socket_walls(socket_bottom)
        self._add_convex_socket(socket_bottom, free_joint=False)

        sites_origin_body = self._add_origin_container(socket_bottom)
        
        link = self._add_link_parts()
        
        # Procedural generation
        # 1. First, generate relay sites (to be shared)
        self._add_relay_sites(link)
        
        # 2. Then, use them for ligaments
        self._generate_ligaments(sites_origin_body, link)
        
        # 3. (Optional) Use them for motor tendons later
        self._set_tendon_origin_points()
        self._add_motor_tendons(pelvis_frame, link)

        # 4. Set all geoms to have the same alpha if hide_geom is True        
        if self.hide_geom: 
            self._set_all_geoms_alpha(0.25)

        return self.spec

    def _set_defaults(self):
        """Set global defaults for sites and tendons"""
        self.spec.default.site.size = [0.002, 0.002, 0.002]
        self.spec.default.site.rgba = [0.5, 0.5, 0.5, 1]
        self.spec.default.tendon.width = 0.0005
        self.spec.default.tendon.rgba = [0.9, 0.9, 0.9, 0.25]

    def _arena_xml(self, hide_background=False):
        """Generate the arena XML string (floor, walls, lighting) copied from https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb"""
        if hide_background:
            return """
            <mujoco>
            <visual>
                <headlight diffuse=".5 .5 .5" specular="1 1 1"/>
                <global elevation="-10" offwidth="2048" offheight="1536"/>
                <quality shadowsize="8192"/>
            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="10" height="10"/>
                <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.3"/>
            </asset>

            <worldbody>
                <light pos="0 0 3" diffuse="1 1 1" specular="1 1 1"/>
            </worldbody>
            </mujoco>
            """
        else:
            return """
            <mujoco>
            <visual>
                <headlight diffuse=".5 .5 .5" specular="1 1 1"/>
                <global elevation="-10" offwidth="2048" offheight="1536"/>
                <quality shadowsize="8192"/>
            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1=".5 .5 .5" rgb2="0 0 0" width="10" height="10"/>
                <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.3"/>
            </asset>

            <worldbody>
                <geom name="floor" size="5 5 0.01" type="plane" material="groundplane"/>
                <light pos="0 0 3" diffuse="1 1 1" specular="1 1 1"/>
            </worldbody>
            </mujoco>
            """

    def _setup_world(self):
        """Add lighting and floor"""
        world = self.spec.worldbody
        world.add_light(diffuse=[0.5, 0.5, 0.5], ambient=[0.3, 0.3, 0.3], specular=[0, 0, 0], pos=[0, 0, 1.5], dir=[0, 0, -1])

        world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], rgba=[.9, .9, .9, 1])

    def _add_pelvis_frame(self, parent):
        """Create the static pelvis frame structure"""
        frame = parent.add_body(name="pelvis_frame", pos=[0, 0, 0], euler=[0, 0, 0])

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
        """Add the base plate"""
        bottom_plate = parent.add_body(name="socket_bottom", pos=self.config.socket_bottom_pos, euler=[-135, 0, 0])
        bottom_plate.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.005], size=[0.05, 0.05, self.config.socket_bottom_thickness/2], rgba=[.3, .3, .3, 1])
        return bottom_plate

    def _add_socket_walls(self, parent):
        """Add the socket walls to constrain the floating socket"""
        socket_width = 0.09
        space_from_socket = 0.001
        wall_width = 0.1
        wall_thickness = 0.005
        wall_height = 0.038
        wall_size = [wall_thickness/2, wall_width/2, wall_height/2]
        wall_position = [socket_width/2 + space_from_socket+wall_thickness/2, 0, wall_height/2]

        ceiling_width = 0.1
        ceiling_thickness = 0.002
        ceiling_depth = 0.012
        ceiling_size = [ceiling_depth/2, ceiling_width/2, ceiling_thickness/2]
        ceiling_position = [socket_width/2 + space_from_socket+wall_thickness-ceiling_depth/2, 0, wall_height-ceiling_thickness/2]

        for i in range(4):
            angle = 2 * np.pi * i / 4
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
            current_wall_position = np.dot(rot, wall_position)
            current_ceiling_position = np.dot(rot, ceiling_position)
            parent.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=np.dot(rot, wall_position), euler=[0, 0, np.rad2deg(angle)], size=wall_size, rgba=[.3, .3, .3, .1])
            parent.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=np.dot(rot, ceiling_position), euler=[0, 0, np.rad2deg(angle)], size=ceiling_size, rgba=[.3, .3, .3, .1])

        return

    def _add_socket_sensors(self, parent):
        """Add sensors to the socket for joint interaction force"""
        sensors = []
        r = 0.07 /np.sqrt(2)  # Place sensors at the corners of a square around the center
        bottom_thickness = self.config.socket_bottom_thickness
        sensor_thickness = self.config.socket_sensor_thickness
        for i in range(4):
            theta = 2 * np.pi * i / 4 + np.pi / 4  # Offset by 45 degrees for better coverage
            sensors.append(parent.add_body(name=f"socket_bottom_{i}", pos=[r * np.cos(theta), r * np.sin(theta), bottom_thickness + sensor_thickness/2], euler=[0, 0, 0]))
            sensors[-1].add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.01, sensor_thickness/2], rgba=[1, 1, 1, 1])
            sensors[-1].add_site(name=f"force_sensor_{i}", pos=[0, 0, bottom_thickness + sensor_thickness], size=[0.01], rgba=[1, 0, 0, 0])
        return sensors

    def _add_origin_container(self, parent):
        """Virtual body to hold origin sites"""
        return parent.add_body(name="sites_origin", pos=[0, 0, 0.01], euler=[180, 0, 0])

    def _add_convex_socket(self, parent_body, free_joint=False):
        """Load an external MJCF socket and attach it to the parent body"""
        # 1. Load the external spec
        socket_file = f"../{self.config.assets_dir}/{self.config.socket_name}.xml"
        socket_spec = mujoco.MjSpec.from_file(socket_file)
        socket_source = socket_spec.body(f"{self.config.socket_name}")  # Assuming the body in the XML is named 'socket_0'
        
        if free_joint: 
            socket_free_body = self.spec.worldbody.add_body(
                        name="socket_free_link", 
                        pos=[0, self.config.base_link_pos[1] + 0.09414, self.config.base_link_pos[2] - 0.09414], 
                        euler=[-135, 0, 0]
                    )
            socket_free_body.add_joint(name="socket_free_ignore", type=mujoco.mjtJoint.mjJNT_FREE)
            temp_frame = socket_free_body.add_frame(name="socket_attach_frame")
            temp_frame.attach_body(socket_source, 'L_hip_', '')
        else:
            # 2. Get the root body from the external spec
            # (Usually socket_spec.worldbody.bodies[0] is the body defined in the XML)
            socket_pos = parent_body.add_frame(name="socket_frame", pos=[0, 0, 0.02], euler=[0, 0, 0])
            
            # 3. Create a new body in our current spec and copy EVERYTHING from the source
            # This will copy geoms, child bodies, sites, etc.
            socket_pos.attach_body(socket_source, 'L_hip_', '')  # Attach the new body to the frame for correct positioning
            
            # Optional: If you want to change its position after copying
            # new_socket_body.pos = [0, 0, 0] 
        
        return

    def _add_link_parts(self):
        """Create the leg/link body with its joint and geoms"""
        r_ten_ins = self.config.r_tendon_ins

        # Calculate position relative to world
        #pos = [0, self.config.base_link_pos[1] + 0.07 + 0.045, self.config.base_link_pos[2] - 0.05 - 0.045]
        theta = -np.pi*3/4
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, np.cos(theta), -np.sin(theta)],
                                      [0, np.sin(theta), np.cos(theta)]])
        pos_from_bottom_to_socket = np.array([0, 0, self.config.socket_bottom_thickness + self.config.socket_sensor_thickness])
        pos_from_socket_to_ball = np.array([0, 0, self.config.socket_thickness_min+self.config.ball_diameter/2]) #self.config.socket_bottom_pos
        pos = self.config.base_link_pos + self.config.socket_bottom_pos + rotation_matrix_x @ (pos_from_bottom_to_socket + pos_from_socket_to_ball)
        
        link = self.spec.worldbody.add_body(name="link", pos=pos)
        link.add_joint(name="ball_joint", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.05)
        

        # Link to replicate femoral inclination
        link_inclined = link.add_body(name="link_inclined", pos=[0, 0, 0], euler=[45, 0, 0])

        # Link geoms
        link_inclined.add_geom(name="ligament_insertion_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
                      pos=[0, 0, -0.05], size=[0.025, 0.005], rgba=[.3, .3, .3, 1])
        link_inclined.add_geom(name="tendon_insertion_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
                      pos=[0, 0, -0.06], size=[r_ten_ins, 0.005], rgba=[.3, .3, .3, 1])
        link_inclined.add_geom(name="sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[self.config.ball_diameter/2], 
                      rgba=[0, .7, .7, 0.5], friction=[0.001, 0.001, 0.001])
        link_inclined.add_geom(name="short_cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
                      fromto=[0, 0, 0, 0, 0, -self.config.ball_diameter], size=[0.015], rgba=[0.7, 0.7, 0.7, 1])

        link_stick = link_inclined.add_body(name="link_stick", pos=[0, 0, -self.config.ball_diameter], euler=[-50, 0, 0])
        link_stick.add_geom(name="long_cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
                      fromto=[0, 0, 0.01, 0, 0, -0.3], size=[0.015], rgba=[0.7, 0.7, 0.7, 1])
        link_stick.add_geom(name="weight", type=mujoco.mjtGeom.mjGEOM_SPHERE, 
                      pos=[0, 0, -0.3], size=[0.08], mass=5, rgba=[.2, .2, .2, 1])

        return link_inclined

    def _add_relay_sites(self, parent_body):
        """Creates relay sites and stores them for multiple uses (ligaments/motors)"""
        # Create a container for organization
        self.relay_container = parent_body.add_body(name="sites_relay")
        num = self.config.num_sites

        alpha_relay = 0.0 if self.hide_relay else 1.0
        
        for i in range(num):
            angle = 2 * np.pi * i / num
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            s_relay = self.relay_container.add_site(
                name=f"relay_{i}", 
                pos=[self.config.r_relay * cos_a, self.config.r_relay * sin_a, 0],
                rgba=[0, 1, 0, alpha_relay] if i != 0 else [0, 0, 1, alpha_relay]
            )
            self.relay_sites.append(s_relay)


    def _generate_ligaments(self, origin_body, link_body):
        """Create sites and ligaments using pre-generated relay sites"""
        ins_container = link_body.add_body(name="sites_ligament_insertion", pos=[0, 0, -0.045])
        num = self.config.num_sites
        
        lig_alpha = 0.0 if self.hide_ligament else 0.5

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
                tendon_name = f"lig_{i}_{j}"
                spatial = self.spec.add_tendon(name=tendon_name, width=0.0005, rgba=[1, 1, 1, lig_alpha])
                spatial.wrap_site(f"lig_origin_{i}")
                # Reusing the relay site name (or object)
                spatial.wrap_geom("sphere", f"relay_{i}") 
                spatial.wrap_site(f"lig_insertion_{j}")
                
                spatial.frictionloss = self.config.ligament_friction
                spatial.limited = True

                ref_length = self.ligament_data.get(tendon_name, self.config.ligament_range[1])                
                # spatial.range = [0, ref_length * 1.05]  # Allow some stretch beyond rest length
                spatial.range = self.config.ligament_range
                

    def _set_tendon_origin_points(self):
        fs = self.config.frame_size
        r_off = 0.03
        self.config.tendon_origin_points = np.array([[0.05+fs/2, 0.05, -0.11], [0.05+fs/2, 0.0825, -0.0775], [0.05+fs/2, 0.115, -0.045], 
                              [0.03, 0.13+fs, -fs], [0, 0.13+fs, -fs], [-0.03, 0.13+fs, -fs], 
                              [-0.05-fs/2, 0.115+r_off, -0.045-r_off], [-0.05-fs/2, 0.0825+r_off, -0.0775-r_off], [-0.05-fs/2, 0.05+r_off, -0.11-r_off], 
                              [-0.03, fs, -0.13-fs], [0, fs, -0.13-fs], [0.03, fs, -0.13-fs], 
                              [0.05+fs, 0.05, fs], [0.05+fs, 0.0825, fs], [0.05+fs, 0.115, fs]])
    

    def _add_motor_tendons(self, origin_body, link_body):
        """Create sites and tendons using pre-generated relay sites"""
        ins_container = link_body.add_body(name="sites_tendon_insertion", pos=[0, 0, -0.06])
        num_origins = self.config.num_tendon_origins
        num_insertions = self.config.num_tendon_insertions - 1
        
        tendon_alpha = 0.0 if self.hide_tendon else 0.5

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

        ins_container.add_site(name=f"tendon_insertion_{num_insertions}", 
                                pos=[0.015, -0.045, -0.03],
                                rgba=[1, 0, 0, 1])

        # Tendon connection logic
        for i in range(num_origins):
            for offset in [0]:
                if i < num_origins-3:
                    j = (i//3 + offset) % num_insertions  # Connect each origin to 3 insertions, using integer division for grouping
                    k = int(i/3*3)
                else:
                    j = num_insertions  # Connect remaining origins to the last insertion
                    k = 0  # Use the first relay site for the last few tendons

                tendon_name = f"tendon_{i}_{j}"
                spatial = self.spec.add_tendon(name=tendon_name, width=0.002, rgba=[1, 0, 0, tendon_alpha])
                spatial.wrap_site(f"tendon_origin_{i}")
                # Reusing the relay site name (or object)
                spatial.wrap_geom("sphere", f"relay_{k}") 
                spatial.wrap_site(f"tendon_insertion_{j}")

                # Adding motor actuation to the tendon
                actuator_name = f"motor_{tendon_name}"
                motor = self.spec.add_actuator(name=actuator_name)
                motor.set_to_motor()
                motor.trntype = mujoco.mjtTrn.mjTRN_TENDON
                motor.target = tendon_name
                
                motor.gear = [1, 0, 0, 0, 0, 0] # motor gear="1"
                motor.ctrllimited = True
                motor.ctrlrange = [-2000, 0]

    def _set_all_geoms_alpha(self, alpha: float):
            """
            Iterate through all bodies and update the alpha channel of all geoms.
            RGB values remain unchanged.
            """
            # derive all geoms recursively and update their rgba
            for body in self.spec.bodies:
                for geom in body.geoms:
                    if geom.rgba is not None:
                        current_rgba = list(geom.rgba)
                        current_rgba[3] = alpha # Update only the alpha channel
                        geom.rgba = current_rgba


def main(argv):
    del argv  # Unused

    # --- Execution ---
    builder = LigamentousHipBuilder(
        yaml_path=FLAGS.yaml_path, 
        hide_ligament=FLAGS.hide_ligament, 
        hide_tendon=FLAGS.hide_tendon, 
        hide_relay=FLAGS.hide_relay, 
        hide_geom=FLAGS.hide_geom, 
        hide_background=FLAGS.hide_background
        )
    spec = builder.build()
    spec.compiler.meshdir = f"{builder.config.assets_dir}"  # Set the mesh directory for the compiler
    model = spec.compile()

    # Verification
    print(spec.to_xml())


if __name__ == "__main__":
    app.run(main)