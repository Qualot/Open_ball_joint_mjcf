import mujoco
import numpy as np
import os
import sys
from absl import app
from absl import flags
from absl import logging
import mediapy as media 

# --- 1. Define Flags ---
FLAGS = flags.FLAGS
flags.DEFINE_string('xml_path', None, 'Path to the MuJoCo XML model file.')
flags.DEFINE_integer('steps', 100, 'Number of simulation steps.')
flags.DEFINE_integer('interval', 1, 'Logging interval (every N steps).')

flags.mark_flag_as_required('xml_path')

class TendonTester:
    def __init__(self, xml_path):
        """Initializes the MuJoCo model and data."""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            logging.info(f"Model loaded successfully from {xml_path}")
        except Exception as e:
            logging.fatal(f"Failed to initialize MuJoCo: {e}")

        # Cache tendon names for CSV header and reporting
        self.tendon_names = self._get_tendon_names()

    def _get_tendon_names(self):
        """Internal helper to retrieve all tendon names from the model."""
        names = []
        for i in range(self.model.ntendon):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, i)
            names.append(name if name else f"tendon_{i}")
        return names

    def test_forward(self, steps, interval):
        """
        Performs a forward simulation and prints tendon lengths in CSV format.
        This method uses mj_step for physics and mj_forward for property updates.
        """
        # Print CSV Header
        header = "step," + ",".join(self.tendon_names)
        print(header)

        for step in range(steps):
            # Advance simulation
            mujoco.mj_step(self.model, self.data)
            
            if step % interval == 0:
                # Synchronize all derived quantities (like tendon lengths)
                mujoco.mj_forward(self.model, self.data)
                
                # Format current lengths as CSV row
                lengths = [f"{self.data.ten_length[i]:.6f}" for i in range(self.model.ntendon)]
                row = f"{step}," + ",".join(lengths)
                print(row)

    def test_kinematics_trajectory(self, angles, trajectory):
            """
            Calculates tendon lengths for a given joint trajectory without physics.
            Args:
                angles (np.ndarray): An array of joint angles for each frame.
                trajectory (np.ndarray): A 2D array where each row is a qpos vector.
            """
            # Print CSV Header
            header = "frame," + "angle," + ",".join([f"qpos{i}" for i in range(self.model.nq)]) + "," + ",".join(self.tendon_names)
            print(header)

            for frame, qpos in enumerate(trajectory):
                # 1. Set the joint positions directly
                if len(qpos) != self.model.nq:
                    logging.error(f"qpos dimension mismatch at frame {frame}. Expected {self.model.nq}, got {len(qpos)}")
                    continue
                
                self.data.qpos[:] = qpos

                # 2. Compute only kinematic quantities (positions, site/tendon paths)
                # mj_kinematics is sufficient for tendon lengths
                mujoco.mj_kinematics(self.model, self.data)
                
                # 3. Specifically update tendon lengths (depends on site positions)
                # This follows mj_kinematics to resolve wrapping and path lengths
                mujoco.mj_tendon(self.model, self.data)

                # 4. Format and print
                lengths = [f"{self.data.ten_length[i]:.6f}" for i in range(self.model.ntendon)]
                #print(f"{frame}," + f"{qpos}," + ",".join(lengths))
                print(f"{frame}," + f"{angles[frame]:.6f}," + ",".join([f"{x:.6f}" for x in qpos]) + "," + ",".join(lengths))


def generate_pitch_motion(tester, n_frames=200):
    """
    Generate a pitch motion trajectory for testing tendon lengths.
    This is a basic sanity check to see if tendon lengths change as expected.
    """
    n_frames = n_frames
    # qpos ... pox3, quat4
    angles = np.zeros(n_frames)  # for debugging/visualization: store the pitch angle for each frame
    trajectory = np.zeros((n_frames, tester.model.nq))

    # initial qpos is from the model data. 
    # copying from data.qpos 
    # initial_qpos = np.array([0, 0.025, 0.45, 1, 0, 0, 0]) # x, y, z, qw, qx, qy, qz
    initial_qpos = tester.data.qpos # x

    for i in range(n_frames):
        t = i / n_frames * 2 * np.pi  # 0 to 2pi
        
        current_qpos = initial_qpos.copy()
        angle = -1/6 * np.pi - 1/3 * np.pi * np.sin(t - np.pi/6) # -pi/6 to pi/2

        current_qpos[3] = np.cos(angle / 2) # w 
        current_qpos[5] = np.sin(angle / 2) # y
        
        angles[i] = angle
        trajectory[i] = current_qpos

    return angles, trajectory


def generate_roll_motion(tester, n_frames=200):
    """
    Generate a roll motion trajectory for testing tendon lengths.
    """
    # Initialize trajectory array
    angles = np.zeros(n_frames)  # for debugging/visualization: store the pitch angle for each frame
    trajectory = np.zeros((n_frames, tester.model.nq))

    # Start with the current state (initial position)
    initial_qpos = tester.data.qpos.copy()

    for i in range(n_frames):
        # Time variable from 0 to 2*pi
        t = i / n_frames * 2 * np.pi
        
        current_qpos = initial_qpos.copy()
        
        # Define the roll angle (rotation around X-axis)
        # Oscillate between 0 and +90 degrees (pi/2 rad)
        angle = - (np.pi / 4) * np.cos(t) + np.pi / 4

        # Quaternion for rotation around X-axis:
        # q = [cos(theta/2), sin(theta/2), 0, 0] -> [w, x, y, z]
        current_qpos[3] = np.cos(angle / 2)  # w (scalar part)
        current_qpos[4] = np.sin(angle / 2)  # x (vector part - X axis)
        current_qpos[5] = 0                  # y
        current_qpos[6] = 0                  # z

        angles[i] = angle        
        trajectory[i] = current_qpos

    return angles, trajectory


def generate_circumduction_motion(tester, n_frames=200):
    """
    Tests tendon lengths during a circumduction motion.
    The joint axis rotates around a central axis defined by [0, 1, -1].
    """
    angles = np.zeros(n_frames)  # for debugging/visualization: store the pitch angle for each frame
    trajectory = np.zeros((n_frames, tester.model.nq))
    initial_qpos = tester.data.qpos.copy()

    # 1. Define and normalize the central axis of the cone
    # Direction: [0, 1, -1]
    center_axis = np.array([0.0, 1.0, -1.0])
    center_axis /= np.linalg.norm(center_axis)

    # 2. Define the initial vector (downward)
    v_orig = np.array([0.0, 0.0, -1.0])
    
    for i in range(n_frames):
        t = - i / n_frames * 2 * np.pi  # 0 to 2pi
        
        # 4. Rotation around the central axis (q_rot)
        # Rotating by angle 't' around center_axis
        q_rot = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_rot, center_axis, t)

        # 5. Combine rotations: q_final = q_rot * q_offset
        # This performs the rotation around the world-fixed center_axis
        q_mid = np.zeros(4)
        mujoco.mju_mulQuat(q_mid, q_rot, initial_qpos[3:7])  # Apply q_rot to the initial orientation

        v_rotated = np.zeros(3)
        mujoco.mju_rotVecQuat(v_rotated, v_orig, q_rot)

        q_to_front = np.zeros(4)
        q_final = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_to_front, v_rotated, -t)
        mujoco.mju_mulQuat(q_final, q_to_front, q_mid)  # Apply q_to_front to the initial orientation


        current_qpos = initial_qpos.copy()
        # Assume qpos[3:7] is the quaternion (w, x, y, z)
        current_qpos[3:7] = q_final
        
        angles[i] = t
        trajectory[i] = current_qpos

    return angles, trajectory


def save_trajectory_video(tester, trajectory, filename="circumduction.mp4", 
                          fps=30, 
                          distance=1.0, azimuth=180, elevation=0, 
                          lookat=[0, 0, 0.5], render_tendons=False):

    """
    Renders the given trajectory and saves it as an MP4 video file.
    """
    # Create a renderer for the model
    # Note: width and height can be adjusted as needed
    renderer = mujoco.Renderer(tester.model, height=480, width=640)

# --- Add/Modify from here ---
    # Create a camera object
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    
    # Adjust camera parameters
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = lookat
    # ----------------------------

    # 2. Setup Visualization Options
    scene_option = mujoco.MjvOption()
    # Explicitly enable tendon rendering
    # mjtVisFlag.mjVIS_TENDON corresponds to the tendon visibility
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = render_tendons
    
    # Optional: Enable sites or actuators if needed
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_SITE] = True

    frames = []

    print(f"Rendering {len(trajectory)} frames...")

    for qpos in trajectory:
        # 1. Update the robot state
        tester.data.qpos[:] = qpos
        
        # 2. Synchronize kinematics (positions of all bodies/sites)
        mujoco.mj_kinematics(tester.model, tester.data)

        # 3. Specifically update tendon lengths (depends on site positions)
        # This follows mj_kinematics to resolve wrapping and path lengths
        if render_tendons:
            mujoco.mj_tendon(tester.model, tester.data)

        # 4. Update the renderer with the current data
        renderer.update_scene(tester.data, camera=cam, scene_option=scene_option)
        
        # 5. Render the frame and append to the list
        pixels = renderer.render()
        frames.append(pixels)

    # 5. Write the frames to a video file
    media.write_video(filename, frames, fps=fps)
    print(f"Video saved successfully: {filename}")


def main(argv):
    del argv 

    try:
        # 1. Instantiate the tester
        tester = TendonTester(FLAGS.xml_path)
        
        # 2. Run the forward simulation test
        # tester.test_forward(steps=FLAGS.steps, interval=FLAGS.interval)
        
        # 3. Run the kinematics trajectory test
        # angles, traj = generate_pitch_motion(tester, n_frames=200)
        # angles, traj = generate_roll_motion(tester, n_frames=200)
        angles, traj = generate_circumduction_motion(tester, n_frames=200)

        tester.test_kinematics_trajectory(angles, traj)

        # Save from the left
        # save_trajectory_video(tester, traj, filename="left_view.mp4", fps=30, azimuth=-90, render_tendons=False)

        # Save from the front
        # save_trajectory_video(tester, traj, filename="front_view.mp4", fps=30, azimuth=180, render_tendons=False)

    except Exception as e:
        print(f"Error during testing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    FLAGS.logtostderr = True
    app.run(main)