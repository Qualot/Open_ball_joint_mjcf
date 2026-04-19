import mujoco
import numpy as np
import os
import sys
from absl import app
from absl import flags
from absl import logging

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

    def test_kinematics_trajectory(self, trajectory):
            """
            Calculates tendon lengths for a given joint trajectory without physics.
            Args:
                trajectory (np.ndarray): A 2D array where each row is a qpos vector.
            """
            # Print CSV Header
            header = "frame," + ",".join([f"qpos{i}" for i in range(self.model.nq)]) + "," + ",".join(self.tendon_names)
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
                print(f"{frame}," + ",".join([f"{x:.6f}" for x in qpos]) + "," + ",".join(lengths))


def test_pitch_motion(tester, n_frames=200):
    """
    Test tendon lengths during a simple pitch motion of the joint.
    This is a basic sanity check to see if tendon lengths change as expected.
    """
    n_frames = n_frames
    # qpos ... pox3, quat4
    trajectory = np.zeros((n_frames, tester.model.nq))

    # initial qpos is from the model data. 
    # copying from data.qpos 
    # initial_qpos = np.array([0, 0.025, 0.45, 1, 0, 0, 0]) # x, y, z, qw, qx, qy, qz
    initial_qpos = tester.data.qpos # x

    for i in range(n_frames):
        t = i / n_frames * 2 * np.pi  # 0 to 2pi
        
        current_qpos = initial_qpos.copy()
        angle = 1/3 * np.pi + 1/6 * np.pi * np.sin(t) # -pi/6 to pi/2

        current_qpos[3] = np.cos(angle / 2) # w 
        current_qpos[5] = np.sin(angle / 2) # y
        
        trajectory[i] = current_qpos

    qpos_trajectory = trajectory
    tester.test_kinematics_trajectory(qpos_trajectory)


def main(argv):
    del argv 

    try:
        # 1. Instantiate the tester
        tester = TendonTester(FLAGS.xml_path)
        
        # 2. Run the forward simulation test
        # tester.test_forward(steps=FLAGS.steps, interval=FLAGS.interval)
        
        # 3. Run the kinematics trajectory test
        test_pitch_motion(tester, n_frames=200)


    except Exception as e:
        print(f"Error during testing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    FLAGS.logtostderr = True
    app.run(main)