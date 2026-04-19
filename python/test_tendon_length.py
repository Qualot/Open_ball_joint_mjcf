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

def main(argv):
    del argv 

    # --- 2. Validation & Model Loading ---
    if not os.path.exists(FLAGS.xml_path):
        # Using stderr for errors so they don't corrupt the redirected CSV file
        print(f"Error: File '{FLAGS.xml_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        model = mujoco.MjModel.from_xml_path(FLAGS.xml_path)
        data = mujoco.MjData(model)
        
        # Log info to stderr to keep stdout clean for CSV data
        logging.info(f"Loaded: {FLAGS.xml_path}")
    except Exception as e:
        logging.fatal(f"Error loading MuJoCo model: {e}")

    # --- 3. CSV Header Preparation ---
    # Get all tendon names
    tendon_names = []
    for i in range(model.ntendon):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        tendon_names.append(name if name else f"tendon_{i}")

    # Print CSV Header: step, tendon_name1, tendon_name2, ...
    header = "step," + ",".join(tendon_names)
    print(header)

    # --- 4. Execution & CSV Data Output ---
    for step in range(FLAGS.steps):
        mujoco.mj_step(model, data)
        
        if step % FLAGS.interval == 0:
            # Update physics for accurate length calculation
            mujoco.mj_forward(model, data)
            
            # Collect all lengths
            lengths = [f"{data.ten_length[i]:.6f}" for i in range(model.ntendon)]
            
            # Print CSV Row: current_step, len1, len2, ...
            row = f"{step}," + ",".join(lengths)
            print(row)

if __name__ == "__main__":
    # Setting logtostderr=True ensures that logging info goes to stderr, 
    # which won't be captured when you redirect stdout to a CSV file.
    FLAGS.logtostderr = True
    app.run(main)