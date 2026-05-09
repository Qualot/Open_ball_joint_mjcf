import cv2
import numpy as np
from PIL import Image

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "video_path",
    "input.mp4",
    "Path to input video",
)

flags.DEFINE_integer(
    "step",
    30,
    "Frame interval for strobe synthesis",
)

flags.DEFINE_string(
    "output_path",
    "strobe.png",
    "Output image path",
)


def main(argv):

    cap = cv2.VideoCapture(FLAGS.video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {FLAGS.video_path}")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # RGBA image
        img = Image.fromarray(frame).convert("RGBA")

        data = np.array(img)

        # white -> transparent
        r = data[:, :, 0]
        g = data[:, :, 1]
        b = data[:, :, 2]

        white = (r > 240) & (g > 240) & (b > 240)

        data[:, :, 3] = np.where(white, 0, 255)

        img = Image.fromarray(data)

        frames.append(img)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames found.")

    # compose strobe image
    base = Image.new(
        "RGBA",
        frames[0].size,
        (255, 255, 255, 255),
    )

    for img in frames[::FLAGS.step]:
        base.alpha_composite(img)

    base.save(FLAGS.output_path)

    print(f"Saved: {FLAGS.output_path}")


if __name__ == "__main__":
    app.run(main)