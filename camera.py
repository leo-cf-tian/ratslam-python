import cv2
import numpy as np
from matplotlib import pyplot as plot
import mpl_toolkits.mplot3d.axes3d as p3

import ratslam
import time

# ========================== Webcam Setup ==========================
# 0 = default webcam. Change if needed.
video = cv2.VideoCapture(0)

# Requested processing FPS
TARGET_FPS = 1
FRAME_INTERVAL = 1.0 / TARGET_FPS
# ==================================================================

slam = ratslam.Ratslam()

loop = 0
last_time = time.time()

while True:

    # ---- FPS Throttling ----
    # now = time.time()
    # if now - last_time < FRAME_INTERVAL:
    #     continue
    # last_time = now
    # ------------------------

    loop += 1
    ret, frame = video.read()
    if not ret:
        print("Camera disconnected or error")
        break

    # Convert to grayscale for SLAM
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # slam.digest(img)
    cv2.imshow("Camera", frame)

    # # ========================= PLOTTING ============================
    # b, g, r = cv2.split(frame)
    # rgb_frame = cv2.merge([r, g, b])

    # plot.clf()

    # # ---------- RAW IMAGE ----------
    # ax = plot.subplot(2, 2, 1)
    # plot.title('RAW IMAGE')
    # plot.imshow(rgb_frame)
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])

    # # ---------- RAW ODOMETRY ----------
    # plot.subplot(2, 2, 2)
    # plot.title('RAW ODOMETRY')
    # plot.plot(slam.odometry[0], slam.odometry[1])
    # plot.plot(slam.odometry[0][-1], slam.odometry[1][-1], 'ko')

    # # ---------- POSE CELL ACTIVATION (3D) ----------
    # ax = plot.subplot(2, 2, 3, projection='3d')
    # plot.title('POSE CELL ACTIVATION')
    # x, y, th = slam.pc
    # ax.plot(x, y, 'x')
    # ax.plot3D([0, 60], [y[-1], y[-1]], [th[-1], th[-1]], color='k')
    # ax.plot3D([x[-1], x[-1]], [0, 60], [th[-1], th[-1]], color='k')
    # ax.plot3D([x[-1], x[-1]], [y[-1], y[-1]], [0, 36], color='k')
    # ax.plot3D([x[-1]], [y[-1]], [th[-1]], color='m', marker='o')
    # ax.grid()
    # ax.set_xlim(0, 60)
    # ax.set_ylim(0, 60)
    # ax.set_zlim(0, 36)

    # # ---------- EXPERIENCE MAP ----------
    # plot.subplot(2, 2, 4)
    # plot.title('EXPERIENCE MAP')
    # xs = [exp.x_m for exp in slam.experience_map.exps]
    # ys = [exp.y_m for exp in slam.experience_map.exps]
    # plot.plot(xs, ys, 'bo')
    # plot.plot(slam.experience_map.current_exp.x_m,
    #           slam.experience_map.current_exp.y_m, 'ko')

    # plot.tight_layout()
    # plot.pause(0.01)
    # # ===============================================================

print('DONE!')
print('n_templates:', len(slam.view_cells.cells))
print('n_experiences:', len(slam.experience_map.exps))

plot.show()
