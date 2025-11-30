import cv2
import numpy as np
from matplotlib import pyplot as plot
import mpl_toolkits.mplot3d.axes3d as p3

import ratslam
import yt_dlp

# ========================== YouTube stream setup ==========================
url = "https://www.youtube.com/watch?v=gpEEuk4xdFk"

ydl_opts = {
    'format': 'best[ext=mp4]/best',
    'quiet': True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    video_url = info['url']

video = cv2.VideoCapture(video_url)
fps = video.get(cv2.CAP_PROP_FPS)  # for time calculation
# ==========================================================================

slam = ratslam.Ratslam()

DATASET_START = 120

loop = 0
_, frame = video.read()
while True:
    loop += 1

    # RUN A RATSLAM ITERATION ==================================
    _, frame = video.read()
    if frame is None:
        print("End of stream or error")
        break
    
    # Skip frames (dataset starts here)
    if loop < DATASET_START:
        continue
    
    # Plot each 10 frames
    if loop % 3 != 0:
        continue

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    slam.digest(img)
    # ==========================================================

    # Plot each 10 frames
    # if loop % 15 != 0:
    #     continue

    # PLOT THE CURRENT RESULTS =================================
    b, g, r = cv2.split(frame)
    rgb_frame = cv2.merge([r, g, b])

    plot.clf()

    # RAW IMAGE -------------------
    ax = plot.subplot(2, 2, 1)
    plot.title('RAW IMAGE')
    plot.imshow(rgb_frame)
    
    frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - DATASET_START
    time_sec = frame_num / fps
    ax.text(10, 20, f'{time_sec}s', color='white', fontsize=12, 
        bbox=dict(facecolor='black', alpha=0.5, pad=2))
    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    # -----------------------------

    # RAW ODOMETRY ----------------
    plot.subplot(2, 2, 2)
    plot.title('RAW ODOMETRY')
    plot.plot(slam.odometry[0], slam.odometry[1])
    plot.plot(slam.odometry[0][-1], slam.odometry[1][-1], 'ko')
    # -----------------------------

    # POSE CELL ACTIVATION --------
    ax = plot.subplot(2, 2, 3, projection='3d')
    plot.title('POSE CELL ACTIVATION')
    x, y, th = slam.pc
    ax.plot(x, y, 'x')
    ax.plot3D([0, 60], [y[-1], y[-1]], [th[-1], th[-1]], color='k')
    ax.plot3D([x[-1], x[-1]], [0, 60], [th[-1], th[-1]], color='k')
    ax.plot3D([x[-1], x[-1]], [y[-1], y[-1]], [0, 36], color='k')
    ax.plot3D([x[-1]], [y[-1]], [th[-1]], color='m', marker='o')
    ax.grid()
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_zlim(0, 36)
    # -----------------------------

    # EXPERIENCE MAP --------------
    plot.subplot(2, 2, 4)
    plot.title('EXPERIENCE MAP')
    xs = [exp.x_m for exp in slam.experience_map.exps]
    ys = [exp.y_m for exp in slam.experience_map.exps]

    plot.plot(xs, ys, 'bo')
    plot.plot(slam.experience_map.current_exp.x_m,
              slam.experience_map.current_exp.y_m, 'ko')
    # -----------------------------

    plot.tight_layout()
    plot.pause(0.1)
    # ==========================================================

print('DONE!')
print(('n_ templates:', len(slam.view_cells.cells)))
print(('n_ experiences:', len(slam.experience_map.exps)))
plot.show()
