import cv2
import ratslam
import time
import matplotlib.pyplot as plt

# Requested processing FPS
TARGET_FPS = 5
FRAME_INTERVAL = 1.0 / TARGET_FPS
# ==================================================================

time.sleep(60)

def test_camera(camera_index=0):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter("out.avi", fourcc, 10, (640, 480))
    # slam = ratslam.Ratslam()

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    print(f"Camera {camera_index} opened successfully. Press 'q' to quit.")

    loop = 0
    last_time = time.time()

    while True:

        # ---- FPS Throttling ----
        # now = time.time()
        # if now - last_time < FRAME_INTERVAL:
        #     continue
        # last_time = now
        # ------------------------


        # Capture frame-by-frame
        ret, frame = cap.read()
        print(frame.shape)
        if not ret:
            print("Failed to grab frame")
            break

        # Display the resulting frame
        cv2.imshow('Camera Test', frame)
        out.write(frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.imwrite('./final.png', frame)

            # xs = [exp.x_m for exp in slam.experience_map.exps]
            # ys = [exp.y_m for exp in slam.experience_map.exps]

            # plt.figure(figsize=(8, 6))
            # plt.title('EXPERIENCE MAP')
            # plt.plot(xs, ys, 'bo', label='Experiences')
            # plt.plot(slam.experience_map.current_exp.x_m,
            #         slam.experience_map.current_exp.y_m, 'ko', label='Current Experience')

            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.legend()

            # # Save the plot as an image
            # plt.savefig('experience_map.png', dpi=300)  # You can change the filename and dpi
            # plt.close()  # Close the plot to free memory
            print("Exiting...")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # slam.digest(img)

    # Release the camera and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
