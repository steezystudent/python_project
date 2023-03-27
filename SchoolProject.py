import cv2
import numpy as np
import keyboard
from imutils.video import FPS
import time

################
WORK IN PROGRESS
################


print("press 'r' to run the program 'q' to exit\n")

while True:
    if keyboard.is_pressed('r'):
        print('\n[INFO] running, please wait\n')
        print('\n[INFO] Versions installed:')

        while True:
            if keyboard.is_pressed('q'):
                print('\n[INFO] Program terminated')
                break  

            print('[INFO] cv2:', cv2.__version__)
            print('[INFO] numpy:', np.__version__)

            fps = FPS().start()


            def detect_apples(frame):
                ################################################################
                # TODO get frame and how big user wants it to be

                frame_height = #input
                frame_width = #input
                
                #################################################################
                # Convert to HSV color space
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Define color range for red apples
                lower_red = np.array([40, 100, 100])
                upper_red = np.array([10, 255, 255])

                # Define color range for green apples
                lower_green = np.array([0, 50, 50])
                upper_green = np.array([40, 215, 96])

                # Threshold the HSV image to get only red and green colors
                red_mask = cv2.inRange(hsv, lower_red, upper_red)
                green_mask = cv2.inRange(hsv, lower_green, upper_green)

                # Find contours in the masks
                red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw bounding boxes around detected red and green objects
                for cnt in red_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                for cnt in green_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                return frame


            # Initialize the video feed object
            camera_indexes = []
            for i in range(10):
                VideoFeed = cv2.VideoCapture(i)
                if not VideoFeed.isOpened():
                    break
                camera_indexes.append(i)
                VideoFeed.release()

            # Print the list of available cameras
            print("[INFO] Loading available cameras:")
            for i in camera_indexes:
                if camera_indexes is None:
                    print("Error! no Cameras available :( ")
                else:
                    print(f"{i}: {cv2.VideoCapture(i).getBackendName()}")

            VideoFeed = cv2.VideoCapture(0 + cv2.CAP_DSHOW)


            while True:
                # Read a frame from video feed for the object
                ret, frame = VideoFeed.read()

                if not ret:
                    break

                # Detect the apples in the frame
                frame_with_apples = detect_apples(frame)

                # Show the frame with apples
                cv2.imshow('Video Feed', frame_with_apples)

                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO]Program Terminated")
                    break

                fps.update()

            # stop the timer and display FPS information
            fps.stop()

            print("[After action report] elapsed time: {:.2f}".format(fps.elapsed()))
            print("[After action report] FPS: {:.2f}".format(fps.fps()))

            # do a bit of cleanup
            cv2.destroyAllWindows()

            # close windows
            VideoFeed.release()
            cv2.destroyAllWindows()
            break

    elif keyboard.is_pressed('q'):
        print('\n[INFO]Program Terminated')
        break
