import cv2
import numpy as np

def visualize_coordinates(video_path, pixel_interval=100):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    print(video_path)

    # Read one frame
    ret, frame = cap.read()

    # Ensure the frame is successfully read
    if not ret:
        print("Error reading frame from the video.")
        return

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Create a blank image to add text annotations
    annotated_frame = np.zeros_like(frame)

    # Add coordinates as text to the annotated frame
    for y in range(0, height, pixel_interval):
        for x in range(0, width, pixel_interval):
            text = f"({x}, {y})"
            cv2.putText(annotated_frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the original frame and the annotated frame
    cv2.imshow("Original Frame", frame)
    # cv2.imshow("Annotated Frame", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Replace with the path of video file
    video_path = 'data/frame.mp4'

    # Visualize coordinates with a pixel interval of 100
    visualize_coordinates(video_path, pixel_interval=100)

