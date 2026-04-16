"""Quick test to check phone camera connection."""
import cv2
import sys

url = input("Enter phone camera URL (e.g. http://192.168.1.5:8080/video): ").strip()

print(f"Trying to connect to: {url}")
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("FAILED: Could not open stream.")
    print("Try these fixes:")
    print("  1. Make sure phone and laptop are on the SAME WiFi")
    print("  2. Try adding /video or /videofeed to the URL")
    print("  3. Try the MJPEG stream: http://<ip>:8080/video")
    print("  4. Disable phone's mobile data (force WiFi)")
    print("  5. Check firewall on laptop isn't blocking the connection")
    sys.exit(1)

print("SUCCESS: Connected! Reading frames...")
for i in range(10):
    ret, frame = cap.read()
    if ret:
        print(f"  Frame {i+1}: {frame.shape}")
    else:
        print(f"  Frame {i+1}: FAILED to read")

cap.release()
print("Done. If frames read successfully, use the same URL in the dashboard.")
