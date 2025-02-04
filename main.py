import cv2
from filters.filter_overlay import apply_filters

# List of filters to cycle through
FILTERS = ["none", "hat", "glasses", "mask","bald","full_face"]
filter_index = 0  # Start with the first filter

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the selected filter
    current_filter = FILTERS[filter_index]
    if current_filter != "none":
        frame = apply_filters(frame, current_filter)

    # Display the current filter name
    cv2.putText(frame, f"Filter: {current_filter.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show output
    cv2.imshow("AR Face Filters", frame)

    # Listen for key presses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Press 'Q' to exit
        break
    elif key == ord('f'):  # Press 'F' to cycle filters
        filter_index = (filter_index + 1) % len(FILTERS)
        print(f"Switched to {FILTERS[filter_index]} filter 🎭")

cap.release()
cv2.destroyAllWindows()
