import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- Configuration ---
IMAGE_PATH = "images\\50_whorl.jpg"  # <-- **CHANGE THIS to your image file**
OUTPUT_FILE = "clicked_points.txt"  # The file where coordinates will be saved
# ---------------------

# A list to keep track of all points clicked in the current session
session_points = []


def onclick(event):
    """Event handler for mouse clicks."""
    if event.xdata is not None and event.ydata is not None:
        # 1. Get and format coordinates
        x = int(round(event.xdata))
        y = int(round(event.ydata))

        # Format the line to save
        point_data = f"X: {x}, Y: {y}"

        # 2. Print to console for immediate feedback
        print(f"Clicked point: {point_data}")

        # 3. Store and persist to file
        session_points.append((x, y))
        try:
            # Use 'a' (append mode) to add the new point without erasing old ones
            with open(OUTPUT_FILE, "a") as f:
                f.write(point_data + "\n")
        except IOError:
            print(f"Error: Could not write to file {OUTPUT_FILE}")

        # 4. Plot on image for visual confirmation
        plt.plot(x, y, "ro")  # 'ro' means red circle
        plt.draw()  # Redraw the figure to show the new point


# --- Main execution ---
try:
    # Check if the output file already exists and prompt the user
    if os.path.exists(OUTPUT_FILE):
        print(f"⚠️ Warning: File '{OUTPUT_FILE}' already exists.")
        response = (
            input("Do you want to (A)ppend new points or (O)verwrite the file? (A/O): ")
            .strip()
            .lower()
        )
        if response == "o":
            # Clear the file by opening it in 'w'rite mode and immediately closing
            with open(OUTPUT_FILE, "w") as f:
                f.write("# Coordinates stored as X: [pixel_x], Y: [pixel_y]\n")
            print(f"File '{OUTPUT_FILE}' has been cleared. Starting fresh.")
        elif response == "a":
            print(f"Appending to existing file '{OUTPUT_FILE}'.")
        else:
            print("Invalid choice. Exiting script.")
            sys.exit(0)
    else:
        # Create the file with a header if it doesn't exist
        with open(OUTPUT_FILE, "w") as f:
            f.write("# Coordinates stored as X: [pixel_x], Y: [pixel_y]\n")

    # Load the image
    img = plt.imread(IMAGE_PATH)

    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(
        "Click anywhere to pick and save coordinates. Close the window when finished."
    )

    # Connect the click handler
    fig.canvas.mpl_connect("button_press_event", onclick)

    print("\n--- Point Picker Active ---")
    print(f"Loading image from: {IMAGE_PATH}")
    print(f"Coordinates are being saved to: {OUTPUT_FILE}")
    print("Close the image window to stop the script.")

    # Show the interactive window (blocking call)
    plt.show()

except FileNotFoundError:
    print(f"\nError: Image file not found at {IMAGE_PATH}. Please check the path.")
    sys.exit(1)
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    sys.exit(1)
