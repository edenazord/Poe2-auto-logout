import tkinter as tk
import threading
import time
import cv2
import numpy as np
from mss import mss
import pyautogui  # Simulate key presses
import sys
import os

def resource_path(relative_path):
    """
    Returns the absolute path to the resource, compatible with PyInstaller.
    If the app is frozen (e.g. in an EXE), sys._MEIPASS contains the temporary folder path.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Global flags and variables
monitoring = False         # Indicates whether monitoring is active
monitor_option = "mana"    # "mana" or "life" (chosen via RadioButton)
monitor_thread = None      # Monitoring thread
color_threshold = 10       # Minimum threshold (%) for trigger (set by the user)
protection_blocked = False # Flag: if True, the trigger has already occurred
trigger_key = "esc"        # Default trigger key; can be "esc" or "f9"

# GUI variables (initialized in main)
root = None
status_label = None
threshold_entry = None
toggle_btn = None         # Single Start/Stop button

def get_blue_mask(orb_img):
    """
    Returns a mask of the "blue" pixels for the mana orb.
    Uses an HSV range to include the different shades.
    """
    hsv = cv2.cvtColor(orb_img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([70, 40, 40])
    upper_blue = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask

def get_red_mask(orb_img):
    """
    Returns a mask of the "red" pixels for the life orb.
    In HSV, red is split into two ranges: [0..10] and [170..180].
    """
    hsv = cv2.cvtColor(orb_img, cv2.COLOR_BGR2HSV)
    # First range (low red)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # Second range (high red)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def get_green_mask(orb_img):
    """
    Returns a mask of the "green" pixels for the life orb when poisoned.
    Uses an HSV range to capture green shades.
    """
    hsv = cv2.cvtColor(orb_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def update_status(message, color):
    """Updates the status message in the GUI in a thread-safe manner."""
    if status_label:
        status_label.after(0, lambda: status_label.config(text=message, fg=color))

def monitor_function():
    """
    Function executed in a separate thread while 'monitoring' is True.
    Captures the screen, performs template matching and counts the colored pixels
    to calculate the percentage fill of the orb.
    
    If the template is found and the percentage (of colored pixels) is below the threshold
    (and there is some color detected), the selected trigger key is pressed (Esc or F9),
    protection is blocked, and monitoring stops (the button then shows "Start").
    
    If the template is not found (i.e. the orbs are not detected), the status is updated to
    "Orbs not detected" without triggering any key.
    
    For the life orb, both the red and green spectra are combined.
    """
    global monitoring, monitor_option, color_threshold, protection_blocked, trigger_key

    # Load the templates using resource_path (useful for the EXE)
    mana_template = cv2.imread(resource_path("mana_template.png"), cv2.IMREAD_COLOR)
    life_template = cv2.imread(resource_path("life_template.png"), cv2.IMREAD_COLOR)
    
    if mana_template is None:
        print("ERROR: 'mana_template.png' not found or unreadable.")
        return
    if life_template is None:
        print("ERROR: 'life_template.png' not found or unreadable.")
        return
    
    # Template dimensions
    mana_h, mana_w = mana_template.shape[:2]
    life_h, life_w = life_template.shape[:2]
    
    # Initialize mss to capture the screen
    sct = mss()
    monitor_region = sct.monitors[1]  # Usually the main monitor
    template_threshold = 0.7  # Threshold for template matching

    print(">>> Monitor thread started. Press the button to stop.")

    while monitoring:
        # 1) Capture a screenshot of the monitored area
        screenshot = sct.grab(monitor_region)
        screen_img = np.array(screenshot, dtype=np.uint8)[..., :3]
        screen_img = np.ascontiguousarray(screen_img)

        # 2) Select the template based on the chosen option (mana or life)
        if monitor_option == "mana":
            template = mana_template
            t_h, t_w = mana_h, mana_w
        else:
            template = life_template
            t_h, t_w = life_h, life_w

        # 3) Perform template matching
        result = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= template_threshold:
            x, y = max_loc
            # 4) Crop the region of the orb
            region = {"top": y, "left": x, "width": t_w, "height": t_h}
            orb_screenshot = sct.grab(region)
            orb_img = np.array(orb_screenshot, dtype=np.uint8)[..., :3]
            orb_img = np.ascontiguousarray(orb_img)

            # 5) Count the colored pixels:
            # For mana orb, use blue mask.
            # For life orb, combine red and green masks.
            if monitor_option == "mana":
                mask = get_blue_mask(orb_img)
            else:
                red_mask = get_red_mask(orb_img)
                green_mask = get_green_mask(orb_img)
                mask = cv2.bitwise_or(red_mask, green_mask)

            pixel_count = cv2.countNonZero(mask)
            total_pixels = mask.size
            percent = (pixel_count / total_pixels) * 100

            print(f"[{monitor_option.upper()} MATCH] {percent:.2f}% | conf={max_val:.2f}")

            if not protection_blocked:
                if percent > color_threshold:
                    update_status("PROTECTED", "green")
                elif percent > 0.1:  # If there's at least some color but below the threshold
                    update_status("UNPROTECTED", "red")
                    print(f"*** {monitor_option.upper()} below {color_threshold}%! Pressing {trigger_key.upper()} ***")
                    pyautogui.press(trigger_key)
                    protection_blocked = True
                    monitoring = False
                    toggle_btn.after(0, lambda: toggle_btn.config(text="Start"))
                    break
                else:
                    # If the percentage is virtually 0, it means the orbs are not detected
                    update_status("Orbs not detected", "blue")
            else:
                update_status("UNPROTECTED", "red")
        else:
            # If the template is not found, the orbs are not detected.
            print(f"[{monitor_option.upper()} NO MATCH] conf={max_val:.2f}")
            update_status("Orbs not detected", "blue")

        time.sleep(0.3)

    print(">>> Monitor thread terminated.")

def start_monitoring():
    """Starts the monitoring thread and resets the protection block."""
    global monitoring, monitor_thread, color_threshold, protection_blocked
    if not monitoring:
        try:
            color_threshold = int(threshold_entry.get())
        except ValueError:
            color_threshold = 10
            threshold_entry.delete(0, tk.END)
            threshold_entry.insert(0, "10")
        protection_blocked = False  # Reset the block when restarting
        monitoring = True
        monitor_thread = threading.Thread(target=monitor_function, daemon=True)
        monitor_thread.start()

def stop_monitoring():
    """Stops monitoring."""
    global monitoring
    monitoring = False

def toggle_monitoring():
    """
    Single button that toggles between Start and Stop.
      - If monitoring is active, it stops it and shows "Start".
      - If it's inactive (or has been blocked), it starts it and shows "Stop".
    """
    global monitoring
    if monitoring:
        stop_monitoring()
        toggle_btn.config(text="Start")
        update_status("Monitoring stopped", "black")
    else:
        start_monitoring()
        toggle_btn.config(text="Stop")

def set_option_mana():
    """Callback for the RadioButton: selects 'mana'."""
    global monitor_option
    monitor_option = "mana"

def set_option_life():
    """Callback for the RadioButton: selects 'life'."""
    global monitor_option
    monitor_option = "life"

def set_trigger_esc():
    """Callback for the RadioButton: sets the trigger key to 'esc'."""
    global trigger_key
    trigger_key = "esc"

def set_trigger_f9():
    """Callback for the RadioButton: sets the trigger key to 'f9'."""
    global trigger_key
    trigger_key = "f9"

def main():
    global root, status_label, threshold_entry, toggle_btn

    root = tk.Tk()
    root.title("POE2 SaveAss")
    # Double the width of the UI (e.g., 600px wide, 400px tall)
    root.geometry("300x250")

    # RadioButtons for selecting the orb type
    orb_var = tk.StringVar(value="mana")
    r1 = tk.Radiobutton(root, text="Monitor Mana", variable=orb_var, value="mana", command=set_option_mana)
    r1.pack(anchor="w", padx=10, pady=5)
    r2 = tk.Radiobutton(root, text="Monitor Life", variable=orb_var, value="life", command=set_option_life)
    r2.pack(anchor="w", padx=10, pady=5)

    # Input for the minimum threshold (%) for trigger
    threshold_lbl = tk.Label(root, text="Minimum threshold (%) for trigger:")
    threshold_lbl.pack(anchor="w", padx=10, pady=5)
    threshold_entry = tk.Entry(root, width=10)
    threshold_entry.pack(anchor="w", padx=10, pady=5)
    threshold_entry.insert(0, "10")  # Default value

    # Frame for trigger key selection (on the same line)
    trigger_frame = tk.Frame(root)
    trigger_frame.pack(anchor="w", padx=10, pady=5)
    trigger_lbl = tk.Label(trigger_frame, text="Select Trigger Key:")
    trigger_lbl.pack(side="left", padx=(0,10))
    trigger_var = tk.StringVar(value="esc")
    t1 = tk.Radiobutton(trigger_frame, text="Esc", variable=trigger_var, value="esc", command=set_trigger_esc)
    t1.pack(side="left", padx=5)
    t2 = tk.Radiobutton(trigger_frame, text="F9", variable=trigger_var, value="f9", command=set_trigger_f9)
    t2.pack(side="left", padx=5)

    # Single Start/Stop button
    toggle_btn = tk.Button(root, text="Start", width=10, command=toggle_monitoring)
    toggle_btn.pack(padx=10, pady=5)

    # Label to display status ("PROTECTED" in green, "UNPROTECTED" in red, etc.)
    status_label = tk.Label(root, text="Monitoring stopped", font=("Helvetica", 12))
    status_label.pack(padx=10, pady=10)

    root.mainloop()
    stop_monitoring()

if __name__ == "__main__":
    main()
