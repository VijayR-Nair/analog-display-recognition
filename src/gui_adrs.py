"""
ADRS GUI (Tkinter)

Shows a simple UI to display the latest analog meter reading computed in main_adrs.py.
This GUI reads: main_adrs.measurement
"""

import tkinter as tk
import threading
import main_adrs  # imports the module where `measurement` is updated

# Window setup

window = tk.Tk()
window.geometry("400x400")
window.title("Analog Display Recognition System")

window.configure(bg="#F0F0F0")
font_title = ("Arial", 16, "bold")
font_label = ("Arial", 12)
font_entry = ("Arial", 11)

title_label = tk.Label(window, text="Analog Display Recognizer", font=font_title, bg="#F0F0F0")
title_label.pack(pady=20)

# Inputs 

tk.Label(window, text="Maximum", font=font_label, bg="#F0F0F0").place(x=70, y=100)
tk.Label(window, text="Minimum", font=font_label, bg="#F0F0F0").place(x=70, y=140)
tk.Label(window, text="Unit", font=font_label, bg="#F0F0F0").place(x=70, y=180)

maximum_entry = tk.Entry(window, font=font_entry, width=20)
minimum_entry = tk.Entry(window, font=font_entry, width=20)
unit_entry = tk.Entry(window, font=font_entry, width=20)

maximum_entry.place(x=150, y=100)
minimum_entry.place(x=150, y=140)
unit_entry.place(x=150, y=180)

# Output display

tk.Label(window, text="Result", font=font_label, bg="#F0F0F0").place(x=70, y=300)
output_value_label = tk.Label(window, text="--", font=font_label, bg="#F0F0F0")
output_value_label.place(x=150, y=300)

# Live update function

def refresh_value():
    """Fetch latest measurement and update UI (runs on main thread via after())."""
    try:
        # If you used the updated main_adrs with a lock, this is the safest read:
        if hasattr(main_adrs, "measurement_lock"):
            with main_adrs.measurement_lock:
                value = main_adrs.measurement
        else:
            value = main_adrs.measurement

        output_value_label.config(text=f"{value:.2f}")
    except Exception:
        output_value_label.config(text="N/A")

    # refresh again after 300ms
    window.after(300, refresh_value)

# Start ADRS backend in a thread

def start_backend():
    """
    Runs the main ADRS loop.
    This keeps the GUI responsive while main_adrs captures frames + runs inference.
    """
    try:
        if hasattr(main_adrs, "main"):
            main_adrs.main()
        # else: if your old main_adrs runs on import, no need to call anything
    except Exception as e:
        print("Backend error:", e)


backend_thread = threading.Thread(target=start_backend, daemon=True)
backend_thread.start()


# Start UI updates + run Tkinter loop
refresh_value()
window.mainloop()
