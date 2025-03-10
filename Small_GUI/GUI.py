import os.path
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox
import shutil


def get_user_input_output():
    # Create a main window and hide it
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Popup input box to let the user enter information
    input_folder = simpledialog.askstring(
        "Input folder path", "Please enter the input folder path:"
    )
    output_folder = simpledialog.askinteger(
        "Output folder path", "Please enter the output folder path:"
    )

    if os.path.isfile(input_folder) and os.path.isfile(output_folder):
        # Return the information entered by the user
        return input_folder, output_folder
    else:
        print("These paths don't exist.")


def select_folders():
    """
    Opens two folder selection dialogs and returns the paths for the input and output folders.
    """
    # Popup folder selection dialog to select the input folder
    input_folder = filedialog.askdirectory(title="Select Input Folder")
    if not input_folder:
        print("No input folder selected")
        return None, None  # Return None if no input folder is selected

    # Popup folder selection dialog to select the output folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        print("No output folder selected")
        return None, None  # Return None if no output folder is selected

    # Check if the output folder is empty
    if os.listdir(output_folder):
        print(f"Output folder {output_folder} is not empty.")
        clear_choice = input("Clear output folder? (yes/no): ")
        if clear_choice.lower() == "yes":
            for filename in os.listdir(output_folder):
                file_path = os.path.join(output_folder, filename)
                if os.path.isfile(file_path) or os.path.isdir(file_path):
                    (
                        os.remove(file_path)
                        if os.path.isfile(file_path)
                        else shutil.rmtree(file_path)
                    )
            print("Output folder cleared.")
        else:
            print("Please select a different output folder.")
            return select_folders()

    return input_folder, output_folder


def select_count_folder():
    # Popup folder selection dialog to select the output folder
    output_folder = filedialog.askdirectory(title="Select Folder")
    if not output_folder:
        print("No output folder selected")
        return None, None  # Return None if no output folder is selected
    return output_folder


def show_error_popup():
    # Initialize the tkinter main window
    root = tk.Tk()
    root.withdraw()  # Hide the main window and only show the popup

    # Use messagebox to display error information
    messagebox.showerror(
        "Run time error", "The Image can not be automatically segmented"
    )

    # Destroy the main window
    root.destroy()


def get_number():
    """
    Creates a GUI to ask the user for a number and returns the entered value.
    """
    # Create the main window
    root = tk.Tk()
    root.withdraw()  # Hide the root window as we only need the dialog

    # Ask the user for a number using a simple dialog
    user_input = simpledialog.askstring(
        "Input", "Please enter the number of test tubes:"
    )

    # Check if the input is valid
    try:
        # Convert the input string to a floating-point number
        number = float(user_input)
        return number
    except (ValueError, TypeError):
        print("Invalid input. Please enter a valid number.")
        return None  # Return None if the input is invalid
