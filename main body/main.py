from Count.Fluorescent_analysis import root_path
from Test_tubes_segmentation.pic_batch_straightening        import (auto_pic_point_select,
                                                                    auto_batch_straighten_and_crop,
                                                                    manual_batch_straighten,
                                                                    select_two_points,
                                                                    batch_crop,
                                                                    select_two_points_crop)
from Small_GUI.GUI              import select_folders,show_error_popup,get_number,select_count_folder
from Test_tubes_segmentation.intensity_scan import (segmenting,
                                                    threshold_segmenting,
                                                    scan_low_intensity_on_overlay)
from Count import (
                   analysis_intensity_over_time,
                    plot_cell_count_over_time_by_ratio,
                    plot_normalized_intensity_over_time_by_ratio,
                    save_to_excel,
                    organize_files_by_time_part)
from semiauto import gather_all_semi_segment_part
from semiauto.semiauto import crop_two_sides,select_segment_area,select_two_points_crop
import tkinter as tk
from tkinter import messagebox
import os

def main():

    # preprocess
    def Automation():
        # Explicitly call folder selection on each run
        input_folder, output_folder = select_folders()
        channel_number = get_number()

        if not input_folder or not output_folder:
            print("File selection canceled.")
            return  # Exit the program if no folder is selected

        # The next processing logic
        organize_files_by_time_part(input_folder)

        try:
            #straighten and crop part

            auto_batch_straighten_and_crop(*auto_pic_point_select(input_folder, output_folder))
            batch_crop(*select_two_points_crop(output_folder, output_folder))
            overlay_coordinates = scan_low_intensity_on_overlay(output_folder, output_folder, channel_number)
            segmenting(overlay_coordinates, output_folder, output_folder)
            threshold_segmenting(overlay_coordinates, output_folder, output_folder)



        except RuntimeError:
            show_error_popup()
            #straighten and crop part
            # Pop-up prompt at the end of the run
            messagebox.showinfo("Manual Straighten", "Please select two points in both half in the mid of traps")
            manual_batch_straighten(*select_two_points(input_folder, output_folder))
            messagebox.showinfo("Manual Crop", "Please select one point on the upper left and one point on the lower right to crop out the trap part.")
            batch_crop(*select_two_points_crop(output_folder, output_folder))
            overlay_coordinates = scan_low_intensity_on_overlay(output_folder, output_folder, channel_number)
            segmenting(overlay_coordinates, output_folder, output_folder)
            threshold_segmenting(overlay_coordinates, output_folder, output_folder)

        # Pop-up prompt at the end of the run
        messagebox.showinfo("Notice", "preprocess has finished!")

    # Count
    def function2():
        #count part
        root_path=select_count_folder()
        df = analysis_intensity_over_time(root_path)
        save_to_excel(df,root_path)
        plot_normalized_intensity_over_time_by_ratio(df,root_path)
        plot_cell_count_over_time_by_ratio(df,root_path)

        # Notice after finished running
        messagebox.showinfo("Notice", "count has finished！")

    def semiauto():
        input = select_count_folder()
        crop_two_sides(*select_two_points_crop(input, input))
        select_segment_area(input, input)
        gather_all_semi_segment_part(root_path)

    def function4():
        # count part
        root_path = select_count_folder()
        for semi_segment in os.listdir(root_path):
            if "semi_segment" != semi_segment:
                continue
            semi_segment_folder = os.path.join(root_path, semi_segment)
            for all in os.listdir(semi_segment_folder):
                if all != "all":
                    continue
                all_folder = os.path.join(semi_segment_folder, all)
                df = analysis_intensity_over_time(all_folder)
                save_to_excel(df, all_folder)
                plot_normalized_intensity_over_time_by_ratio(df, all_folder)
                messagebox.showinfo("Notice", "count in semi segmentation has finished！")

    # Create the main window
    window = tk.Tk()
    window.title("Choose the function")
    window.geometry("300x300")

    # The notice of function choosing
    instruction = tk.Label(window, text="Please select a function：", font=("Arial", 12))
    instruction.pack(pady=10)  # add spacing

    # Add the buttons
    button1 = tk.Button(window, text="preprocess(Automation)", command=Automation, width=30, height=2)
    button1.pack(pady=10)  # set the spacing between the buttons

    button2 = tk.Button(window, text="count", command=function2, width=30, height=2)
    button2.pack(pady=10)

    button3 = tk.Button(window, text="preprocess(semi-automation)", command=semiauto, width=30, height=2)
    button3.pack(pady=10)
    # main window loop
    window.mainloop()

    button4 = tk.Button(window, text="count(semi-automation)", command=function4, width=30, height=2)
    button4.pack(pady=10)
    # main window loop
    window.mainloop()


if __name__ == "__main__":
    main()