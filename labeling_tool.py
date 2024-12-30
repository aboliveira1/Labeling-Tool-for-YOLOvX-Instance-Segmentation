import os
import cv2
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import shutil
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from ultralytics import SAM

################################################################################
# STYLE CONFIGURATION
################################################################################
# Minimum dimensions for the Tkinter front-end window (auto-resizes above this)
MIN_WIDTH = 600
MIN_HEIGHT = 400

# General Tkinter window background color
APP_BG_COLOR = "#c8c9cf"

# Font settings for Tkinter
FONT_FAMILY = "Helvetica"  # e.g. "Times New Roman"
FONT_SIZE = 12
FONT = (FONT_FAMILY, FONT_SIZE)

# Text color for Tkinter labels
TEXT_FG_COLOR = "black"

# Tkinter button style
BUTTON_BG_COLOR = "#191921"
BUTTON_FG_COLOR = "#FFFFFF"

# Tkinter entry (input box) style
ENTRY_BG_COLOR = "#FFFFFF"
ENTRY_FG_COLOR = "#000000"

# Matplotlib figure/axes background colors
MATPLOTLIB_FIG_BG = "#c8c9cf"
MATPLOTLIB_AX_BG = "white"
MATPLOTLIB_TEXT_COLOR = "black"

# Matplotlib button style
MATPLOTLIB_BUTTON_BG = "#2b2b2b"       # normal color
MATPLOTLIB_BUTTON_HOVER = "#454545"    # hover color
MATPLOTLIB_BUTTON_TEXT = "white"       # label text color

# Radio button text color
MATPLOTLIB_RADIO_TEXT_COLOR = "black"

################################################################################
# DIRECTORIES, FUNCTIONS, AND CLASSES
################################################################################

# Subfolders in which we store images/labels after saving
images_dir = 'images'
labels_dir = 'labels'

def get_image_files(root_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(os.path.join(dirpath, filename))
    return image_files

def darken_color(bgr, factor=0.6):
    """
    Given a BGR color tuple (each in [0..255]),
    return a slightly darker version by multiplying each channel.
    """
    db = int(bgr[0] * factor)
    dg = int(bgr[1] * factor)
    dr = int(bgr[2] * factor)
    return (db, dg, dr)

class Labeler:
    def __init__(self, image_files, model, class_names):
        self.class_names = class_names
        self.class_ids = list(range(len(self.class_names)))
        
        # Use colors from 'tab10' colormap for better distinction
        cmap = matplotlib.cm.get_cmap('tab10')
        self.class_colors = {}
        for cls_id in self.class_ids:
            col = cmap(cls_id % 10)  # col is (R, G, B, A)
            r = int(col[0] * 255)
            g = int(col[1] * 255)
            b = int(col[2] * 255)
            self.class_colors[cls_id] = [b, g, r]  # store in BGR

        self.image_files = image_files
        self.model = model
        self.current_index = 0
        
        # "clicks" is for debugging/storing points if needed
        self.clicks = []
        
        # "masks" = list of finalized objects
        self.masks = []
        
        # "current_masks" = partial masks for the object currently being labeled
        self.current_masks = []
        
        # "newly_labeled_masks" = masks just finalized (for highlighting)
        self.newly_labeled_masks = []
        
        self.hover_mask = None
        self.last_hover_time = 0
        self.hover_update_interval = 0.5
        self.hover_x = None
        self.hover_y = None

        self.current_class_id = self.class_ids[0] if self.class_ids else None
        self.current_class_name = self.class_names[0] if self.class_names else ""

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Configure matplotlib styles
        plt.rcParams['figure.facecolor'] = MATPLOTLIB_FIG_BG
        plt.rcParams['axes.facecolor'] = MATPLOTLIB_AX_BG
        plt.rcParams['text.color'] = MATPLOTLIB_TEXT_COLOR
        plt.rcParams['font.size'] = 10

        self.load_image()

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.3, bottom=0.2)
        if self.img is not None:
            self.image_display = self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            self.ax.set_title(f'Image 1/{len(self.image_files)} - Current Class: {self.current_class_name}')
        else:
            self.image_display = self.ax.imshow(np.zeros((10,10,3), dtype=np.uint8))
            self.ax.set_title("No Image Loaded")
        plt.axis('off')

        # Add buttons
        self.add_buttons()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    def add_buttons(self):
        # Next/Previous/Redo/Label
        axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
        axredo = plt.axes([0.32, 0.05, 0.1, 0.075])
        axdone = plt.axes([0.44, 0.05, 0.14, 0.075])

        self.bprev = Button(axprev, 'Previous', color=MATPLOTLIB_BUTTON_BG, hovercolor=MATPLOTLIB_BUTTON_HOVER)
        self.bnext = Button(axnext, 'Next', color=MATPLOTLIB_BUTTON_BG, hovercolor=MATPLOTLIB_BUTTON_HOVER)
        self.bredo = Button(axredo, 'Redo', color=MATPLOTLIB_BUTTON_BG, hovercolor=MATPLOTLIB_BUTTON_HOVER)
        self.bdone = Button(axdone, 'Label Object', color=MATPLOTLIB_BUTTON_BG, hovercolor=MATPLOTLIB_BUTTON_HOVER)

        self.bprev.label.set_color(MATPLOTLIB_BUTTON_TEXT)
        self.bnext.label.set_color(MATPLOTLIB_BUTTON_TEXT)
        self.bredo.label.set_color(MATPLOTLIB_BUTTON_TEXT)
        self.bdone.label.set_color(MATPLOTLIB_BUTTON_TEXT)

        self.bprev.on_clicked(self.prev_image)
        self.bnext.on_clicked(self.next_image)
        self.bredo.on_clicked(self.redo)
        self.bdone.on_clicked(self.finish_object)

        # Radio buttons for class selection
        rax = plt.axes([0.05, 0.4, 0.2, 0.15])
        self.radio = RadioButtons(rax, self.class_names)
        for label in self.radio.labels:
            label.set_color(MATPLOTLIB_RADIO_TEXT_COLOR)
        self.radio.on_clicked(self.select_class)

    def select_class(self, label):
        self.current_class_name = label
        self.current_class_id = self.class_names.index(label)
        self.ax.set_title(
            f'Image {self.current_index + 1}/{len(self.image_files)} - Current Class: {self.current_class_name}'
        )
        self.fig.canvas.draw_idle()

    def load_image(self):
        if not self.image_files:
            print("No images to load.")
            self.img = None
            return
        image_path = self.image_files[self.current_index]
        self.img = cv2.imread(image_path)
        self.img_name = os.path.splitext(os.path.basename(image_path))[0]
        if self.img is None:
            print(f"Failed to load: {image_path}")
            return
        self.img_height, self.img_width = self.img.shape[:2]

        # Reset state
        self.clicks = []
        self.masks = []
        self.current_masks = []
        self.newly_labeled_masks = []
        self.hover_mask = None
        self.last_hover_time = 0
        self.hover_x = None
        self.hover_y = None

        # Load existing annotations if any
        annotation_path = os.path.join(labels_dir, self.img_name + '.txt')
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) < 6:
                        continue
                    cls_id = int(tokens[0])
                    coords = [float(x) for x in tokens[5:]]
                    if len(coords) % 2 != 0:
                        continue

                    if cls_id not in self.class_ids:
                        print(f"Warning: Skipping annotation with class_id={cls_id} not in {self.class_ids}")
                        continue

                    # Reconstruct the polygon as a binary mask
                    contour = []
                    for i in range(0, len(coords), 2):
                        x_val = int(coords[i] * self.img_width)
                        y_val = int(coords[i+1] * self.img_height)
                        contour.append([x_val, y_val])
                    contour = np.array(contour, dtype=np.int32)
                    mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 1)
                    self.masks.append({'mask': mask.astype(bool), 'class_id': cls_id})

    def update_display(self):
        self.ax.clear()
        if self.img is None:
            self.ax.set_title("No Image Loaded")
            plt.axis('off')
            self.fig.canvas.draw_idle()
            return

        overlay = self.img.copy()

        # Draw finalized masks + in-progress masks
        for mask_dict in self.masks + self.current_masks:
            mask = mask_dict['mask']
            class_id = mask_dict['class_id']
            if class_id not in self.class_colors:
                print(f"Warning: No color for class_id={class_id}; skipping.")
                continue
            color = np.array(self.class_colors[class_id], dtype=np.uint8)
            mask_img = np.zeros_like(self.img)
            mask_img[mask] = color
            overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.3, 0)

        # If hover mask, show it in current class color
        if self.hover_mask is not None:
            if self.current_class_id in self.class_colors:
                color = np.array(self.class_colors[self.current_class_id], dtype=np.uint8)
                mask_img = np.zeros_like(self.img)
                mask_img[self.hover_mask] = color
                overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.3, 0)

        # Highlight newly labeled masks with darker contour
        for mask_dict in self.newly_labeled_masks:
            mask = mask_dict['mask'].astype(np.uint8)
            class_id = mask_dict['class_id']
            if class_id not in self.class_colors:
                continue
            base_color = self.class_colors[class_id]
            contour_color = darken_color(base_color, factor=0.6)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    cv2.drawContours(overlay, [cnt], -1, contour_color, thickness=2)

        self.image_display = self.ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        self.ax.set_title(
            f'Image {self.current_index + 1}/{len(self.image_files)} - Current Class: {self.current_class_name}'
        )
        plt.axis('off')
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Left-click => add partial mask
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            x_click = int(event.xdata)
            y_click = int(event.ydata)
            self.clicks.append([x_click, y_click])

            # Run SAM for that point
            input_point = [[x_click, y_click]]
            input_label = [1]
            results = self.model.predict(self.img, points=input_point, labels=input_label)
            mask = results[0].masks.data[0].cpu().numpy()
            binary_mask = (mask > 0)

            self.current_masks.append({'mask': binary_mask, 'class_id': self.current_class_id})
            self.update_display()

        # Right-click => delete if we clicked on a mask
        elif event.button == 3 and event.xdata is not None and event.ydata is not None:
            x_click = int(event.xdata)
            y_click = int(event.ydata)
            # Find if this point is inside any existing finalized mask (topmost if overlap)
            found_index = self.find_mask_at_point(x_click, y_click)
            if found_index is not None:
                # Prompt user
                answer = messagebox.askyesno(
                    "Delete Object",
                    "Do you want to remove this object from the .txt file?"
                )
                if answer:
                    # Remove that mask
                    self.masks.pop(found_index)
                    self.newly_labeled_masks = []
                    self.save_annotation()
                    self.update_display()

    def find_mask_at_point(self, x, y):
        """
        Check from the top (last in self.masks) down to find 
        which mask (if any) contains this point. Return its index or None.
        """
        for i in reversed(range(len(self.masks))):
            mask_dict = self.masks[i]
            if mask_dict['mask'][y, x]:
                return i
        return None

    def on_hover(self, event):
        if event.inaxes != self.ax:
            return
        current_time = time.time()
        if current_time - self.last_hover_time < self.hover_update_interval:
            return
        if event.xdata is not None and event.ydata is not None:
            x_hover = int(event.xdata)
            y_hover = int(event.ydata)
            # Only update if the mouse moved significantly
            if self.hover_x is not None and self.hover_y is not None:
                if abs(x_hover - self.hover_x) < 5 and abs(y_hover - self.hover_y) < 5:
                    return
            self.hover_x = x_hover
            self.hover_y = y_hover
            self.last_hover_time = current_time

            # Generate hover mask
            input_point = [[x_hover, y_hover]]
            input_label = [1]
            results = self.model.predict(self.img, points=input_point, labels=input_label)
            mask = results[0].masks.data[0].cpu().numpy()
            self.hover_mask = (mask > 0)
            self.update_display()

    def on_key_press(self, event):
        if event.key == ' ':
            # Same as "Label Object" button
            self.finish_object(None)
        elif event.key == 'n':
            # Next image
            self.save_annotation()
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.load_image()
            self.update_display()
        elif event.key == 'p':
            # Previous image
            self.save_annotation()
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.load_image()
            self.update_display()
        elif event.key == 'd':
            # Redo
            self.redo(None)

    def finish_object(self, event):
        """
        Finalize the current object(s).
        Merge partial masks in self.current_masks into self.masks,
        highlight them, then save annotation.
        """
        self.masks.extend(self.current_masks)
        self.newly_labeled_masks = self.current_masks[:]
        self.current_masks = []
        self.clicks = []
        self.save_annotation()
        self.update_display()

    def save_annotation(self):
        if self.img is None:
            return

        # Save original image in 'images' folder
        image_save_path = os.path.join(images_dir, os.path.basename(self.image_files[self.current_index]))
        cv2.imwrite(image_save_path, self.img)
        
        all_masks = self.masks + self.current_masks
        annotation_path = os.path.join(labels_dir, self.img_name + '.txt')

        # If no masks left, remove .txt if it exists
        if not all_masks:
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
            return

        # Otherwise, write them
        with open(annotation_path, 'w') as f:
            for mask_dict in all_masks:
                mask = mask_dict['mask'].astype(np.uint8)
                class_id = mask_dict['class_id']
                if class_id not in self.class_ids:
                    continue

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        x_min = np.min(contour[:, 0, 0])
                        y_min = np.min(contour[:, 0, 1])
                        x_max = np.max(contour[:, 0, 0])
                        y_max = np.max(contour[:, 0, 1])
                        x_center = (x_min + x_max) / 2 / self.img_width
                        y_center = (y_min + y_max) / 2 / self.img_height
                        width = (x_max - x_min) / self.img_width
                        height = (y_max - y_min) / self.img_height
                        contour = contour.reshape(-1, 2)
                        contour_norm = []
                        for x_val, y_val in contour:
                            x_norm = x_val / self.img_width
                            y_norm = y_val / self.img_height
                            contour_norm.extend([str(x_norm), str(y_norm)])
                        line = [
                            str(class_id),
                            str(x_center),
                            str(y_center),
                            str(width),
                            str(height)
                        ] + contour_norm
                        f.write(' '.join(line) + '\n')

    def next_image(self, event):
        self.save_annotation()
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.load_image()
        self.update_display()

    def prev_image(self, event):
        self.save_annotation()
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.load_image()
        self.update_display()

    def redo(self, event):
        """
        'Redo' removes the most recently added partial mask if any exist;
        otherwise, asks if the user wants to remove the most recently
        finalized mask from the .txt file.
        """
        if self.current_masks:
            self.current_masks.pop()
            if self.clicks:
                self.clicks.pop()
            self.update_display()
        elif self.masks:
            answer = messagebox.askyesno(
                "Remove Labeled Object",
                "Do you want to remove the most recently labeled object from the .txt file?"
            )
            if answer:
                self.masks.pop()
                self.newly_labeled_masks = []
                self.save_annotation()
            self.update_display()

################################################################################
# TKINTER FRONT-END
################################################################################
def run_frontend():
    """
    A Tkinter GUI that lets the user:
      1) Select a directory of images,
      2) Dynamically add label names,
      3) Launch the Labeler (with local 'sam2.1_b.pt'),
      4) Optionally split data into train/valid/test.
    """
    root = tk.Tk()
    root.title("Labeling Tool")

    # Let the window auto-size, but maintain a minimum so it doesn't get too small
    root.minsize(MIN_WIDTH, MIN_HEIGHT)
    root.resizable(True, True)
    root.configure(bg=APP_BG_COLOR)

    # Title label
    label_title = tk.Label(
        root,
        text="Labeling Tool",
        bg=APP_BG_COLOR,
        fg=TEXT_FG_COLOR,
        font=(FONT_FAMILY, 16, 'bold')
    )
    label_title.pack(pady=10)

    label_entries = []

    def browse_images_dir():
        directory = filedialog.askdirectory(title="Select Images Directory")
        if directory:
            images_dir_var.set(directory)

    def add_label_entry():
        entry_var = tk.StringVar()
        entry_widget = tk.Entry(
            labels_frame,
            textvariable=entry_var,
            width=30,
            bg=ENTRY_BG_COLOR,
            fg=ENTRY_FG_COLOR,
            font=FONT
        )
        entry_widget.pack(pady=2, fill=tk.X)
        label_entries.append(entry_var)
        root.update_idletasks()

    def start_labeling():
        root_dir = images_dir_var.get()
        user_labels = [var.get().strip() for var in label_entries if var.get().strip()]

        if not user_labels:
            user_labels = ["object"]  # fallback

        image_files = get_image_files(root_dir)
        if not image_files:
            print("No images found in the specified directory.")
            root.destroy()
            return

        sam_path = "sam2.1_b.pt"  # local model file
        model = SAM(sam_path)
        Labeler(image_files, model, user_labels)

        root.destroy()

    def split_data_gui():
        """
        Opens a small dialog for splitting data into train/valid/test.
        """
        popup = tk.Toplevel(root)
        popup.title("Split Data")
        popup.configure(bg=APP_BG_COLOR)
        popup.geometry("300x200")

        tk.Label(popup, text="Train %:", bg=APP_BG_COLOR, fg=TEXT_FG_COLOR, font=FONT).pack(pady=5)
        train_var = tk.StringVar(value="70")
        tk.Entry(popup, textvariable=train_var, width=10).pack()

        tk.Label(popup, text="Valid %:", bg=APP_BG_COLOR, fg=TEXT_FG_COLOR, font=FONT).pack(pady=5)
        valid_var = tk.StringVar(value="20")
        tk.Entry(popup, textvariable=valid_var, width=10).pack()

        tk.Label(popup, text="Test %:", bg=APP_BG_COLOR, fg=TEXT_FG_COLOR, font=FONT).pack(pady=5)
        test_var = tk.StringVar(value="10")
        tk.Entry(popup, textvariable=test_var, width=10).pack()

        def do_split_now():
            train_p = float(train_var.get())
            valid_p = float(valid_var.get())
            test_p = float(test_var.get())
            total = train_p + valid_p + test_p
            if total <= 0:
                messagebox.showerror("Error", "Sum of percentages must be > 0.")
                return

            # Normalize
            train_ratio = train_p / total
            valid_ratio = valid_p / total
            test_ratio = test_p / total

            # All images in 'images' folder
            all_images = get_image_files(images_dir)
            # Pair each image with its .txt in labels_dir if present
            data_pairs = []
            for img_path in all_images:
                base = os.path.splitext(os.path.basename(img_path))[0]
                txt_path = os.path.join(labels_dir, base + ".txt")
                data_pairs.append((img_path, txt_path if os.path.exists(txt_path) else None))

            # Shuffle
            random.shuffle(data_pairs)

            # Compute split indices
            n_total = len(data_pairs)
            n_train = int(n_total * train_ratio)
            n_valid = int(n_total * valid_ratio)
            # test = the rest
            n_test = n_total - (n_train + n_valid)

            train_split = data_pairs[:n_train]
            valid_split = data_pairs[n_train:n_train + n_valid]
            test_split = data_pairs[n_train + n_valid:]

            # Create subdirs
            for split_name in ["train", "valid", "test"]:
                os.makedirs(os.path.join(split_name, "images"), exist_ok=True)
                os.makedirs(os.path.join(split_name, "labels"), exist_ok=True)

            def copy_files(pairs, split_name):
                for img_path, lbl_path in pairs:
                    # images
                    dst_img = os.path.join(split_name, "images", os.path.basename(img_path))
                    shutil.copyfile(img_path, dst_img)
                    # labels
                    if lbl_path:
                        dst_lbl = os.path.join(split_name, "labels", os.path.basename(lbl_path))
                        shutil.copyfile(lbl_path, dst_lbl)

            copy_files(train_split, "train")
            copy_files(valid_split, "valid")
            copy_files(test_split, "test")

            # Create data.yaml
            yaml_path = os.path.join("data.yaml")
            n_classes = len(label_entries)
            # If user didn't input anything, we used ["object"], so handle that:
            names_list = [var.get().strip() for var in label_entries if var.get().strip()] or ["object"]

            with open(yaml_path, 'w') as f:
                f.write(f"train: ../train/images\n")
                f.write(f"val: ../valid/images\n")
                f.write(f"test: ../test/images\n\n")
                f.write(f"nc: {len(names_list)}\n")
                f.write(f"names: {names_list}\n")

            messagebox.showinfo("Done", "Data has been split and 'data.yaml' created.")
            popup.destroy()

        tk.Button(popup, text="Split Now", command=do_split_now, bg=BUTTON_BG_COLOR, fg=BUTTON_FG_COLOR).pack(pady=10)

    images_dir_var = tk.StringVar()

    # Folder selection
    tk.Label(
        root,
        text="Please select your images directory:",
        bg=APP_BG_COLOR,
        fg=TEXT_FG_COLOR,
        font=FONT
    ).pack(pady=5)

    frame_dir = tk.Frame(root, bg=APP_BG_COLOR)
    frame_dir.pack(pady=(0,10))

    tk.Entry(
        frame_dir, 
        textvariable=images_dir_var, 
        width=50, 
        bg=ENTRY_BG_COLOR,
        fg=ENTRY_FG_COLOR, 
        font=FONT
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        frame_dir, 
        text="Browse", 
        command=browse_images_dir,
        bg=BUTTON_BG_COLOR, 
        fg=BUTTON_FG_COLOR, 
        font=FONT,
        width=10,
        height=1
    ).pack(side=tk.LEFT)

    # Label entry area
    tk.Label(
        root,
        text="Add your labels/classes below:",
        bg=APP_BG_COLOR,
        fg=TEXT_FG_COLOR,
        font=FONT
    ).pack(pady=(0,5))

    labels_frame = tk.Frame(root, bg=APP_BG_COLOR)
    labels_frame.pack()

    # Add a default label entry
    add_label_entry()

    tk.Button(
        root,
        text="Add Another Label",
        command=add_label_entry,
        bg=BUTTON_BG_COLOR,
        fg=BUTTON_FG_COLOR,
        font=FONT,
        width=18,
        height=1
    ).pack(pady=5)

    # Start labeling
    tk.Button(
        root,
        text="Start Labeling",
        command=start_labeling,
        bg=BUTTON_BG_COLOR,
        fg=BUTTON_FG_COLOR,
        font=FONT,
        width=15,
        height=2
    ).pack(pady=5)

    # Split Data button
    tk.Button(
        root,
        text="Split and Prepare Data",
        command=split_data_gui,
        # bg=BUTTON_BG_COLOR,
        bg="#152036",
        fg=BUTTON_FG_COLOR,
        font=FONT,
        width=20,
        height=2
    ).pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    run_frontend()
