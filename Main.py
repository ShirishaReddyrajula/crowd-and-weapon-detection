import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Initialize models as None
crowd_model = None
weapon_model = None

# Setup GUI
root = tk.Tk()
root.title("CCTV Crowd Management & Crime Prevention")
root.title("Crowd & Weapon Detection")
root.geometry("1000x720")
root.configure(bg="#e6f2ff")  # Light blue background

canvas = tk.Canvas(root, width=700, height=500, bg="white")
canvas.pack(pady=10)

# Status label
status_label = tk.Label(root, text="", font=("Arial", 12), bg="#e6f2ff")
status_label.pack(pady=5)

# Image placeholder
canvas_image_id = None
canvas_close_button = None

# Graph setup
fig = Figure(figsize=(5, 2), dpi=100)
ax = fig.add_subplot(111)
canvas_fig = None  # Will be initialized only on View Graph

def load_crowd_model():
    global crowd_model
    try:
        crowd_model = YOLO("yolov8_model/best.pt")
        status_label.config(text="Crowd Model Loaded ✅", fg="green")
    except Exception as e:
        status_label.config(text=f"Error: {e}", fg="red")

def load_weapon_model():
    global weapon_model
    try:
        weapon_model = YOLO("weapon_model/best.pt")
        status_label.config(text="Weapon Model Loaded ✅", fg="green")
    except Exception as e:
        status_label.config(text=f"Error: {e}", fg="red")

def show_image(img):
    global canvas_image_id, canvas_close_button
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((700, 500))
    img_tk = ImageTk.PhotoImage(image=img_pil)
    canvas_image_id = canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk

    # Add X button to close image
    if canvas_close_button:
        canvas.delete(canvas_close_button)
    canvas_close_button = canvas.create_text(680, 20, text="❌", font=("Arial", 20), fill="red", activefill="darkred")
    canvas.tag_bind(canvas_close_button, "<Button-1>", lambda e: clear_canvas())

def clear_canvas():
    global canvas_image_id, canvas_close_button
    if canvas_image_id:
        canvas.delete(canvas_image_id)
        canvas_image_id = None
    if canvas_close_button:
        canvas.delete(canvas_close_button)
        canvas_close_button = None
    canvas.image = None

def detect_objects(img):
    person_count = 0
    weapon_count = 0

    if crowd_model:
        results_crowd = crowd_model(img)[0]
        for box in results_crowd.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = crowd_model.names[cls]
            if label.lower() == "person":
                person_count += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if weapon_model:
        results_weapon = weapon_model(img)[0]
        for box in results_weapon.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = weapon_model.names[cls]
            weapon_count += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return img, person_count, weapon_count

def open_image():
    file_path = filedialog.askopenfilename(initialdir="images", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            status_label.config(text="Failed to read image", fg="red")
            return
        detected_img, _, _ = detect_objects(img)
        show_image(detected_img)

def open_video():
    file_path = filedialog.askopenfilename(initialdir="video", filetypes=[("Video files", "*.mp4 *.avi")])
    if not file_path:
        status_label.config(text="Video selection cancelled.", fg="orange")
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        status_label.config(text="Unable to open video file.", fg="red")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detected_frame, _, _ = detect_objects(frame)
        show_image(detected_frame)
        root.update()
    cap.release()


def graph():
    graph_img = cv2.imread('yolov8_model/results.png')
    graph_img = cv2.resize(graph_img, (800, 600))
    cv2.imshow("YOLO Training Graph", graph_img)
    cv2.waitKey(0)

# Button Layout
btn_frame = tk.Frame(root, bg="#e6f2ff")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Load Crowd Model", command=load_crowd_model).grid(row=0, column=0, padx=8)
tk.Button(btn_frame, text="Load Weapon Model", command=load_weapon_model).grid(row=0, column=1, padx=8)
tk.Button(btn_frame, text="Upload Image", command=open_image).grid(row=0, column=2, padx=8)
tk.Button(btn_frame, text="Upload Video", command=open_video).grid(row=0, column=3, padx=8)
tk.Button(btn_frame, text="View Graph", command=graph).grid(row=0, column=4, padx=8)

root.mainloop()
