import tkinter as tk
from tkinter import filedialog, Toplevel
import cv2
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from filters import Filters
from model_yunet import detect_face_yunet

class App:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed') 
        self.root.title("Image Processing Dashboard")
        self.root.configure(bg="#111827")

        self.BG   = "#111827"
        self.CARD = "#1f2937"
        self.BTN  = "#2563eb"
        self.TEXT = "white"

        self.original_img  = None 
        self.base_img      = None 
        self.processed_img = None 

        self.val_brightness = tk.DoubleVar(value=0)
        self.val_contrast   = tk.DoubleVar(value=1.0)
        self.val_gamma      = tk.DoubleVar(value=1.0)
        self.val_blur       = tk.IntVar(value=1)
        self.val_rotate     = tk.DoubleVar(value=0)

        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.adj_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Adjustments", menu=self.adj_menu)
        self.adj_menu.add_command(label="Brightness & Contrast", command=self.open_brightness_window)
        self.adj_menu.add_command(label="Gamma Correction", command=self.open_gamma_window)
        self.adj_menu.add_command(label="Blur Tools", command=self.open_blur_window)
        self.adj_menu.add_command(label="Rotation", command=self.open_rotation_window)

        self.filter_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Filters", menu=self.filter_menu)
        self.filter_menu.add_command(label="Histogram Equalization", command=self.apply_hist_eq)
        self.filter_menu.add_separator()
        self.filter_menu.add_command(label="Sobel Edge", command=self.apply_sobel)
        self.filter_menu.add_command(label="Canny Edge", command=self.apply_canny)
        self.filter_menu.add_command(label="Laplacian", command=self.apply_laplacian)
        self.filter_menu.add_separator()
        self.filter_menu.add_command(label="Averaging", command=self.apply_averaging)
        self.filter_menu.add_command(label="Median Blur", command=self.apply_median)
        self.filter_menu.add_command(label="Bilateral Filter", command=self.apply_bilateral)
        self.filter_menu.add_separator()
        self.filter_menu.add_command(label="Zoom (Nearest)", command=self.apply_zoom_nearest)
        self.filter_menu.add_command(label="Zoom (Bilinear)", command=self.apply_zoom_bilinear)

        self.main_canvas = tk.Canvas(self.root, bg=self.BG, highlightthickness=0)
        self.scrollbar_y = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollbar_x = tk.Scrollbar(self.root, orient="horizontal", command=self.main_canvas.xview)
        self.main_canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.main_canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame = tk.Frame(self.main_canvas, bg=self.BG)
        self.scroll_window = self.main_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>", self.on_frame_configure)
        self.main_canvas.bind("<Configure>", self.on_canvas_configure)
        self.main_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        self.top_frame = tk.Frame(self.scroll_frame, bg=self.BG)
        self.top_frame.pack(fill="x", side="top", padx=10, pady=(0, 10), anchor="nw")

        self.upload_btn = tk.Button(self.top_frame, text="Upload Image", command=self.load_image,
                            bg=self.BTN, fg="white", font=("Arial", 11, "bold"), width=16)
        self.upload_btn.grid(row=0, column=0, padx=10, pady=5)

        self.detect_btn = tk.Button(self.top_frame, text="Face Detection", command=self.open_face_window,
                                    bg="#db2777", fg="white", font=("Arial", 11, "bold"), width=16)
        self.detect_btn.grid(row=0, column=1, padx=10)

        self.reset_btn = tk.Button(self.top_frame, text="Reset All", command=self.reset_image,
                                   bg="#dc2626", fg="white", font=("Arial", 11, "bold"), width=12)
        self.reset_btn.grid(row=0, column=2, padx=10)
        
        self.zoom_card = tk.Frame(self.top_frame, bg=self.CARD, bd=1, relief="solid")
        self.zoom_card.grid(row=0, column=3, padx=20)
        tk.Label(self.zoom_card, text="Zoom Scale", bg=self.CARD, fg="white", font=("Arial", 8)).pack(side="left", padx=5)
        self.zoom_slider = tk.Scale(self.zoom_card, from_=1.0, to=5.0, resolution=0.1, orient="horizontal", 
                                    bg=self.CARD, fg="white", troughcolor="#374151", length=120, highlightthickness=0)
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(side="left")

        self.content_frame = tk.Frame(self.scroll_frame, bg=self.BG)
        self.content_frame.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(self.content_frame, bg=self.BG)
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.content_frame, bg=self.BG)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.panel_original = tk.Label(self.left_frame, bg="black", bd=3, relief="solid")
        self.panel_original.pack(pady=10)
        self.panel_processed = tk.Label(self.left_frame, bg="black", bd=3, relief="solid")
        self.panel_processed.pack(pady=10)

        self.fig1, self.ax1, self.canvas1 = self.setup_hist_widget(self.right_frame, "Original Histogram")
        self.fig2, self.ax2, self.canvas2 = self.setup_hist_widget(self.right_frame, "Processed Histogram")

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        img = cv2.imread(path)
        if img is None: return
        self.original_img = img.copy()
        self.commit_changes(img)

    def commit_changes(self, new_img):
        self.base_img = new_img.copy()
        self.processed_img = new_img.copy()
        self.val_brightness.set(0)
        self.val_contrast.set(1.0)
        self.val_gamma.set(1.0)
        self.val_blur.set(1)
        self.val_rotate.set(0)
        self.update_display()

    def reset_image(self):
        if self.original_img is not None:
            self.commit_changes(self.original_img)

    def run_slider_pipeline(self, event=None):
        if self.base_img is None: return
        img = self.base_img.copy()
        img = Filters.apply_brightness_contrast(img, self.val_brightness.get(), self.val_contrast.get())
        img = Filters.apply_gamma(img, self.val_gamma.get())
        img = Filters.apply_gaussian_blur(img, self.val_blur.get())
        img = Filters.apply_rotation(img, self.val_rotate.get())
        self.processed_img = img
        self.update_display()

    def open_brightness_window(self):
        if self.base_img is None: return
        win = Toplevel(self.root); win.title("Brightness & Contrast"); win.geometry("300x250"); win.configure(bg=self.CARD)
        tk.Label(win, text="Brightness", bg=self.CARD, fg="white").pack(pady=5)
        tk.Scale(win, from_=-100, to=100, orient="horizontal", variable=self.val_brightness, command=self.run_slider_pipeline, bg=self.CARD, fg="white").pack(fill="x", padx=20)
        tk.Label(win, text="Contrast", bg=self.CARD, fg="white").pack(pady=5)
        tk.Scale(win, from_=0.5, to=3.0, resolution=0.1, orient="horizontal", variable=self.val_contrast, command=self.run_slider_pipeline, bg=self.CARD, fg="white").pack(fill="x", padx=20)

    def open_gamma_window(self):
        if self.base_img is None: return
        win = Toplevel(self.root); win.title("Gamma Correction"); win.geometry("300x150"); win.configure(bg=self.CARD)
        tk.Label(win, text="Gamma Value", bg=self.CARD, fg="white").pack(pady=5)
        tk.Scale(win, from_=0.1, to=3.0, resolution=0.1, orient="horizontal", variable=self.val_gamma, command=self.run_slider_pipeline, bg=self.CARD, fg="white").pack(fill="x", padx=20)

    def open_blur_window(self):
        if self.base_img is None: return
        win = Toplevel(self.root); win.title("Gaussian Blur"); win.geometry("300x150"); win.configure(bg=self.CARD)
        tk.Label(win, text="Blur Intensity (Odd)", bg=self.CARD, fg="white").pack(pady=5)
        tk.Scale(win, from_=1, to=200, orient="horizontal", variable=self.val_blur, command=self.run_slider_pipeline, bg=self.CARD, fg="white").pack(fill="x", padx=20)

    def open_rotation_window(self):
        if self.base_img is None: return
        win = Toplevel(self.root); win.title("Rotation"); win.geometry("300x150"); win.configure(bg=self.CARD)
        tk.Label(win, text="Angle (Degrees)", bg=self.CARD, fg="white").pack(pady=5)
        tk.Scale(win, from_=-180, to=180, orient="horizontal", variable=self.val_rotate, command=self.run_slider_pipeline, bg=self.CARD, fg="white").pack(fill="x", padx=20)

    def apply_hist_eq(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_hist_eq(self.processed_img))

    def apply_sobel(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_sobel(self.processed_img))

    def apply_canny(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_canny(self.processed_img))

    def apply_laplacian(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_laplacian(self.processed_img))

    def apply_averaging(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_averaging(self.processed_img, self.val_blur.get()))

    def apply_median(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_median(self.processed_img, self.val_blur.get()))

    def apply_bilateral(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_bilateral(self.processed_img))

    def apply_zoom_nearest(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_zoom_nearest(self.processed_img, self.zoom_slider.get()))

    def apply_zoom_bilinear(self):
        if self.processed_img is None: return
        self.commit_changes(Filters.apply_zoom_bilinear(self.processed_img, self.zoom_slider.get()))

    def open_face_window(self):
        if self.processed_img is None: return
        face_win = Toplevel(self.root); face_win.title("Face Detection Analysis"); face_win.geometry("700x800"); face_win.configure(bg=self.BG)
        input_frame = tk.Frame(face_win, bg=self.CARD, padx=10, pady=10); input_frame.pack(pady=10, fill="x", padx=20)
        tk.Label(input_frame, text="Expected Faces:", bg=self.CARD, fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        expected_entry = tk.Entry(input_frame, width=10); expected_entry.insert(0, "1"); expected_entry.pack(side="left", padx=5)
        res_frame = tk.Frame(face_win, bg=self.BG); res_frame.pack(pady=5)
        time_label = tk.Label(res_frame, text="Time Taken: ---", bg=self.BG, fg="#60a5fa", font=("Arial", 10)); time_label.pack()
        success_label = tk.Label(res_frame, text="Success Rate: ---", bg=self.BG, fg="#34d399", font=("Arial", 10)); success_label.pack()
        result_panel = tk.Label(face_win, text="[ Detection Result ]", bg="black", bd=2, relief="solid"); result_panel.pack(pady=10, expand=True)

        def perform_detection():
            try:
                res_img, count, time_taken = detect_face_yunet(self.processed_img)
                try:
                    expected = int(expected_entry.get())
                    rate = min((count / expected) * 100, 100.0) if expected > 0 else 0
                except ValueError: rate = 0
                time_label.config(text=f"Time Taken: {time_taken:.4f} seconds")
                success_label.config(text=f"Success Rate: {rate:.1f}% (Found {count})")
                img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]; scale = min(600/w, 400/h, 1.0)
                img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
                img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
                result_panel.config(image=img_tk, text=""); result_panel.image = img_tk
            except Exception as e: print(f"Detection Error: {e}")

        tk.Button(face_win, text="Start Detection", command=perform_detection, bg=self.BTN, fg="white", font=("Arial", 11, "bold"), width=20).pack(pady=10)

    def setup_hist_widget(self, parent, title):
        tk.Label(parent, text=title, bg=self.BG, fg="white", font=("Arial", 12, "bold")).pack()
        fig = Figure(figsize=(5, 2.5), dpi=80, facecolor=self.CARD); ax = fig.add_subplot(111); ax.set_facecolor("#111827")
        canvas = FigureCanvasTkAgg(fig, master=parent); canvas.get_tk_widget().pack(pady=5)
        return fig, ax, canvas

    def update_display(self):
        self.show_image(self.original_img, self.panel_original)
        self.show_image(self.processed_img, self.panel_processed)
        if self.original_img is not None: self.draw_histogram(self.original_img, self.ax1, self.canvas1, "Original")
        if self.processed_img is not None: self.draw_histogram(self.processed_img, self.ax2, self.canvas2, "Processed")

    def show_image(self, img, panel):
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img_rgb.shape[:2]; scale = min(600/w, 350/h, 1.0)
        img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        panel.config(image=img_tk); panel.image = img_tk

    def draw_histogram(self, img, ax, canvas, title):
        ax.clear()
        if img.shape[0] > 1000 or img.shape[1] > 1000:
            sample_img = img[::4, ::4] 
        else:
            sample_img = img
        if len(sample_img.shape) == 2: ax.plot(cv2.calcHist([sample_img], [0], None, [256], [0, 256]), color="white")
        else:
            for i, col in enumerate(['b', 'g', 'r']): ax.plot(cv2.calcHist([sample_img], [i], None, [256], [0, 256]), color=col)
        ax.set_title(title, color="white"); ax.tick_params(colors='white'); canvas.draw()

    def on_frame_configure(self, event):
        content_width = self.scroll_frame.winfo_reqwidth()
        content_height = self.scroll_frame.winfo_reqheight()
    
        canvas_width = self.main_canvas.winfo_width()
        canvas_height = self.main_canvas.winfo_height()

        scroll_w = max(content_width, canvas_width)
        scroll_h = max(content_height, canvas_height)
    
        self.main_canvas.configure(scrollregion=(0, 0, scroll_w, scroll_h))
    def on_canvas_configure(self, event): self.main_canvas.itemconfig(self.scroll_window, width=event.width)
    def on_mousewheel(self, event): self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")