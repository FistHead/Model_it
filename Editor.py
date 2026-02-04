import customtkinter as ctk
import os, sys, threading, re
from PIL import Image
import pywinstyles
from tkinter import filedialog, messagebox
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# --- Инициализация путей (без изменений) ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


icon_path = resource_path("Oit.ico")
main_logo = resource_path("Oit.png")


# --- UI Компоненты ---

class CnnFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color='transparent', **kwargs)
        for i in range(5): self.grid_columnconfigure(i, weight=1)
        self.class_count_input = self._add_entry("Classes", 0)
        self.batch_size_input = self._add_entry("Batch Size", 1)
        self.filters_count_input = self._add_entry("Filters", 2)
        self.image_size_input = self._add_entry("Size", 3)
        self.color_channels_input = self._add_entry("Channels", 4)

    def _add_entry(self, placeholder, col):
        e = ctk.CTkEntry(self, placeholder_text=placeholder, corner_radius=15, fg_color="#0d0d0d", border_width=0,
                         height=30)
        e.grid(row=0, column=col, padx=5, pady=4, sticky="ew")
        return e


class LinearFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color='transparent', **kwargs)
        for i in range(4): self.grid_columnconfigure(i, weight=1)
        self.input_shape_input = self._add_entry("In Shape", 0)
        self.hidden_shape_input = self._add_entry("Hidden", 1)
        self.output_shape_input = self._add_entry("Out Shape", 2)
        self.batch_size_input = self._add_entry("Batch", 3)

    def _add_entry(self, placeholder, col):
        e = ctk.CTkEntry(self, placeholder_text=placeholder, corner_radius=15, fg_color="#0d0d0d", border_width=0,
                         height=30)
        e.grid(row=0, column=col, padx=5, pady=4, sticky="ew")
        return e


class CnnEditorFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color='transparent', **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.btn = ctk.CTkButton(self, text='📁 SELECT DATASET', corner_radius=20, fg_color="#141414", border_width=1,
                                 border_color='#474747', hover_color="#222222", command=self.select_cnn_folder,
                                 height=35)
        self.btn.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.scroll = ctk.CTkScrollableFrame(self, corner_radius=20, fg_color="#0d0d0d", border_width=1,
                                             border_color='#333')
        self.scroll.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.lbl = ctk.CTkLabel(self.scroll, text="Dataset not selected", text_color="gray", font=("Arial", 12))
        self.lbl.pack(padx=10, pady=10)

    def select_cnn_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.master.master.master.dataset_path = path
            classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            self.master.master.master.classes_list = classes
            self.lbl.configure(text='\n'.join([f'📁 {c}' for c in classes]), text_color="white")
            self.master.master.master.cnn_frame.class_count_input.delete(0, "end")
            self.master.master.master.cnn_frame.class_count_input.insert(0, str(len(classes)))


# --- НОВЫЙ ФРЕЙМ ДЛЯ LINEAR ---
class LinearEditorFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color='transparent', **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.btn_x = ctk.CTkButton(self, text='📄 SELECT X (DATA)', corner_radius=20, fg_color="#141414", border_width=1,
                                   border_color='#474747', command=lambda: self.select_file('x'), height=35)
        self.btn_x.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.btn_y = ctk.CTkButton(self, text='📄 SELECT Y (LABELS)', corner_radius=20, fg_color="#141414",
                                   border_width=1, border_color='#474747', command=lambda: self.select_file('y'),
                                   height=35)
        self.btn_y.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.status_box = ctk.CTkTextbox(self, corner_radius=20, fg_color="#0d0d0d", border_width=1,
                                         border_color='#333', height=150)
        self.status_box.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.status_box.insert("0.0", "Files not selected...")

    def select_file(self, mode):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            if mode == 'x':
                self.master.master.master.dataset_path_linear_x = path
            else:
                self.master.master.master.dataset_path_linear_y = path

            x_status = os.path.basename(getattr(self.master.master.master, 'dataset_path_linear_x', 'None'))
            y_status = os.path.basename(getattr(self.master.master.master, 'dataset_path_linear_y', 'None'))

            self.status_box.delete("0.0", "end")
            self.status_box.insert("0.0", f"X: {x_status}\nY: {y_status}")


# --- ОКНО ПРЕДСКАЗАНИЯ ---
class PredictWindow(ctk.CTkToplevel):
    def __init__(self, parent, mode, model, extra_data=None):
        super().__init__(parent)
        self.geometry("400x300")
        self.title("Prediction")
        self.model = model
        self.mode = mode
        self.extra_data = extra_data  # classes for CNN
        self.parent = parent

        self.grid_columnconfigure(0, weight=1)
        self.configure(fg_color="#0d0d0d")

        if self.mode == "LINEAR":
            self.lbl = ctk.CTkLabel(self, text="Enter values separated by comma:", pady=10)
            self.lbl.pack()
            self.input_val = ctk.CTkEntry(self, width=300, placeholder_text="e.g. 1.2, 3.4, 5.0")
            self.input_val.pack(pady=10)
            self.btn = ctk.CTkButton(self, text="Get Result", command=self.predict_linear, fg_color="#141414",
                                     border_width=1)
            self.btn.pack(pady=10)
        else:
            self.btn = ctk.CTkButton(self, text="Select Image", command=self.predict_cnn, fg_color="#141414",
                                     border_width=1)
            self.btn.pack(expand=True)

        self.result_lbl = ctk.CTkLabel(self, text="", font=("Arial", 16, "bold"))
        self.result_lbl.pack(pady=20)

    def predict_linear(self):
        try:
            raw = self.input_val.get()
            vals = [float(x.strip()) for x in raw.split(",")]
            data = np.array([vals])
            res = self.model.predict(data)
            self.result_lbl.configure(text=f"Result: {res[0]}", text_color="#3498db")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def predict_cnn(self):
        path = filedialog.askopenfilename()
        if path:
            res, acc = self.parent.cnn_ref.predict(path, self.model, self.extra_data)
            self.result_lbl.configure(text=f"{res} ({acc:.1f}%)", text_color="#3498db")


class TrainerFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, corner_radius=25, fg_color="#0d0d0d", border_width=1, border_color="#333", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.name_entry = ctk.CTkEntry(self, placeholder_text="Model Name", corner_radius=15, fg_color="#141414",
                                       border_color='#474747', border_width=1, height=35)
        self.name_entry.grid(row=0, column=0, padx=20, pady=(25, 5), sticky="ew")
        self.num_of_epochs = ctk.CTkLabel(self, text="EPOCH: 0 / 0", font=("Arial", 15, "bold"))
        self.num_of_epochs.grid(row=1, column=0, pady=15)
        self.epochs_entry = ctk.CTkEntry(self, placeholder_text="Epochs Count", corner_radius=15, fg_color="#141414",
                                         border_color='#474747', border_width=1, height=35)
        self.epochs_entry.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        self.btn_train = ctk.CTkButton(self, text="TRAIN", corner_radius=20, fg_color="#141414", border_width=1,
                                       border_color='#474747', hover_color="#222222", font=("Arial", 13, "bold"),
                                       height=40, command=lambda: self.master.master.start_training_thread())
        self.btn_train.grid(row=3, column=0, padx=20, pady=15, sticky="ew")
        self.btn_pred = ctk.CTkButton(self, text="PREDICT", corner_radius=20, fg_color="#141414", border_width=1,
                                      border_color='#474747', hover_color="#222222", font=("Arial", 13, "bold"),
                                      height=40, command=lambda: self.master.master.start_predict())
        self.btn_pred.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("900x550")
        self.title("model-it")
        self.iconbitmap(icon_path)
        self.configure(fg_color="#0d0d0d")

        # Данные
        self.dataset_path = None
        self.dataset_path_linear_x = None
        self.dataset_path_linear_y = None
        self.classes_list = []
        self.active_model = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header UI (без изменений)
        logo_img = ctk.CTkImage(Image.open(main_logo), size=(30, 30))
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="ew", padx=15, pady=10)
        self.header.columnconfigure(1, weight=1)
        ctk.CTkLabel(self.header, text="", image=logo_img).grid(row=0, column=0, padx=10)
        self.param_cont = ctk.CTkFrame(self.header, fg_color='#141414', corner_radius=50, border_width=1,
                                       border_color="#333")
        self.param_cont.grid(row=0, column=1, sticky="ew", padx=10)
        self.param_cont.columnconfigure(1, weight=1)
        self.mode_sel = ctk.CTkOptionMenu(self.param_cont, values=["CNN", "LINEAR"], corner_radius=20,
                                          fg_color="#0d0d0d", button_color="#1a1a1a", width=100,
                                          command=self.update_mode)
        self.mode_sel.grid(row=0, column=0, padx=15, pady=10)

        self.cnn_frame = CnnFrame(self.param_cont)
        self.linear_frame = LinearFrame(self.param_cont)

        # Main Content UI
        self.main_content = ctk.CTkFrame(self, fg_color="#141414", corner_radius=25, border_width=1,
                                         border_color="#333")
        self.main_content.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
        self.main_content.columnconfigure(0, weight=2)
        self.main_content.columnconfigure(1, weight=1)
        self.main_content.rowconfigure(0, weight=1)

        self.ds_container = ctk.CTkFrame(self.main_content, fg_color='#0d0d0d', corner_radius=20, border_width=1,
                                         border_color="#333")
        self.ds_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.ds_container.grid_columnconfigure(0, weight=1)
        self.ds_container.grid_rowconfigure(0, weight=1)

        self.cnn_editor = CnnEditorFrame(self.ds_container)
        self.linear_editor = LinearEditorFrame(self.ds_container)

        self.trainer = TrainerFrame(self.main_content)
        self.trainer.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.update_mode("CNN")

    def update_mode(self, mode):
        self.cnn_frame.grid_forget()
        self.linear_frame.grid_forget()
        self.cnn_editor.grid_forget()
        self.linear_editor.grid_forget()

        if mode == "CNN":
            self.cnn_frame.grid(row=0, column=1, sticky="ew", padx=10)
            self.cnn_editor.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        else:
            self.linear_frame.grid(row=0, column=1, sticky="ew", padx=10)
            self.linear_editor.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def start_training_thread(self):
        threading.Thread(target=self.train_logic, daemon=True).start()

    def train_logic(self):
        try:
            from modelMaker import CNN as ModelCNN, LINEAR as ModelLinear
            mode = self.mode_sel.get()
            epochs = int(self.trainer.epochs_entry.get() or 10)

            if mode == "CNN" and self.dataset_path:
                cnn = ModelCNN(classes=len(self.classes_list),
                               batch_size=int(self.cnn_frame.batch_size_input.get() or 8),
                               image_w=int(self.cnn_frame.image_size_input.get() or 32),
                               image_h=int(self.cnn_frame.image_size_input.get() or 32))
                self.active_model = cnn.build_model()
                ds = cnn.create_dataset(self.dataset_path)
                self.cnn_ref = cnn
                self.active_model.fit(ds, epochs=epochs, callbacks=[self.get_cb(epochs)])

            elif mode == "LINEAR" and self.dataset_path_linear_x:
                in_s = int(self.linear_frame.input_shape_input.get() or 1)
                lin = ModelLinear(input_shape=in_s,
                                  hidden_shape=int(self.linear_frame.hidden_shape_input.get() or 16),
                                  output_shape=int(self.linear_frame.output_shape_input.get() or 1),
                                  batch_size=int(self.linear_frame.batch_size_input.get() or 8))
                self.active_model = lin.build_model()
                x = lin.create_dataset(self.dataset_path_linear_x, in_s)
                y = lin.create_dataset(self.dataset_path_linear_y, int(self.linear_frame.output_shape_input.get() or 1))
                self.active_model.fit(x, y, epochs=epochs, callbacks=[self.get_cb(epochs)])

            if self.trainer.name_entry.get(): self.active_model.save(f"{self.trainer.name_entry.get()}.keras")
            self.trainer.num_of_epochs.configure(text="TRAINING DONE")
        except Exception as e:
            self.trainer.num_of_epochs.configure(text="ERROR")
            print(f"Train error: {e}")

    def get_cb(self, total):
        class GuiCB(tf.keras.callbacks.Callback):
            def __init__(self, lbl): self.lbl = lbl

            def on_epoch_end(self, epoch, logs=None):
                self.lbl.configure(text=f"EPOCH: {epoch + 1} / {total}")

        return GuiCB(self.trainer.num_of_epochs)

    def start_predict(self):
        if not self.active_model:
            messagebox.showwarning("Warning", "Train model first!")
            return

        mode = self.mode_sel.get()
        PredictWindow(self, mode, self.active_model, self.classes_list if mode == "CNN" else None)


app = App()
pywinstyles.change_header_color(app, color="#0d0d0d")
app.mainloop()