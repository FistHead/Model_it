import customtkinter as ctk
import os, sys, threading, re
from PIL import Image
import pywinstyles

from tkinter import filedialog

ctk.set_appearance_mode("dark")
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


font_path = resource_path("SpaceMono-Bold.ttf")
icon_path = resource_path("Oit.ico")
main_logo = resource_path("Oit.png")
# ctk.FontManager.load_font(font_path)

class CnnFrame(ctk.CTkFrame):
    def __init__(self, master, height=30, corner_radius=20, **kwargs):
        super().__init__(master, corner_radius=corner_radius,border_width=0,fg_color='transparent',height=height, **kwargs)
        for i in range(5):
            self.grid_columnconfigure(i, weight=1)

        self.class_count_input = ctk.CTkEntry(self,placeholder_text="Classes", corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.class_count_input.grid(row=0, column=0, padx=5, pady=4, sticky="ew")

        self.batch_size_input = ctk.CTkEntry(self,placeholder_text="Batch Size",  corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.batch_size_input.grid(row=0, column=1, padx=5, pady=4, sticky="ew")

        self.filters_count_input = ctk.CTkEntry(self,placeholder_text="Filters count", corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.filters_count_input.grid(row=0, column=2, padx=5, pady=4, sticky="ew")

        self.image_size_input = ctk.CTkEntry(self,placeholder_text="Image size", corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.image_size_input.grid(row=0, column=3, padx=5, pady=4, sticky="ew")

        self.color_channels_input = ctk.CTkEntry(self,placeholder_text="Color channels", corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.color_channels_input.grid(row=0, column=4, padx=5, pady=4, sticky="ew")

class LinearFrame(ctk.CTkFrame):
    def __init__(self, master, height=30, corner_radius=20, **kwargs):
        super().__init__(master, corner_radius=corner_radius,border_width=0,fg_color='transparent',height=height, **kwargs)
        for i in range(4):
            self.grid_columnconfigure(i, weight=1)

        self.input_shape_input = ctk.CTkEntry(self,placeholder_text="Input shape", corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.input_shape_input.grid(row=0, column=0, padx=5, pady=4, sticky="ew")

        self.hidden_shape_input = ctk.CTkEntry(self,placeholder_text="hidden shape",  corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.hidden_shape_input.grid(row=0, column=1, padx=5, pady=4, sticky="ew")

        self.output_shape_input = ctk.CTkEntry(self,placeholder_text="output shape", corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.output_shape_input.grid(row=0, column=2, padx=5, pady=4, sticky="ew")

        self.batch_size = ctk.CTkEntry(self,placeholder_text="Batch size", corner_radius=corner_radius,fg_color="#0d0d0d",border_width=0,width=100)
        self.batch_size.grid(row=0, column=3, padx=5, pady=4, sticky="ew")
        # input_shape = 2, hidden_shape = 16, output_shape = 1, batch_size = 2

class CnnEditorFrame(ctk.CTkFrame):
    def __init__(self, master, height=30, corner_radius=20, **kwargs):
        super().__init__(master, corner_radius=corner_radius,border_width=0,fg_color='transparent',height=height, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.add_dataset_button = ctk.CTkButton(self,text = '+',corner_radius=corner_radius,fg_color="#141414",border_width=0)
        self.add_dataset_button.grid(row=0, column=0, padx=(10,0), pady=4, sticky="nsew")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("900x500")
        self.title("model-it")
        self.iconbitmap(icon_path)
        self.configure(fg_color="#0d0d0d")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)


        main_logo_image = ctk.CTkImage(light_image=Image.open(main_logo),
                               dark_image=Image.open(main_logo),
                               size=(30,30))

        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        self.header_frame.columnconfigure(1, weight=1)

        self.logo_label = ctk.CTkLabel(self.header_frame, text="", image=main_logo_image, compound="left",
                                       font=("Arial", 20, "bold"), text_color="#ffffff")
        self.logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")


        self.constructor_container = ctk.CTkFrame(self.header_frame,fg_color='#141414',height=50,corner_radius=50,border_width=1)
        self.constructor_container.grid(column=1, row=0,padx = 10,pady = 10,sticky = 'ew')
        self.constructor_container.columnconfigure(1, weight=1)


        self.optionmenu = ctk.CTkOptionMenu(self.constructor_container,
                                            values=["CNN", "LINEAR"],
                                            width=120, height=30, corner_radius=20, fg_color="#0d0d0d",button_hover_color='#4a4a4a',
                                            button_color="#1a1a1a", dropdown_fg_color="#0d0d0d",command=self.update_mode)
        self.optionmenu.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.cnn_frame = CnnFrame(self.constructor_container,height=30,corner_radius=20)
        # self.cnn_frame.grid(row=0, column=1, padx=(5,15), pady=10, sticky='nsew')

        self.linear_frame = LinearFrame(self.constructor_container,height=30,corner_radius=20)
        # self.linear_frame.grid(row=0, column=1, padx=(5,15), pady=10, sticky='nsew')

        self.main_content = ctk.CTkFrame(self, fg_color="#141414", border_width=1,corner_radius=25)
        self.main_content.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 0), sticky="nsew")
        self.grid_rowconfigure(1, weight=1)

        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(0, weight=1)

        self.info_text = ctk.CTkLabel(self, text="Version: dev_0", fg_color="#0d0d0d")
        self.info_text.grid(row=2, column=0, padx=20, pady=(0,10), sticky="ws")

        self.dataset_container = ctk.CTkFrame(self.main_content, fg_color='#0d0d0d', corner_radius=20, border_width=1)
        self.dataset_container.grid(column=0, row=0, padx=10, pady=10, sticky='nsew')

        self.dataset_container.columnconfigure(0, weight=1)
        self.dataset_container.rowconfigure(0, weight=0)  # Для редактора (верхняя панель)
        self.dataset_container.rowconfigure(1, weight=1)  # Для списка данных (если будет)

        # 2. Создаем и РАЗМЕЩАЕМ фрейм редактора
        self.cnn_editor_frame = CnnEditorFrame(self.dataset_container, corner_radius=20)
        # Обязательно добавляем .grid() и sticky='ew'
        self.cnn_editor_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # 3. Контейнер обучения
        self.train_container = ctk.CTkFrame(self.main_content, fg_color='#0d0d0d', corner_radius=20, border_width=1)
        self.train_container.grid(column=1, row=0, padx=10, pady=10, sticky='nsew')
        # Исправляем индекс колонки (обычно 0, если внутри фрейма один столбец)
        self.train_container.columnconfigure(0, weight=1)

        self.process_mode("CNN")

    def select_cnn_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_path_cnn = path
            self.lbl_path.configure(text=f".../{os.path.basename(path)}")
            try:
                classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                count = len(classes)
                self.lbl_classes_found.configure(text=f"Found {count} classes: {', '.join(classes[:3])}...")
                self.cnn_settings.class_count.delete(0, "end")
                self.cnn_settings.class_count.insert(0, str(count))
            except Exception as e:
                self.lbl_classes_found.configure(text=f"Error reading folder")

    def select_linear_file(self, type_f):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            if type_f == 'x':
                self.dataset_path_linear_x = path
            else:
                self.dataset_path_linear_y = path

    def update_mode(self, *args):
        selected_mode = self.optionmenu.get()
        self.process_mode(selected_mode)

    def process_mode(self, selected_mode):
        self.cnn_frame.grid_forget()
        self.cnn_editor_frame.grid_forget()

        self.linear_frame.grid_forget()

        if selected_mode == "CNN":
            self.cnn_frame.grid(row=0, column=1, padx=(5, 15), pady=10, sticky='nsew')
            self.cnn_editor_frame.grid(row=0, column=0, padx=(5, 15), pady=10, sticky='ew')

        elif selected_mode == "LINEAR":
            self.linear_frame.grid(row=0, column=1, padx=(5, 15), pady=10, sticky='nsew')




app = App()
pywinstyles.change_header_color(app, color="#0d0d0d")
app.mainloop()