import wx
import os
import cv2
import pytesseract
import threading


class OCRApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="OCR Image Folder App", size=(700, 400))
        panel = wx.Panel(self)

        self.input_folder = ""
        self.output_folder = ""

        # UI Elements
        vbox = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(panel, label="🧠 OCR Image Folder App")
        title_font = wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        title.SetFont(title_font)
        vbox.Add(title, 0, wx.ALIGN_CENTER | wx.TOP, 20)

        self.input_btn = wx.Button(panel, label="Choose Input Folder")
        self.input_btn.Bind(wx.EVT_BUTTON, self.on_choose_input)
        vbox.Add(self.input_btn, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        self.input_label = wx.StaticText(panel, label="📂 No input folder selected")
        vbox.Add(self.input_label, 0, wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, 10)

        self.output_btn = wx.Button(panel, label="Choose Output Folder")
        self.output_btn.Bind(wx.EVT_BUTTON, self.on_choose_output)
        vbox.Add(self.output_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        self.output_label = wx.StaticText(panel, label="📁 No output folder selected")
        vbox.Add(self.output_label, 0, wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, 10)

        self.start_btn = wx.Button(panel, label="Start OCR")
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start_ocr)
        vbox.Add(self.start_btn, 0, wx.ALIGN_CENTER | wx.ALL, 15)

        self.progress = wx.Gauge(panel, range=100, size=(500, 20))
        vbox.Add(self.progress, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        panel.SetSizer(vbox)
        self.Centre()
        self.Show()

    def on_choose_input(self, event):
        dlg = wx.DirDialog(self, "Choose Input Folder")
        if dlg.ShowModal() == wx.ID_OK:
            self.input_folder = dlg.GetPath()
            self.input_label.SetLabel(f"📂 {self.input_folder}")
        dlg.Destroy()

    def on_choose_output(self, event):
        dlg = wx.DirDialog(self, "Choose Output Folder")
        if dlg.ShowModal() == wx.ID_OK:
            self.output_folder = dlg.GetPath()
            self.output_label.SetLabel(f"📁 {self.output_folder}")
        dlg.Destroy()

    def on_start_ocr(self, event):
        if not self.input_folder or not self.output_folder:
            wx.MessageBox("Please select both input and output folders.", "Error", wx.ICON_ERROR)
            return

        threading.Thread(target=self.run_ocr_process, daemon=True).start()

    def run_ocr_process(self):
        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(supported)]

        total = len(files)
        if total == 0:
            wx.CallAfter(wx.MessageBox, "No image files found.", "Warning", wx.ICON_WARNING)
            return

        wx.CallAfter(self.progress.SetRange, total)

        for idx, filename in enumerate(files, start=1):
            image_path = os.path.join(self.input_folder, filename)
            text = self.ocr_image(image_path)
            output_file = os.path.join(self.output_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)

            wx.CallAfter(self.progress.SetValue, idx)

        wx.CallAfter(wx.MessageBox, "OCR completed successfully!", "Done", wx.ICON_INFORMATION)

    def ocr_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            return pytesseract.image_to_string(image, lang='vie')
        except Exception as e:
            return f"Error processing image: {e}"


if __name__ == "__main__":
    app = wx.App(False)
    OCRApp()
    app.MainLoop()
