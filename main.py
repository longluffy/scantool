import wx
import os
import threading
from pathlib import Path
import sys
import requests
import base64
import time

# Add the ocrServer directory to path to import the client
sys.path.append(os.path.join(os.path.dirname(__file__), 'ocrServer'))
from working_vintern_client import WorkingVinternClient


class OCRApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="✨ Nhận diện văn bản bằng hình ảnh", size=(1000, 700))
        
        # Set modern background color (light gray-blue)
        self.SetBackgroundColour(wx.Colour(248, 250, 252))
        
        self.input_folder = ""
        self.output_folder = ""
        self.image_files = []
        
        # Initialize Vintern client
        self.vintern_client = WorkingVinternClient("http://localhost:8000")
        self.server_ready = False
        
        self.init_ui()
        self.Centre()
        self.Show()
        
        # Check server status on startup
        self.check_server_status()

    def init_ui(self):
        # Main panel
        main_panel = wx.Panel(self)
        main_panel.SetBackgroundColour(wx.Colour(248, 250, 252))
        
        # Main sizer with padding
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add((0, 20), 0)  # Top padding
        
        # Header section
        header_panel = self.create_header_panel(main_panel)
        main_sizer.Add(header_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 30)
        main_sizer.Add((0, 25), 0)  # Spacing
        
        # Content area with horizontal layout
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left panel for folder selection
        left_panel = self.create_folder_panel(main_panel)
        content_sizer.Add(left_panel, 1, wx.EXPAND | wx.RIGHT, 15)
        
        # Right panel for file list
        right_panel = self.create_file_list_panel(main_panel)
        content_sizer.Add(right_panel, 1, wx.EXPAND | wx.LEFT, 15)
        
        main_sizer.Add(content_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 30)
        main_sizer.Add((0, 20), 0)  # Bottom padding
        
        # Action panel
        action_panel = self.create_action_panel(main_panel)
        main_sizer.Add(action_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 30)
        main_sizer.Add((0, 30), 0)  # Bottom padding
        
        main_panel.SetSizer(main_sizer)

    def create_header_panel(self, parent):
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add((0, 20), 0)
        # Main title
        title = wx.StaticText(panel, label="🧠 Nhận diện văn bản bằng hình ảnh")
        title_font = wx.Font(24, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        title.SetForegroundColour(wx.Colour(30, 41, 59))  # Dark blue-gray
        sizer.Add(title, 0, wx.ALIGN_CENTER)
        
        sizer.Add((0, 8), 0)
        
        # Subtitle
        subtitle = wx.StaticText(panel, label="Sử dụng Vintern-1B-v3.5 AI để trích xuất văn bản tiếng Việt từ hình ảnh")
        subtitle_font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        subtitle.SetFont(subtitle_font)
        subtitle.SetForegroundColour(wx.Colour(100, 116, 139))  # Medium gray
        sizer.Add(subtitle, 0, wx.ALIGN_CENTER)
        
        sizer.Add((0, 20), 0)
        panel.SetSizer(sizer)
        
        return panel

    def create_folder_panel(self, parent):
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add((0, 20), 0)
        
        # Panel title
        title = wx.StaticText(panel, label="📁 Folder Configuration")
        title_font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        title.SetForegroundColour(wx.Colour(30, 41, 59))
        sizer.Add(title, 0, wx.LEFT | wx.RIGHT, 20)
        
        sizer.Add((0, 20), 0)
        
        # Input folder section
        input_section = self.create_folder_section(panel, "Input Folder", "📂", "Select folder containing images")
        self.input_btn, self.input_label, input_panel = input_section
        self.input_btn.Bind(wx.EVT_BUTTON, self.on_choose_input)
        sizer.Add(input_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 20)
        
        sizer.Add((0, 25), 0)
        
        # Output folder section
        output_section = self.create_folder_section(panel, "Output Folder", "📁", "Select folder for OCR results")
        self.output_btn, self.output_label, output_panel = output_section
        self.output_btn.Bind(wx.EVT_BUTTON, self.on_choose_output)
        sizer.Add(output_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 20)
        
        sizer.Add((0, 20), 0)
        panel.SetSizer(sizer)
        
        return panel

    def create_folder_section(self, parent, title, icon, placeholder):
        section_panel = wx.Panel(parent)
        section_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Section label
        label = wx.StaticText(section_panel, label=f"{icon} {title}")
        label_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        label.SetFont(label_font)
        label.SetForegroundColour(wx.Colour(71, 85, 105))
        sizer.Add(label, 0, wx.BOTTOM, 8)
        
        # Button
        btn = wx.Button(section_panel, label=f"Choose {title}")
        btn.SetBackgroundColour(wx.Colour(59, 130, 246))  # Blue
        btn.SetForegroundColour(wx.Colour(255, 255, 255))
        btn_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        btn.SetFont(btn_font)
        btn.SetMinSize((200, 35))
        sizer.Add(btn, 0, wx.BOTTOM, 8)
        
        # Path label
        path_label = wx.StaticText(section_panel, label=placeholder)
        path_font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL)
        path_label.SetFont(path_font)
        path_label.SetForegroundColour(wx.Colour(156, 163, 175))
        sizer.Add(path_label, 0, wx.EXPAND)
        
        section_panel.SetSizer(sizer)
        return btn, path_label, section_panel

    def create_file_list_panel(self, parent):
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add((0, 20), 0)
        
        # Panel title
        title = wx.StaticText(panel, label="📋 Image Files Preview")
        title_font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        title.SetForegroundColour(wx.Colour(30, 41, 59))
        sizer.Add(title, 0, wx.LEFT | wx.RIGHT, 20)
        
        sizer.Add((0, 15), 0)
        
        # File count label
        self.file_count_label = wx.StaticText(panel, label="No files loaded")
        count_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.file_count_label.SetFont(count_font)
        self.file_count_label.SetForegroundColour(wx.Colour(100, 116, 139))
        sizer.Add(self.file_count_label, 0, wx.LEFT | wx.RIGHT, 20)
        
        sizer.Add((0, 10), 0)
        
        # File list
        self.file_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.file_list.SetBackgroundColour(wx.Colour(249, 250, 251))
        self.file_list.AppendColumn('Filename', width=200)
        self.file_list.AppendColumn('Size', width=80)
        self.file_list.AppendColumn('Type', width=60)
        
        # Set header colors
        list_font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.file_list.SetFont(list_font)
        
        sizer.Add(self.file_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 20)
        
        sizer.Add((0, 20), 0)
        panel.SetSizer(sizer)
        
        return panel

    def create_action_panel(self, parent):
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(wx.Colour(248, 250, 252))
        
        # Main horizontal sizer
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Server status section (left)
        server_sizer = wx.BoxSizer(wx.VERTICAL)
        
        server_title = wx.StaticText(panel, label="🤖 Palm TEK Server")
        server_title_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        server_title.SetFont(server_title_font)
        server_title.SetForegroundColour(wx.Colour(71, 85, 105))
        server_sizer.Add(server_title, 0, wx.BOTTOM, 5)
        
        self.server_status_label = wx.StaticText(panel, label="🔍 Checking...")
        server_status_font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.server_status_label.SetFont(server_status_font)
        self.server_status_label.SetForegroundColour(wx.Colour(100, 116, 139))
        server_sizer.Add(self.server_status_label, 0)
        
        main_sizer.Add(server_sizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 20)
        
        # Progress section (center)
        progress_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.progress_label = wx.StaticText(panel, label="Ready to process")
        progress_label_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.progress_label.SetFont(progress_label_font)
        self.progress_label.SetForegroundColour(wx.Colour(100, 116, 139))
        progress_sizer.Add(self.progress_label, 0, wx.BOTTOM, 5)
        
        self.progress = wx.Gauge(panel, range=100, size=(300, 8))
        self.progress.SetBackgroundColour(wx.Colour(229, 231, 235))
        progress_sizer.Add(self.progress, 0, wx.EXPAND)
        
        main_sizer.Add(progress_sizer, 1, wx.ALIGN_CENTER_VERTICAL)
        
        main_sizer.Add((20, 0), 0)
        
        # Start button (right)
        self.start_btn = wx.Button(panel, label="🚀 Start OCR Processing")
        self.start_btn.SetBackgroundColour(wx.Colour(16, 185, 129))  # Green
        self.start_btn.SetForegroundColour(wx.Colour(255, 255, 255))
        start_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.start_btn.SetFont(start_font)
        self.start_btn.SetMinSize((200, 45))
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start_ocr)
        main_sizer.Add(self.start_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        
        panel.SetSizer(main_sizer)
        return panel

    def on_choose_input(self, event):
        dlg = wx.DirDialog(self, "Choose Input Folder with Images")
        if dlg.ShowModal() == wx.ID_OK:
            self.input_folder = dlg.GetPath()
            self.input_label.SetLabel(f"📂 {self.input_folder}")
            self.input_label.SetForegroundColour(wx.Colour(22, 163, 74))  # Green
            self.refresh_file_list()
        dlg.Destroy()

    def on_choose_output(self, event):
        dlg = wx.DirDialog(self, "Choose Output Folder for OCR Results")
        if dlg.ShowModal() == wx.ID_OK:
            self.output_folder = dlg.GetPath()
            self.output_label.SetLabel(f"📁 {self.output_folder}")
            self.output_label.SetForegroundColour(wx.Colour(22, 163, 74))  # Green
        dlg.Destroy()
    
    def refresh_file_list(self):
        """Refresh the file list when input folder changes"""
        self.file_list.DeleteAllItems()
        self.image_files = []
        
        if not self.input_folder:
            self.file_count_label.SetLabel("No files loaded")
            return
        
        # Supported image formats
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp')
        
        try:
            files = [f for f in os.listdir(self.input_folder) 
                    if f.lower().endswith(supported_formats)]
            files.sort()  # Sort alphabetically
            
            self.image_files = files
            
            # Update file count
            count = len(files)
            if count == 0:
                self.file_count_label.SetLabel("No image files found")
                self.file_count_label.SetForegroundColour(wx.Colour(239, 68, 68))  # Red
            else:
                self.file_count_label.SetLabel(f"{count} image file{'s' if count != 1 else ''} found")
                self.file_count_label.SetForegroundColour(wx.Colour(22, 163, 74))  # Green
            
            # Populate list
            for i, filename in enumerate(files):
                file_path = os.path.join(self.input_folder, filename)
                
                # Get file size
                try:
                    size_bytes = os.path.getsize(file_path)
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes // 1024} KB"
                    else:
                        size_str = f"{size_bytes // (1024 * 1024)} MB"
                except:
                    size_str = "Unknown"
                
                # Get file extension
                ext = Path(filename).suffix.upper()[1:]  # Remove the dot
                
                # Add to list
                index = self.file_list.InsertItem(i, filename)
                self.file_list.SetItem(index, 1, size_str)
                self.file_list.SetItem(index, 2, ext)
                
                # Alternate row colors for better readability
                if i % 2 == 1:
                    self.file_list.SetItemBackgroundColour(index, wx.Colour(250, 250, 250))
                    
        except Exception as e:
            self.file_count_label.SetLabel(f"Error reading folder: {str(e)}")
            self.file_count_label.SetForegroundColour(wx.Colour(239, 68, 68))  # Red

    def on_start_ocr(self, event):
        if not self.input_folder or not self.output_folder:
            wx.MessageBox(
                "Please select both input and output folders before starting.", 
                "Missing Folders", 
                wx.ICON_WARNING | wx.OK
            )
            return
        
        if not self.image_files:
            wx.MessageBox(
                "No image files found in the input folder.", 
                "No Files", 
                wx.ICON_WARNING | wx.OK
            )
            return
        
        # Check if Vintern server is ready
        if not self.server_ready:
            wx.MessageBox(
                "Vintern server is not ready yet. Please wait for the model to load.", 
                "Server Not Ready", 
                wx.ICON_WARNING | wx.OK
            )
            return

        # Disable start button during processing
        self.start_btn.Enable(False)
        self.start_btn.SetLabel("⏳ Processing...")
        
        # Start processing in background thread
        threading.Thread(target=self.run_ocr_process, daemon=True).start()

    def run_ocr_process(self):
        total = len(self.image_files)
        wx.CallAfter(self.progress.SetRange, total)
        wx.CallAfter(self.progress_label.SetLabel, "Starting OCR processing...")

        processed = 0
        errors = 0

        for idx, filename in enumerate(self.image_files, start=1):
            # Update progress
            wx.CallAfter(self.progress_label.SetLabel, f"Processing {filename} ({idx}/{total})")
            
            image_path = os.path.join(self.input_folder, filename)
            text = self.ocr_image(image_path)
            
            # Save result (append to existing file)
            output_file = os.path.join(self.output_folder, f"{os.path.splitext(filename)[0]}.txt")
            try:
                # Check if file exists and read existing content
                existing_content = ""
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_content = f.read().strip()
                
                # Prepare new content with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Format the new OCR result
                new_entry = f"\n\n{'='*60}\n🔍 OCR Analysis - {timestamp}\n{'='*60}\n{text}"
                
                # Write combined content (existing + new)
                with open(output_file, 'w', encoding='utf-8') as f:
                    if existing_content:
                        f.write(existing_content + new_entry)
                    else:
                        f.write(f"📄 OCR Results for: {filename}\n{new_entry}")
                
                processed += 1
            except Exception as e:
                errors += 1
                print(f"Error saving {output_file}: {e}")

            wx.CallAfter(self.progress.SetValue, idx)

        # Show completion message
        if errors == 0:
            message = f"✅ OCR completed successfully!\n\nProcessed {processed} files.\nResults appended to existing text files with timestamps."
            wx.CallAfter(wx.MessageBox, message, "Success", wx.ICON_INFORMATION)
        else:
            message = f"OCR completed with {errors} error(s).\n\nProcessed {processed} files successfully.\nResults appended to existing text files with timestamps."
            wx.CallAfter(wx.MessageBox, message, "Completed with Errors", wx.ICON_WARNING)
        
        # Reset UI
        wx.CallAfter(self.progress_label.SetLabel, "Ready to process")
        wx.CallAfter(self.start_btn.Enable, True)
        wx.CallAfter(self.start_btn.SetLabel, "🚀 Start OCR Processing")

    def ocr_image(self, image_path):
        """Process image using Vintern-1B-v3.5 server for Vietnamese OCR"""
        try:
            if not self.server_ready:
                return "Error: Vintern server is not ready. Please wait for model to load."
            
            # Use the exact method from Test 2 that works perfectly
            ocr_prompt = """Hãy chuyển đổi chính xác tất cả văn bản trong ảnh thành text, giữ nguyên định dạng và bố cục. Chỉ trả về văn bản được trích xuất, không thêm giải thích hay bình luận gì khác."""
            
            text = self.vintern_client.chat(ocr_prompt, image_path=image_path, max_tokens=2048)
            
            if text and text.strip():
                return text.strip()
            else:
                return "No text detected in image"
                
        except Exception as e:
            return f"Error processing image with Vintern server: {str(e)}"

    def check_server_status(self):
        """Check if Vintern server is running and model is loaded"""
        try:
            health = self.vintern_client.is_healthy()
            if health.get("status") == "healthy" and health.get("model_loaded"):
                self.server_ready = True
                print("✅ Vintern server is ready!")
                wx.CallAfter(self.update_server_status, "✅ Server Ready", wx.Colour(22, 163, 74))
            elif health.get("status") == "healthy":
                print("⏳ Server running, waiting for model...")
                wx.CallAfter(self.update_server_status, "⏳ Loading Model...", wx.Colour(245, 158, 11))
                # Start a thread to wait for model loading
                threading.Thread(target=self.wait_for_model_loading, daemon=True).start()
            else:
                print("❌ Server not available")
                wx.CallAfter(self.update_server_status, "❌ Server Offline", wx.Colour(239, 68, 68))
        except Exception as e:
            print(f"❌ Failed to check server: {e}")
            wx.CallAfter(self.update_server_status, "❌ Server Offline", wx.Colour(239, 68, 68))
    
    def wait_for_model_loading(self):
        """Wait for model to finish loading"""
        success = self.vintern_client.wait_for_model(max_wait=120)
        if success:
            self.server_ready = True
            wx.CallAfter(self.update_server_status, "✅ Server Ready", wx.Colour(22, 163, 74))
        else:
            wx.CallAfter(self.update_server_status, "❌ Model Load Failed", wx.Colour(239, 68, 68))
    
    def update_server_status(self, message, color):
        """Update server status in the UI"""
        if hasattr(self, 'server_status_label'):
            self.server_status_label.SetLabel(message)
            self.server_status_label.SetForegroundColour(color)


if __name__ == "__main__":
    app = wx.App(False)
    OCRApp()
    app.MainLoop()
