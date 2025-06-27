#!/usr/bin/env python3
"""
Create a test image with Vietnamese text for OCR testing
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    # Create a white background image
    width, height = 800, 400
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a system font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Vietnamese text content
    vietnamese_text = """
Hóa đơn bán hàng
Cửa hàng ABC
123 Đường Nguyễn Văn A, Quận 1, TP.HCM
Tel: 028-1234-5678

Ngày: 15/12/2024
Số hóa đơn: HD001234

Sản phẩm:
1. Áo sơ mi trắng - 350,000 VNĐ
2. Quần jean xanh - 450,000 VNĐ  
3. Giày thể thao - 800,000 VNĐ

Tổng cộng: 1,600,000 VNĐ
Thuế VAT (10%): 160,000 VNĐ
Thành tiền: 1,760,000 VNĐ

Cảm ơn quý khách!
    """.strip()
    
    # Draw the text
    y_position = 20
    line_height = 25
    
    for line in vietnamese_text.split('\n'):
        if line.strip():
            draw.text((20, y_position), line, fill='black', font=font)
        y_position += line_height
    
    # Save the image
    output_path = 'test_images/vietnamese_receipt.png'
    img.save(output_path)
    print(f"✅ Test image created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_image()
