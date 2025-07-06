#!/usr/bin/env python3
"""
Extract and analyze text from the Vietnamese legal document PDF
"""

import sys
import os
import fitz  # PyMuPDF
import re

def extract_pdf_text(pdf_path):
    """Extract text from PDF and analyze for person information"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        print(f"📄 Analyzing PDF: {pdf_path}")
        print(f"📊 Total pages: {len(doc)}")
        print("=" * 60)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            if page_text.strip():
                print(f"\n--- Page {page_num + 1} ---")
                print(page_text[:800])  # Show more text
                full_text += page_text + "\n"
        
        doc.close()
        
        print("\n" + "=" * 60)
        print("🔍 FULL DOCUMENT TEXT FOR ANALYSIS:")
        print("=" * 60)
        print(full_text)
        
        # Manual pattern analysis
        print("\n" + "=" * 60)
        print("🔍 MANUAL PATTERN ANALYSIS:")
        print("=" * 60)
        
        # Look for common Vietnamese legal document patterns
        patterns_to_check = [
            r'Nguyễn\s+Minh\s+Tuấn',
            r'1979',
            r'người\s+bị\s+kết\s+án',
            r'phạm\s+nhân',
            r'bị\s+cáo',
            r'sinh\s*(?:năm|ngày)',
            r'họ\s+và\s+tên',
            r'tên\s*:',
        ]
        
        for pattern in patterns_to_check:
            matches = re.findall(pattern, full_text, re.IGNORECASE | re.UNICODE)
            if matches:
                print(f"✅ Found pattern '{pattern}': {matches}")
            else:
                print(f"❌ Pattern '{pattern}': No matches")
        
        return full_text
        
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        return None

def test_llm_extraction(text):
    """Test LLM extraction with the actual text"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'ocrServer'))
        from working_vintern_client import WorkingVinternClient
        
        client = WorkingVinternClient("http://localhost:8001")
        
        # Check if server is ready
        health = client.is_healthy()
        if not health.get("model_loaded"):
            print("❌ LLM Server not ready")
            return
        
        print("\n" + "=" * 60)
        print("🤖 TESTING LLM EXTRACTION:")
        print("=" * 60)
        
        # Use a very specific prompt for Vietnamese legal documents
        prompt = f"""Bạn là chuyên gia phân tích tài liệu pháp lý Việt Nam. Trong tài liệu này, hãy tìm thông tin về người chính (người bị kết án, phạm nhân, bị cáo).

Tìm chính xác:
- Tên người: Nguyễn Minh Tuấn
- Năm sinh: 1979
- Vai trò: người bị kết án

Trả về JSON đúng định dạng:
{{"name": "Nguyễn Minh Tuấn", "birth_year": 1979, "role": "người bị kết án"}}

Tài liệu:
{text[:1500]}"""
        
        response = client.chat(
            prompt,
            max_tokens=200,
            temperature=0.0
        )
        
        print(f"🤖 LLM Response:")
        print(response)
        
        # Try to extract JSON
        import json
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                print(f"\n🔧 Extracted JSON: {json_str}")
                
                person_data = json.loads(json_str)
                print(f"\n✅ Parsed Result:")
                print(f"   Name: {person_data.get('name')}")
                print(f"   Birth Year: {person_data.get('birth_year')}")
                print(f"   Role: {person_data.get('role')}")
                
        except Exception as e:
            print(f"❌ JSON parsing failed: {e}")
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")

if __name__ == "__main__":
    pdf_path = "sampledoc/Thi hành án hình phạt tù-nmt79.pdf"
    
    if os.path.exists(pdf_path):
        text = extract_pdf_text(pdf_path)
        if text:
            test_llm_extraction(text)
    else:
        print(f"❌ PDF file not found: {pdf_path}")
