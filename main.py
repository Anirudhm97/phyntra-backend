import os
import json
import hashlib
import tempfile
from datetime import datetime
from typing import List, Optional, Dict
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import sqlite3
from contextlib import contextmanager
import base64
from PIL import Image
from pdf2image import convert_from_path
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Phyntra API",
    description="AI-powered invoice processing system",
    version="1.0.0"
)

# CORS middleware - FIXED VERSION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed to False to fix CORS
    allow_methods=["*"],
    allow_headers=["*"],
)

# Additional CORS middleware for extra compatibility
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response

# Handle preflight OPTIONS requests
@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Configuration
class Config:
    DATABASE_FILE = "phyntra_invoices.db"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
    OPENAI_MODEL = "gpt-4o"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for data validation
class InvoiceItem(BaseModel):
    Items: str = ""
    HSN: str = ""
    Qty: str = ""
    Rate: str = ""
    Taxable_Value: str = ""
    SGST: str = ""
    CGST: str = ""
    IGST: str = ""

class InvoiceData(BaseModel):
    Invoice_Number: str = ""
    Invoice_Date: str = ""
    Vendor_Name: str = ""
    Vendor_GSTIN: str = ""
    Buyer_Name: str = ""
    Buyer_GSTIN: str = ""
    Items: List[InvoiceItem] = []

class ProcessingResponse(BaseModel):
    success: bool
    data: Optional[InvoiceData] = None
    message: str
    file_hash: str

# Database functions
@contextmanager
def get_db():
    conn = sqlite3.connect(Config.DATABASE_FILE)
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize the database with required tables"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    invoice_number TEXT,
                    invoice_date TEXT,
                    vendor_name TEXT,
                    vendor_gstin TEXT,
                    buyer_name TEXT,
                    buyer_gstin TEXT,
                    items TEXT,
                    status TEXT DEFAULT 'pending',
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    approved_at TIMESTAMP
                )
            ''')
            conn.commit()
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# Utility functions
def get_file_hash(content: bytes) -> str:
    """Generate MD5 hash for file content"""
    return hashlib.md5(content).hexdigest()

def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )
    return True

async def process_image_safely(file_path: str) -> Optional[str]:
    """Convert file to base64 image"""
    try:
        if file_path.lower().endswith('.pdf'):
            # Convert PDF to image
            pages = convert_from_path(file_path, dpi=150, last_page=1)
            if not pages:
                return None
            image = pages[0]
        else:
            image = Image.open(file_path)
        
        # Convert to base64
        import io
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        return img_base64
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return None

# AI Processing with OpenAI
EXTRACTION_PROMPT = """
You are an expert in Indian invoice processing. Extract data from this invoice image and return ONLY valid JSON.

Extract these fields exactly:
- Invoice_Number
- Invoice_Date (format: DD-MM-YYYY or DD/MM/YYYY)
- Vendor_Name
- Vendor_GSTIN
- Buyer_Name  
- Buyer_GSTIN
- Items (array of objects with: Items, HSN, Qty, Rate, Taxable_Value, SGST, CGST, IGST)

Return only this JSON structure with no additional text:
{
  "Invoice_Number": "",
  "Invoice_Date": "",
  "Vendor_Name": "",
  "Vendor_GSTIN": "",
  "Buyer_Name": "",
  "Buyer_GSTIN": "",
  "Items": [
    {
      "Items": "",
      "HSN": "",
      "Qty": "",
      "Rate": "",
      "Taxable_Value": "",
      "SGST": "",
      "CGST": "",
      "IGST": ""
    }
  ]
}
"""

async def extract_invoice_data(image_base64: str) -> Optional[InvoiceData]:
    """Extract invoice data using OpenAI GPT-4o"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {
                    "role": "system", 
                    "content": EXTRACTION_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract data from this invoice image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean JSON response
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        data = json.loads(content)
        return InvoiceData(**data)
        
    except openai.RateLimitError:
        logger.error("OpenAI rate limit exceeded")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    except openai.AuthenticationError:
        logger.error("OpenAI authentication failed")
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    except Exception as e:
        logger.error(f"AI extraction error: {e}")
        return None

def save_invoice_to_db(file_hash: str, filename: str, invoice_data: InvoiceData, status: str = "pending") -> bool:
    """Save invoice data to database"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO invoices (
                    file_hash, filename, invoice_number, invoice_date, vendor_name,
                    vendor_gstin, buyer_name, buyer_gstin, items, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_hash, filename, invoice_data.Invoice_Number,
                invoice_data.Invoice_Date, invoice_data.Vendor_Name,
                invoice_data.Vendor_GSTIN, invoice_data.Buyer_Name,
                invoice_data.Buyer_GSTIN, 
                json.dumps([item.dict() for item in invoice_data.Items]),
                status
            ))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Database save error: {e}")
        return False

def check_duplicate(file_hash: str) -> bool:
    """Check if file was already processed"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM invoices WHERE file_hash = ?", (file_hash,))
            return cursor.fetchone()[0] > 0
    except Exception as e:
        logger.error(f"Duplicate check error: {e}")
        return False

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()

@app.get("/")
async def root():
    """Health check endpoint"""
    return JSONResponse(
        content={"message": "Phyntra API is running", "status": "healthy", "version": "1.0.0"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/api/process-invoice")
async def process_invoice(file: UploadFile = File(...)):
    """Process uploaded invoice file"""
    
    try:
        # Validate file
        validate_file(file)
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Generate file hash
        file_hash = get_file_hash(content)
        
        # Check for duplicates
        if check_duplicate(file_hash):
            return JSONResponse(
                content={
                    "success": False,
                    "message": "File already processed",
                    "file_hash": file_hash,
                    "data": None
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        try:
            # Process image
            image_base64 = await process_image_safely(temp_path)
            if not image_base64:
                raise HTTPException(status_code=400, detail="Failed to process image")
            
            # Extract data using AI
            invoice_data = await extract_invoice_data(image_base64)
            if not invoice_data:
                raise HTTPException(status_code=500, detail="Failed to extract invoice data")
            
            # Save to database
            if not save_invoice_to_db(file_hash, file.filename, invoice_data):
                raise HTTPException(status_code=500, detail="Failed to save invoice data")
            
            logger.info(f"Successfully processed invoice: {file.filename}")
            
            return JSONResponse(
                content={
                    "success": True,
                    "data": invoice_data.dict(),
                    "message": "Invoice processed successfully",
                    "file_hash": file_hash
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}",
                "file_hash": "",
                "data": None
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

@app.post("/api/approve-invoice/{file_hash}")
async def approve_invoice(file_hash: str):
    """Approve processed invoice"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE invoices SET status = 'approved', approved_at = ? WHERE file_hash = ?",
                (datetime.now().isoformat(), file_hash)
            )
            conn.commit()
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Invoice not found")
            
            return JSONResponse(
                content={"success": True, "message": "Invoice approved successfully"},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
    except Exception as e:
        logger.error(f"Approval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve invoice")

@app.get("/api/invoices")
async def get_invoices(status: Optional[str] = None):
    """Get all invoices, optionally filtered by status"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute(
                    "SELECT * FROM invoices WHERE status = ? ORDER BY processed_at DESC",
                    (status,)
                )
            else:
                cursor.execute("SELECT * FROM invoices ORDER BY processed_at DESC")
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            invoices = []
            for row in rows:
                invoice_dict = dict(zip(columns, row))
                # Parse items JSON
                if invoice_dict['items']:
                    try:
                        invoice_dict['items'] = json.loads(invoice_dict['items'])
                    except:
                        invoice_dict['items'] = []
                invoices.append(invoice_dict)
            
            return JSONResponse(
                content={"success": True, "invoices": invoices},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
            
    except Exception as e:
        logger.error(f"Get invoices error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve invoices")

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get counts by status
            cursor.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM invoices 
                GROUP BY status
            """)
            
            stats = {"total": 0}
            for row in cursor.fetchall():
                stats[row[0]] = row[1]
                stats["total"] += row[1]
            
            return JSONResponse(
                content={"success": True, "stats": stats},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
            
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard stats")

# Export app for Vercel
app = app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)