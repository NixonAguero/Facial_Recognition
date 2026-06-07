import os 
from dotenv import load_dotenv

load_dotenv()
print("Supabase URL:", os.getenv("SUPABASE_URL"))
print("Bucket:", os.getenv("SUPABASE_BUCKET"))

