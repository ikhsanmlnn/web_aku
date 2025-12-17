"""
Debug script untuk melihat struktur data dari Dicoding API
Run ini terlebih dahulu untuk memahami struktur data
"""
import requests
import json

DICODING_API_BASE = "https://jrkqcbmjknzgpbtrupxh.supabase.co/rest/v1"
DICODING_API_KEY = "sb_publishable_h889CjrPIGwCMA9I4oTTaA_2L22Y__R"

headers = {
    "apikey": DICODING_API_KEY,
    "Authorization": f"Bearer {DICODING_API_KEY}",
    "Content-Type": "application/json"
}

def test_endpoint(endpoint: str, limit: int = 2):
    """Test single endpoint and show structure"""
    print(f"\n{'='*60}")
    print(f"Testing: {endpoint}")
    print(f"{'='*60}")
    
    try:
        url = f"{DICODING_API_BASE}/{endpoint}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print("❌ No data returned")
            return
        
        print(f"✅ Success! Total records: {len(data)}")
        
        # Show first record structure
        if len(data) > 0:
            first_record = data[0]
            print(f"\n📋 Columns ({len(first_record)}):")
            for key, value in first_record.items():
                value_type = type(value).__name__
                value_preview = str(value)[:50] if value else "None"
                print(f"   - {key:25} ({value_type:10}): {value_preview}")
            
            print(f"\n📄 Sample Records (first {min(limit, len(data))}):")
            for i, record in enumerate(data[:limit], 1):
                print(f"\n   Record {i}:")
                print(f"   {json.dumps(record, indent=6, ensure_ascii=False)}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    """Test all endpoints"""
    print("\n" + "="*60)
    print("  DICODING API STRUCTURE DEBUGGER")
    print("="*60)
    
    endpoints = [
        "learning_paths",
        "courses",
        "course_levels",
        "tutorials"
    ]
    
    for endpoint in endpoints:
        test_endpoint(endpoint, limit=2)
    
    print("\n" + "="*60)
    print("  DONE!")
    print("="*60)
    print("\nℹ️  Gunakan informasi di atas untuk memahami struktur data API")
    print("   Pastikan kolom yang digunakan di code sesuai dengan API response\n")


if __name__ == "__main__":
    main()