import json
import os
from supabase import create_client, Client
from openai import OpenAI
from typing import Dict, Any, List
import time
from dotenv import load_dotenv

load_dotenv()

def initialize_clients():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not all([supabase_url, supabase_key, openai_api_key]):
        raise ValueError("Missing required environment variables. Please set SUPABASE_URL, SUPABASE_ANON_KEY, and OPENAI_API_KEY")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    openai_client = OpenAI(api_key=openai_api_key)
    
    return supabase, openai_client

def extract_text_content(json_doc: Dict[str, Any]) -> str:
    text_parts = []
    
    if "title" in json_doc:
        text_parts.append(f"Product Title: {json_doc['title']}")
    
    if "description" in json_doc:
        text_parts.append(f"Description: {json_doc['description']}")
    
    if "price" in json_doc:
        text_parts.append(f"Price: ${json_doc['price']}")
    
    if "category" in json_doc and isinstance(json_doc["category"], dict):
        category_name = json_doc["category"].get("name", "")
        if category_name:
            text_parts.append(f"Category: {category_name}")
    
    return " ".join(text_parts)

def generate_embedding(text: str, openai_client: OpenAI) -> List[float]:
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def create_metadata(json_doc: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "product_id": json_doc.get("id"),
        "title": json_doc.get("title"),
        "price": json_doc.get("price"),
        "document_type": "product"
    }
    
    # Handle category information
    if "category" in json_doc and isinstance(json_doc["category"], dict):
        metadata["category"] = {
            "id": json_doc["category"].get("id"),
            "name": json_doc["category"].get("name"),
            "image": json_doc["category"].get("image")
        }
    
    # Handle images
    if "images" in json_doc and isinstance(json_doc["images"], list):
        metadata["images"] = json_doc["images"]
        metadata["image_count"] = len(json_doc["images"])
    
    return metadata

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("JSON file must contain either a single object or an array of objects")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")

def check_table_structure(supabase: Client):
    """Check if the documents table exists and has the correct structure"""
    try:
        # Try to get table info
        result = supabase.table("documents").select("*").limit(1).execute()
        print("✅ Documents table exists and is accessible")
        return True
    except Exception as e:
        print(f"❌ Error accessing documents table: {e}")
        return False

def process_and_insert_documents(supabase: Client, openai_client: OpenAI, json_file_path: str, batch_size: int = 10):
    print(f"Loading products from: {json_file_path}")
    
    # Check table structure first
    if not check_table_structure(supabase):
        print("❌ Cannot proceed without proper table structure")
        return 0, 0
    
    documents = load_json_file(json_file_path)
    print(f"Found {len(documents)} products to process")
    
    successful_inserts = 0
    failed_inserts = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        for j, doc in enumerate(batch):
            doc_index = i + j + 1
            print(f"Processing product {doc_index}/{len(documents)}")
            
            try:
                text_content = extract_text_content(doc)
                print(f"  - Extracted {len(text_content)} characters of text")
                
                print("  - Generating embedding...")
                embedding = generate_embedding(text_content, openai_client)
                
                metadata = create_metadata(doc)
                
                print("  - Preparing data for insertion...")
                insert_data = {
                    "content": text_content,
                    "metadata": metadata,
                    "embedding": embedding
                }
                
                print("  - Inserting into Supabase...")
                result = supabase.table("documents").insert(insert_data).execute()
                
                if result.data and len(result.data) > 0:
                    print(f"  ✅ Successfully inserted product with id: {result.data[0].get('id', 'unknown')}")
                    successful_inserts += 1
                else:
                    print(f"  ❌ Insert returned no data: {result}")
                    failed_inserts += 1
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ Failed to insert product {doc_index}: {e}")
                print(f"  📝 Error details: {type(e).__name__}")
                
                # Print more detailed error information
                if hasattr(e, 'args') and e.args:
                    print(f"  📝 Error args: {e.args}")
                
                failed_inserts += 1
                continue
    
    print(f"\n📊 Summary:")
    print(f"✅ Successfully inserted: {successful_inserts} products")
    print(f"❌ Failed to insert: {failed_inserts} products")
    if successful_inserts + failed_inserts > 0:
        print(f"📈 Success rate: {(successful_inserts/(successful_inserts+failed_inserts)*100):.1f}%")
    
    return successful_inserts, failed_inserts

def search_documents(supabase: Client, openai_client: OpenAI, query: str, match_count: int = 5, filter_metadata: Dict = None) -> List[Dict[str, Any]]:
    try:
        query_embedding = generate_embedding(query, openai_client)
        
        filter_json = filter_metadata or {}
        
        result = supabase.rpc("match_docs", {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter": filter_json
        }).execute()
        
        return result.data
        
    except Exception as e:
        print(f"Error searching documents: {e}")
        raise

def check_and_create_table(supabase: Client):
    """Check if documents table exists and create it if it doesn't"""
    try:
        print("🔍 Checking if documents table exists...")
        # Try to select from the table
        result = supabase.table("documents").select("id").limit(1).execute()
        print("✅ Documents table exists and is accessible")
        return True
    except Exception as e:
        print(f"❌ Documents table check failed: {e}")
        print("📝 This usually means the table doesn't exist or there are permission issues")
        return False

def create_table_sql():
    """Return the SQL to create the documents table"""
    return """
-- Enable the vector extension (required for embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documents table
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1536),  -- OpenAI ada-002 uses 1536 dimensions
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create an index on the embedding column for faster similarity searches
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops);

-- Disable RLS for now (you can enable it later with proper policies)
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;

-- Create the RPC function for similarity search
CREATE OR REPLACE FUNCTION match_docs(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'
)
RETURNS TABLE(
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE documents.metadata @> filter
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$;
"""

def test_simple_insert(supabase: Client):
    """Test a simple insert without embeddings to check basic connectivity"""
    try:
        print("🧪 Testing simple insert...")
        test_data = {
            "content": "Test content",
            "metadata": {"test": True, "document_type": "test"}
        }
        
        result = supabase.table("documents").insert(test_data).execute()
        
        if result.data:
            print("✅ Simple insert successful")
            # Clean up test data
            test_id = result.data[0].get('id')
            if test_id:
                supabase.table("documents").delete().eq('id', test_id).execute()
                print("🧹 Cleaned up test data")
            return True
        else:
            print("❌ Simple insert failed - no data returned")
            return False
            
    except Exception as e:
        print(f"❌ Simple insert failed: {e}")
        return False

def main():
    JSON_FILE_PATH = "db.json"
    
    try:
        print("🔌 Initializing Supabase and OpenAI clients...")
        supabase, openai_client = initialize_clients()
        print("✅ Clients initialized successfully")
        
        # Check if table exists
        if not check_and_create_table(supabase):
            print("\n🛠️  SETUP REQUIRED:")
            print("Your Supabase database needs the 'documents' table and RPC function.")
            print("\n📋 Please run this SQL in your Supabase SQL Editor:")
            print("=" * 60)
            print(create_table_sql())
            print("=" * 60)
            print("\n📍 Steps to fix:")
            print("1. Go to your Supabase Dashboard")
            print("2. Navigate to 'SQL Editor' in the left sidebar")
            print("3. Copy and paste the SQL above")
            print("4. Click 'Run' to execute the SQL")
            print("5. Run this script again")
            print("\n💡 Make sure you're using the correct SUPABASE_URL and SUPABASE_ANON_KEY")
            return
        
        # Test basic connectivity
        if not test_simple_insert(supabase):
            print("❌ Basic connectivity test failed even with table present.")
            print("💡 Check your Supabase credentials and permissions.")
            return
        
        print(f"\n📄 Starting product processing from: {JSON_FILE_PATH}")
        successful, failed = process_and_insert_documents(
            supabase, 
            openai_client, 
            JSON_FILE_PATH,
            batch_size=5
        )
        
        if successful > 0:
            print(f"\n🎉 Process completed! {successful} products added to your Supabase vector store.")
            
            print("\n🔍 Testing search functionality...")
            search_results = search_documents(
                supabase, 
                openai_client, 
                "wooden dining table mid-century modern",
                match_count=3
            )
            
            print(f"Found {len(search_results)} similar products:")
            for i, result in enumerate(search_results):
                print(f"\nResult {i+1}:")
                print(f"  Similarity: {result['similarity']:.4f}")
                print(f"  Product: {result['metadata'].get('title', 'N/A')}")
                print(f"  Price: ${result['metadata'].get('price', 'N/A')}")
                category = result['metadata'].get('category', {})
                if isinstance(category, dict):
                    print(f"  Category: {category.get('name', 'N/A')}")
        else:
            print("❌ No products were successfully inserted.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease check:")
        print("1. Your environment variables are set correctly")
        print("2. Your JSON file exists and is valid")
        print("3. Your Supabase database is accessible")
        print("4. Your OpenAI API key is valid")
        print("5. Your 'documents' table exists with correct schema")
        print("6. Your 'match_docs' RPC function is properly set up")

if __name__ == "__main__":
    main()