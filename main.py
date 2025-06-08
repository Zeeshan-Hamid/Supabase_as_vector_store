import json
import os
from supabase import create_client, Client
from openai import OpenAI
from typing import Dict, Any, List
import time

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
    
    if "Name" in json_doc:
        text_parts.append(f"Project Name: {json_doc['Name']}")
    
    if "Location" in json_doc:
        text_parts.append(f"Location: {json_doc['Location']}")
    
    if "Overview" in json_doc:
        text_parts.append(f"Overview: {json_doc['Overview']}")
    
    if "Developer" in json_doc:
        text_parts.append(f"Developer: {json_doc['Developer']}")
    
    if "Unit Prices" in json_doc:
        text_parts.append(f"Starting Price: {json_doc['Unit Prices']}")
    
    if "Detailed Pricing" in json_doc:
        text_parts.append(f"Pricing Details: {json_doc['Detailed Pricing']}")
    
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
        "uuid": json_doc.get("UUID"),
        "name": json_doc.get("Name"),
        "location": json_doc.get("Location"),
        "developer": json_doc.get("Developer"),
        "launch_date": json_doc.get("Launch date"),
        "document_type": "real_estate_project"
    }
    
    if "Parsed_Units" in json_doc:
        unit_types = list(set([unit["unit_type"] for unit in json_doc["Parsed_Units"]]))
        bedroom_types = list(set([unit["bedrooms"] for unit in json_doc["Parsed_Units"]]))
        metadata["unit_types"] = unit_types
        metadata["bedroom_types"] = bedroom_types
        metadata["price_range"] = {
            "min": min([unit["price_min"] for unit in json_doc["Parsed_Units"]]),
            "max": max([unit["price_max"] for unit in json_doc["Parsed_Units"]])
        }
    
    return metadata

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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

def process_and_insert_documents(supabase: Client, openai_client: OpenAI, json_file_path: str, batch_size: int = 10):
    print(f"Loading documents from: {json_file_path}")
    
    documents = load_json_file(json_file_path)
    print(f"Found {len(documents)} documents to process")
    
    successful_inserts = 0
    failed_inserts = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        for j, doc in enumerate(batch):
            doc_index = i + j + 1
            print(f"Processing document {doc_index}/{len(documents)}")
            
            try:
                text_content = extract_text_content(doc)
                print(f"  - Extracted {len(text_content)} characters of text")
                
                print("  - Generating embedding...")
                embedding = generate_embedding(text_content, openai_client)
                
                metadata = create_metadata(doc)
                
                print("  - Inserting into Supabase...")
                result = supabase.table('documents').insert({
                    'content': text_content,
                    'metadata': metadata,
                    'embedding': embedding
                }).execute()
                
                print(f"  ✅ Successfully inserted document with ID: {result.data[0]['id']}")
                successful_inserts += 1
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ Failed to insert document {doc_index}: {e}")
                failed_inserts += 1
                continue
    
    print(f"\n📊 Summary:")
    print(f"✅ Successfully inserted: {successful_inserts} documents")
    print(f"❌ Failed to insert: {failed_inserts} documents")
    print(f"📈 Success rate: {(successful_inserts/(successful_inserts+failed_inserts)*100):.1f}%")
    
    return successful_inserts, failed_inserts

def search_documents(supabase: Client, openai_client: OpenAI, query: str, match_count: int = 5, filter_metadata: Dict = None) -> List[Dict[str, Any]]:
    try:
        query_embedding = generate_embedding(query, openai_client)
        
        filter_json = filter_metadata or {}
        
        result = supabase.rpc('match_docs', {
            'query_embedding': query_embedding,
            'match_count': match_count,
            'filter': filter_json
        }).execute()
        
        return result.data
        
    except Exception as e:
        print(f"Error searching documents: {e}")
        raise

def main():
    JSON_FILE_PATH = "documents.json"
    
    try:
        print("🔌 Initializing Supabase and OpenAI clients...")
        supabase, openai_client = initialize_clients()
        print("✅ Clients initialized successfully")
        
        print(f"\n📄 Starting document processing from: {JSON_FILE_PATH}")
        successful, failed = process_and_insert_documents(
            supabase, 
            openai_client, 
            JSON_FILE_PATH,
            batch_size=5
        )
        
        if successful > 0:
            print(f"\n🎉 Process completed! {successful} documents added to your Supabase vector store.")
            
            print("\n🔍 Testing search functionality...")
            search_results = search_documents(
                supabase, 
                openai_client, 
                "luxury apartments with garden views",
                match_count=3
            )
            
            print(f"Found {len(search_results)} similar documents:")
            for i, result in enumerate(search_results):
                print(f"\nResult {i+1}:")
                print(f"  Similarity: {result['similarity']:.4f}")
                print(f"  Project: {result['metadata'].get('name', 'N/A')}")
                print(f"  Location: {result['metadata'].get('location', 'N/A')}")
        else:
            print("❌ No documents were successfully inserted.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease check:")
        print("1. Your environment variables are set correctly")
        print("2. Your JSON file exists and is valid")
        print("3. Your Supabase database is accessible")
        print("4. Your OpenAI API key is valid")

if __name__ == "__main__":
    main()