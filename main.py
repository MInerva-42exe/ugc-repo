import os
import json
import time
import requests
from typing import Any, Dict, List, Optional

# The original, correct set of imports for the Vertex AI SDK version
from google.api_core import exceptions
from google.cloud import secretmanager, firestore, storage, aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index import MatchingEngineIndex
import vertexai
from vertexai.language_models import TextEmbeddingModel
from notion_client import Client as NotionClient

# --- Configuration ---
VECTOR_SEARCH_INDEX_ID = "7394437798142410752" # The new, clean v2 index ID
FIRESTORE_COLLECTION = "success_stories"
REGION = "us-central1"
EMBEDDING_BATCH_SIZE = 25

# --- Clients ---
secret_client = secretmanager.SecretManagerServiceClient()
firestore_client = firestore.Client()
storage_client = storage.Client()

# --- Helper Functions ---
def get_notion_property(props: Dict[str, Any], name: str, default: Any = None) -> Any:
    p = props.get(name)
    if not p:
        return default
    t = p.get("type")
    if t in ("title", "rich_text"):
        key = "title" if t == "title" else "rich_text"
        return "".join(item.get("plain_text", "") for item in p.get(key, [])) or default
    if t == "url":
        return p.get("url") or default
    if t == "email":
        return p.get("email") or default
    if t == "checkbox":
        return p.get("checkbox", False)
    if t == "files":
        urls = []
        for f in p.get("files", []):
            u = f.get(f.get("type", ""), {}).get("url")
            if u:
                urls.append(u)
        return urls or default
    if t in ("select", "multi_select"):
        items = p.get("multi_select", []) if t == "multi_select" else [p.get("select", {})]
        return [i.get("name") for i in items if i and i.get("name")] or default
    return default

def get_secret(name: str) -> str:
    proj = os.environ.get("GCP_PROJECT_ID")
    if not proj:
        raise RuntimeError("GCP_PROJECT_ID not set")
    path = f"projects/{proj}/secrets/{name}/versions/latest"
    resp = secret_client.access_secret_version(request={"name": path})
    return resp.payload.data.decode("utf-8")

def download_upload(bucket, url: str, dest: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        blob = bucket.blob(dest)
        blob.upload_from_string(r.content, content_type=r.headers.get("Content-Type"))
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Error downloading/uploading {url}: {e}")
        return None

def get_all_page_text(notion_client, page_id: str) -> str:
    all_text = []
    try:
        blocks = notion_client.blocks.children.list(block_id=page_id).get("results", [])
        for block in blocks:
            if block.get("has_children"):
                all_text.append(get_all_page_text(notion_client, block["id"]))
            block_type = block.get("type")
            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
                text_parts = [part.get("plain_text", "") for part in block[block_type].get("rich_text", [])]
                all_text.append("".join(text_parts))
    except Exception as e:
        print(f"    - Could not fetch content for block {page_id}: {e}")
    return "\n".join(all_text)

# --- Main Execution ---
def main():
    print("Starting data sync with full page content enrichment...")
    try:
        proj = os.environ["GCP_PROJECT_ID"]
        vertexai.init(project=proj, location=REGION)
        index_name = f"projects/{proj}/locations/{REGION}/indexes/{VECTOR_SEARCH_INDEX_ID}"
        index = MatchingEngineIndex(index_name=index_name)
        notion = NotionClient(auth=get_secret("NOTION_API_TOKEN"))
        db_id = get_secret("NOTION_DB_ID")
        bucket = storage_client.bucket(f"{proj}-media-bucket")
        embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    except Exception as e:
        print(f"Initialization error: {e}"); return

    all_pages = []; next_cursor = None; page_count = 0
    while True:
        page_count += 1; print(f"Fetching Notion page {page_count}...")
        try:
            query_params = {"database_id": db_id}
            if next_cursor: query_params["start_cursor"] = next_cursor
            results = notion.databases.query(**query_params)
            current_page_results = results.get("results", []); all_pages.extend(current_page_results)
            next_cursor = results.get("next_cursor")
            if not next_cursor: print(f"Successfully fetched all {page_count} pages from Notion."); break
        except Exception as e:
            print(f"Error fetching Notion page {page_count}: {e}"); break

    print(f"\nProcessing {len(all_pages)} total entries from Notion...")
    to_write_firestore = []; to_embed = []
    for page in all_pages:
        props = page["properties"]; customer_name = get_notion_property(props, "Customer Name ")
        if not customer_name or customer_name.strip() == "-":
            print(f"Skipping entry with ID {page['id']} due to missing/invalid customer name.")
            continue
        
        print(f"Processing: {customer_name}")
        page_content_narrative = get_all_page_text(notion, page['id'])
        
        sd = {
            "notion_id": page["id"], "customer_name": customer_name,
            "customer_email": get_notion_property(props, "Customer Email"),
            "designation": get_notion_property(props, "Designation"),
            "organization_name": get_notion_property(props, "Organization"),
            "country": get_notion_property(props, "Country"), "industry_tags": get_notion_property(props, "Industry"),
            "product_tags": get_notion_property(props, "Product"), "testimonial_quote": get_notion_property(props, "Testimonial Quote"),
            "case_study_link": get_notion_property(props, "Case study link"), "reference_consent": get_notion_property(props, "Reference Consent"),
            "is_enterprise_case_study": get_notion_property(props, "Enterprise Case Study"), "linkedin_profile": get_notion_property(props, "LinkedIn Profile"),
            "page_content_narrative": page_content_narrative,
            "org_logo": None, "org_photo": None,
        }
        
        logos, photos = get_notion_property(props, "Org logo") or [], get_notion_property(props, "Customer Photo") or []
        if logos: sd["org_logo"] = download_upload(bucket, logos[0], f"logos/{page['id']}_logo.png")
        if photos: sd["org_photo"] = download_upload(bucket, photos[0], f"customer_photos/{page['id']}_photo.png")

        embedding_text = (f"Customer: {sd.get('customer_name') or 'NA'}. "
                          f"Organization: {sd.get('organization_name') or 'NA'}. "
                          f"Industry: {', '.join(sd.get('industry_tags')) if sd.get('industry_tags') else 'NA'}. "
                          f"Products Used: {', '.join(sd.get('product_tags')) if sd.get('product_tags') else 'NA'}. "
                          f"Testimonial Quote: {sd.get('testimonial_quote') or 'NA'}. "
                          f"Full Case Study Narrative:\n{page_content_narrative}")
        sd["full_story_text_content"] = embedding_text
        
        to_write_firestore.append(sd)
        to_embed.append(embedding_text)

    print(f"\nGenerating embeddings for {len(to_write_firestore)} valid entries...")
    vectors = []
    for i in range(0, len(to_embed), EMBEDDING_BATCH_SIZE):
        batch_texts = to_embed[i:i + EMBEDDING_BATCH_SIZE]
        print(f"  - Processing batch starting at index {i} (size: {len(batch_texts)})...")
        
        embeddings = None
        for attempt in range(5):
            try:
                embeddings = embed_model.get_embeddings(batch_texts)
                print(f"    - Successfully got embeddings on attempt {attempt + 1}")
                break 
            except Exception as e:
                delay = 5 * (attempt + 1)
                print(f"    - API call attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                if attempt == 4:
                    print(f"    - FAILED to get embeddings after 5 retries for batch at index {i}.")
    
        if embeddings:
            for j, emb_data in enumerate(embeddings):
                original_index = i + j
                page_id = to_write_firestore[original_index]['notion_id']
                vectors.append({"id": page_id, "embedding": emb_data.values})
                
        time.sleep(1)

    print(f"\nSuccessfully generated {len(vectors)} of {len(to_write_firestore)} embeddings.")

    if to_write_firestore:
        print("Writing structured data to Firestore...")
        batch = firestore_client.batch()
        col = firestore_client.collection(FIRESTORE_COLLECTION)
        for doc_data in to_write_firestore:
            batch.set(col.document(doc_data["notion_id"]), doc_data)
        batch.commit()
        print(f"Wrote {len(to_write_firestore)} docs to Firestore.")

    if vectors and len(vectors) == len(to_write_firestore):
        print("Uploading vector data for batch update...")
        jsonl_content = "\n".join([json.dumps(v) for v in vectors])
        update_job_id = os.urandom(8).hex()
        gcs_directory_path = f"vector_updates/{update_job_id}"
        gcs_file_path = f"{gcs_directory_path}/data.json"
        
        bucket = storage_client.bucket(f"{os.environ['GCP_PROJECT_ID']}-media-bucket")
        update_blob = bucket.blob(gcs_file_path)

        update_blob.upload_from_string(jsonl_content, content_type="application/json")
        gcs_directory_uri = f"gs://{bucket.name}/{gcs_directory_path}"
        print(f"Triggering batch update for index {index.name} from directory {gcs_directory_uri}...")
        index.update_embeddings(contents_delta_uri=gcs_directory_uri)
        print(f"Successfully triggered update for {len(vectors)} vectors. The index is updating.")
    elif vectors:
        print(f"Skipping Vector Search update: mismatch in embedding count ({len(vectors)}) vs doc count ({len(to_write_firestore)}).")

    print("Data sync complete.")

if __name__ == "__main__":
    main()
