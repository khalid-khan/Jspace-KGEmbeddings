# V13: Implemented RESUM/CACHING logic to prevnt loss of progress during long runs.
# The script now saves classified attributes to 'classification_cache.json' after every batch.
# If the script restarts, it loads the cache and skips already processed attributes.

import os
import re
import urllib.parse
import json
import time
from google import genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError
from typing import List, Dict, Tuple

# --- 0. Define Namespaces and Constants ---
S_NAMESPACE = "http://schema.org/"
EX_NAMESPACE = "http://example.org/resource/"
NEW_NAMESPACE = "http://example.org/newprop/"
PRODUCT_URI_BASE = "http://example.org/product/"
# --- INPUT AND OUTPUT FILES ---
VERTICES_FILE = '2_vertices'
OUTPUT_FILE = '2_vertices_output_Batches.ttl'
CACHE_FILE = 'classification_cache.json' # NEW CACHE FILE
# --- BATCHING SETTINGS ---
BATCH_SIZE = 250
DELAY_SECONDS = 15

# Define the full set of allowed properties for the LLM's JSON schema constraint
ALLOWED_PROPERTIES = [
    'brand', 'color', 'model', 'sensorType', 'batteryModel',
    'productCategory',
    'additionalProperty'
]

# Map LLM output to the final RDF property prefix
PROPERTY_MAP = {
    'brand': 'schema1:brand', 'color': 'schema1:color', 'model': 'schema1:model',
    'sensorType': 'new:sensorType', 'batteryModel': 'new:batteryModel',
    'productCategory': 'schema1:productCategory', 'additionalProperty': 'schema1:additionalProperty'
}

# --- 0.5 Cache Management Functions ---

def load_cache() -> Dict[str, str]:
    """Loads classification data from the cache file if it exists."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Cache file '{CACHE_FILE}' is corrupted. Starting fresh.")
            return {}
    return {}

def save_cache(classification_map: Dict[str, str]):
    """Saves the current classification map to the cache file."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(classification_map, f, indent=4)
        print("-> Cache updated successfully.")
    except Exception as e:
        print(f"Critical Error: Could not save cache file. Progress may be lost on failure. Error: {e}")


# --- 1. LLM Interaction Functions ---

def sanitize_for_uri(text: str) -> str:
    """Creates a clean, valid URI fragment from a string value."""
    if not text: return "Unknown"
    clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text).strip()
    return "".join(x.capitalize() for x in clean_text.split())

def construct_llm_prompt_batched(attribute_list: List[str]) -> str:
    """
    Constructs the LLM prompt for JSON classification using a detailed, step-by-step instruction set.
    """
    CATEGORY_TERMS_LIST = ['DSLR', 'Mirrorless', 'Advance Point and shoot', 'Compact Digital Camera', 'Bridge Camera', 'Entry Level Camera']

    instruction = (
        "TASK: Semantic Product Attribute Classification for a Batch of Attributes.\n"
        "You are an expert ontology engineer classifying product attributes into specific Schema.org properties.\n"
        "Follow these rules precisely:\n\n"

        "**RULE 1: CRITICAL PRODUCT CATEGORY CLASSIFICATION**\n"
        "You must prioritize classifying terms that describe the fundamental TYPE of the product as **'productCategory'**.\n"
        f"Examples of terms that MUST be classified as 'productCategory' include: {', '.join(CATEGORY_TERMS_LIST)}.\n"
        "If an attribute is a known product type, it MUST NOT be classified as 'additionalProperty'.\n\n"

        "**RULE 2: CORE PROPERTIES**\n"
        f"The list of allowed output properties is: {', '.join(ALLOWED_PROPERTIES)}.\n"
        "Classify 'brand', 'color', 'model', 'sensorType', and 'batteryModel' using specific knowledge and context.\n\n"

        "**RULE 3: CATCH-ALL**\n"
        "Only classify an attribute as **'additionalProperty'** if it is general, numerical, or descriptive text that does not fit any of the other specific properties (Rule 1 or Rule 2).\n\n"

        "Return the result as a JSON array of objects, where each object contains the **'attribute_value'** (the exact input string) and its **'schema_property'** classification. You must return an object for EVERY input attribute value.\n\n"
    )

    input_attributes = "\n".join([f"- '{attr}'" for attr in attribute_list])
    input_block = f"--- CURRENT INPUT ATTRIBUTES ({len(attribute_list)} total) ---\n{input_attributes}\n\nOUTPUT: (JSON array only)"

    return instruction + input_block

def call_llm_api_for_batch_classification(attribute_list: List[str]) -> Dict[str, str]:
    """
    API Call Structure: This method attempts a live call to the Gemini API.
    """
    if not attribute_list: return {}

    full_prompt = construct_llm_prompt_batched(attribute_list)
    classification_map = {}

    batch_response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "attribute_value": {"type": "string", "description": "The exact input attribute value."},
                "schema_property": {"type": "string", "enum": ALLOWED_PROPERTIES, "description": "The classified property name."}
            },
            "required": ["attribute_value", "schema_property"]
        }
    }

    try:
        client = genai.Client()
        print(f"-> Making LIVE API call for {len(attribute_list)} attributes...")

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": batch_response_schema
            }
        )

        json_array = json.loads(response.text)
        for item in json_array:
            value = item.get('attribute_value')
            classification = item.get('schema_property', 'additionalProperty')
            if value and isinstance(classification, str):
                classification_map[value] = classification

        print(f"-> API call successful. {len(classification_map)} attributes classified in this batch.")
        return classification_map

    except Exception as e:
        print(f"LLM API Critical Error for this batch. Failing classification to 'additionalProperty'. Error: {e}")
        # Return fallback map for failed batch to keep processing
        return {attr: 'additionalProperty' for attr in attribute_list}

# --- 2. RDF Generation Logic ---

def generate_rdf_string(file_content: str, classification_map: Dict[str, str]) -> str:
    """
    Parses data and generates the final RDF/Turtle string using the LLM's classification map.
    """
    output = []
    output.append(f"@prefix ex: <{EX_NAMESPACE}> .")
    output.append(f"@prefix schema1: <{S_NAMESPACE}> .")
    output.append(f"@prefix new: <{NEW_NAMESPACE}> .")
    output.append("")

    lines = file_content.strip().split('\n')
    product_data_map = {}

    # PASS 1: Re-parse Data and Group
    for line in lines:
        if not line.strip(): continue
        parts = line.split(';')
        if len(parts) < 4: continue

        product_id = parts[0]
        raw_attrs = parts[3].split('|') + parts[4:]
        name = raw_attrs[0].strip() if raw_attrs else "Unknown"
        attrs = [x.strip().replace('\\,', ',') for x in raw_attrs[1:] if x.strip()]
        product_data_map[product_id] = {"name": name, "attributes": attrs}

    defined_entities = {}
    SINGLE_VALUE_PROPS = ['schema1:brand', 'schema1:color', 'new:sensorType', 'new:batteryModel']

    # PASS 2: RDF Generation
    for product_id, data in product_data_map.items():
        name = data["name"]

        output.append(f"<{PRODUCT_URI_BASE}{product_id}> a schema1:Product ;")

        product_props = {
            'schema1:additionalProperty': [], 'schema1:brand': None, 'schema1:color': None,
            'new:sensorType': None, 'new:batteryModel': None, 'schema1:productCategory': [],
            'schema1:model': []
        }

        for val in data["attributes"]:
            # Uses the merged classification map here
            prop_category_unprefixed = classification_map.get(val, 'additionalProperty')
            prop = PROPERTY_MAP.get(prop_category_unprefixed, 'schema1:additionalProperty')
            val_clean = val.replace('"', '\\"')

            if prop == 'schema1:additionalProperty':
                product_props[prop].append(val_clean)
            elif prop in SINGLE_VALUE_PROPS:
                uri_suffix = sanitize_for_uri(val)
                entity_uri = f"ex:{uri_suffix}"

                # --- START: Entity Type Assignment (V13) ---
                if prop == 'schema1:brand':
                    entity_type = 'schema1:Brand'
                elif prop == 'schema1:color':
                    entity_type = 'new:Color'
                else:
                    entity_type = 'schema1:DefinedTerm'
                # --- END: Entity Type Assignment ---

                defined_entities[entity_uri] = (entity_type, val)
                if not product_props[prop]:
                    product_props[prop] = entity_uri

            elif prop == 'schema1:productCategory':
                uri_suffix = sanitize_for_uri(val)
                entity_uri = f"ex:{uri_suffix}"
                entity_type = 'schema1:DefinedTerm'
                defined_entities[entity_uri] = (entity_type, val)
                if entity_uri not in product_props[prop]:
                    product_props[prop].append(entity_uri)
            elif prop == 'schema1:model':
                product_props[prop].append(val_clean)

        ordered_keys = [
            'schema1:additionalProperty', 'schema1:brand', 'schema1:color',
            'schema1:productCategory', 'new:sensorType', 'new:batteryModel',
            'schema1:model'
        ]

        for key in ordered_keys:
            if key == 'schema1:additionalProperty' and product_props[key]:
                joined_props = " | ".join(product_props[key])
                output.append(f'    {key} "{joined_props}" ;')
            elif key in SINGLE_VALUE_PROPS and product_props[key]:
                output.append(f'    {key} {product_props[key]} ;')
            elif key == 'schema1:productCategory' and product_props[key]:
                joined_categories = " , ".join(product_props[key])
                output.append(f'    {key} {joined_categories} ;')
            elif key == 'schema1:model' and product_props[key]:
                joined_models = " | ".join(product_props[key])
                output.append(f'    {key} "{joined_models}" ;')

        output.append(f'    schema1:name "{name}"@en .')
        output.append("")

    # Define Entities (Classes)
    for uri, (type_, label) in sorted(defined_entities.items()):
        output.append(f"{uri} a {type_} ;")
        output.append(f'    schema1:name "{label}" .')
        output.append("")

    return "\n".join(output)

# --- 3. Main Execution (With Batching and Caching) ---
if __name__ == '__main__':
    if os.path.exists(VERTICES_FILE):
        with open(VERTICES_FILE, 'r', encoding='utf-8') as f:
            file_content = f.read().strip()

        # 1. Collect all unique attributes
        all_attributes_list = []
        for line in file_content.strip().split('\n'):
            if not line.strip(): continue
            parts = line.split(';')
            if len(parts) < 4: continue
            raw_attrs = parts[3].split('|') + parts[4:]
            attrs = [x.strip().replace('\\,', ',') for x in raw_attrs[1:] if x.strip()]
            all_attributes_list.extend(attrs)

        unique_attrs = list(set(all_attributes_list))
        print(f"Total unique attributes found: {len(unique_attrs)}")

        # 2. Load Cache and Identify Unprocessed Attributes
        final_classification_map = load_cache()

        processed_attributes = set(final_classification_map.keys())
        unprocessed_attrs = [attr for attr in unique_attrs if attr not in processed_attributes]

        print(f"Attributes already classified (from cache): {len(processed_attributes)}")
        print(f"Attributes remaining to be processed: {len(unprocessed_attrs)}")

        if not unprocessed_attrs:
            print("--- All attributes are already classified. Skipping API calls. ---")

        # 3. Split *unprocessed* attributes into batches
        batched_attributes = [
            unprocessed_attrs[i:i + BATCH_SIZE]
            for i in range(0, len(unprocessed_attrs), BATCH_SIZE)
        ]
        num_batches = len(batched_attributes)

        # 4. Process batches with a time delay and save cache
        for i, batch in enumerate(batched_attributes):
            print(f"\nProcessing Batch {i+1} of {num_batches} ({len(batch)} attributes)...")

            # 4.1 Call API
            batch_map = call_llm_api_for_batch_classification(batch)
            final_classification_map.update(batch_map)

            # 4.2 Save progress to cache immediately
            save_cache(final_classification_map)

            # 4.3 Wait if there are more batches to process
            if i < num_batches - 1:
                print(f"Waiting for {DELAY_SECONDS} seconds before the next API call to respect quotas...")
                time.sleep(DELAY_SECONDS)

        print(f"\n--- Batch processing complete. Total classified attributes: {len(final_classification_map)} ---")

        # 5. Generate final RDF using the complete map
        rdf_output = generate_rdf_string(file_content, final_classification_map)

        # 6. Save final RDF to file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(rdf_output)

        print(f"\n--- Final Output ---")
        print(f"Full RDF (Turtle) saved to {OUTPUT_FILE}.")
        print("Progress is now safe to resume from 'classification_cache.json'.")
        print(rdf_output[:2500] + "\n...")
    else:
        print(f"Error: File '{VERTICES_FILE}' not found. Please ensure the file is present before running.")