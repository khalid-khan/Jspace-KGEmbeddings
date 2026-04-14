#This script takes intoaccount the result of ver13 script and adds the
# category hierarchy created/deduced by our catgory hierarchy script.


import os

# --- 1. Hierarchy Configuration ---
# This matches your specific multi-level camera ontology
CATEGORY_HIERARCHY = {
    "MirrorlessCamera": "MirrorlessInterchangeableLensCamera",
    "MicroFourThirdsSystemCamera": "MirrorlessInterchangeableLensCamera",
    "MirrorlessInterchangeableLensCamera": "InterchangeableLensDigitalCamera",
    "Dslr": "DigitalSingleLensReflexCamera",
    "SlrKit": "DigitalSingleLensReflexCamera",
    "Alpha33Dslr": "DigitalSingleLensReflexCamera",
    "DigitalSingleLensReflexCamera": "InterchangeableLensDigitalCamera",
    "InterchangeableLensDigitalCamera": "DigitalCamera",
    "Ultracompact": "CompactCamera",
    "CompactCamera": "PointShoot",
    "PointShootWithZoomLens": "PointShoot",
    "PointShootDigitalCamera": "PointShoot",
    "AdvancedPointAndShoot": "PointShoot",
    "PointShoot": "DigitalCamera","""
RDF STRUCTURAL CLEANING SCRIPT THAT REMOVES LITERAL VALUES
------------------------------
This script processes Turtle (.ttl) files to remove literal string values (e.g., product names,
descriptions, and model numbers) while preserving high-value Object Properties (URIs).
It specifically handles language tags (e.g., @en) and ensures TTL syntax integrity
by mending trailing punctuation (. vs ;) when lines are removed.

Now we also want to remove the name filed accordingly as those are string laterals
as well.
"""

import re
import os

# Configuration
INPUT_FILE = "2_vertices_output of 73 batches with prodcat with category hierarchy.ttl"
OUTPUT_FILE = "cleaned_structural_kg.ttl"

def clean_turtle_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f" Error: {input_path} not found.")
        return

    # Updated Regex:
    # 1. Matches property
    # 2. Matches "literal content"
    # 3. Matches optional language tag like @en
    # 4. Matches trailing punctuation . or ;
    literal_pattern = re.compile(r'^\s*(schema1|new|ex):\w+\s+".*?"(@\w+)?\s*([.;])') # thisisthe critical part..

    lines_to_write = []
    removed_count = 0

    print(f" Starting cleanup on {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        match = literal_pattern.search(line)

        if match:
            punctuation = match.group(3) # The '.' or ';'

            # If we remove a line that ended the block (with a '.'),
            # we must find the previous kept line and change its ';' to a '.'
            if punctuation == '.' and lines_to_write:
                # Look back for the last non-empty line we decided to keep
                for j in range(len(lines_to_write) - 1, -1, -1):
                    if lines_to_write[j].strip():
                        # Replace trailing semicolon with a period
                        lines_to_write[j] = re.sub(r';\s*$', ' .\n', lines_to_write[j])
                        break

            removed_count += 1
            continue

        lines_to_write.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines_to_write)

    print(f"\n Cleanup Complete!")
    print(f" Literals removed: {removed_count}")
    print(f" Lines preserved: {len(lines_to_write)}")
    print(f" Saved to: {output_path}")

if __name__ == "__main__":
    clean_turtle_file(INPUT_FILE, OUTPUT_FILE)
    "UnderwaterCamera": "WaterproofCamera",
    "Underwatercamera": "WaterproofCamera",
    "WaterproofCamera": "DigitalCamera",
    "ActionCam": "DigitalCamera",
    "3dDigitalCamera": "DigitalCamera",
    "DualDisplayCamera": "DigitalCamera",
    "FisheyeCamera": "DigitalCamera",
}

# Mapping slugs to nice labels for new higher-level categories
LABELS = {
    "DigitalCamera": "Digital Camera",
    "InterchangeableLensDigitalCamera": "Interchangeable Lens Digital Camera",
    "DigitalSingleLensReflexCamera": "Digital Single Lens Reflex Camera",
    "MirrorlessInterchangeableLensCamera": "Mirrorless Interchangeable Lens Camera",
    "PointShoot": "Point and Shoot",
    "WaterproofCamera": "Waterproof Camera"
}

def enrich_ttl(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create the hierarchy block
    hierarchy_output = ["\n# --- ARTIFICIAL HIERARCHY ENRICHMENT -"]

    # 1. Add subCategoryOf relationships
    for child, parent in CATEGORY_HIERARCHY.items():
        hierarchy_output.append(f"ex:{child} new:subCategoryOf ex:{parent} .")

    # 2. Add definitions for the higher-level categories that might not exist in original file
    hierarchy_output.append("\n# --- High-Level Category Definitions ---")
    unique_parents = set(CATEGORY_HIERARCHY.values())
    for parent in sorted(unique_parents):
        label = LABELS.get(parent, parent)
        hierarchy_output.append(f"ex:{parent} a schema1:DefinedTerm ;")
        hierarchy_output.append(f'    schema1:name "{label}" .')
        hierarchy_output.append("")

    # Combine original content with the new hierarchy
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
        f.write("\n".join(hierarchy_output))

    print(f"Done! Created {output_file}")
    print(f"We can now feed {output_file} into the DICEE framework.")

if __name__ == "__main__":
    enrich_ttl("2_vertices_output of 73 batches with prodcat.ttl", "2_vertices_output of 73 batches with prodcat with category hierarchy.ttl")