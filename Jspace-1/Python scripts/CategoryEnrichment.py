#This script takes intoaccount the result of ver13 script and adds the
# category hierarchy created/deduced by our catgory hierarchy script.


import os

# --- 1. Hierarchy Configuration ---
# This matches our specific multi-level camera ontology
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
    "PointShoot": "DigitalCamera",
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
    hierarchy_output = ["\n# --- ARTIFICIAL HIERARCHY ENRICHMENT ---"]

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