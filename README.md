# Jspace-KGEmbeddings
The repository for our research work for SameAs IRI using KG Embeddings to create jspace approach.


# JSpace: Knowledge Graph Embedding Alignment

This repository contains the source code for the JSpace framework, a pipeline designed to structurally evaluate and repair `owl:sameAs` links across disparate Knowledge Graphs using continuous metric learning.

## Pipeline Overview

The framework operates in a sequential, multi-stage pipeline:

1. **LLM-Assisted Knowledge Extraction (`1_llm_extraction.py`)** Utilizes the Google Gemini API to parse heterogeneous product data into a strict RDF schema with caching and batching support.
2. **Hierarchy Enrichment (`2_hierarchy_enrichment.py`)** Injects multi-level subcategory relationships to improve the structural topology of the graph.
3. **Structural Cleaning (`3_rdf_cleaner.py`)** Removes raw string literals, preparing the dataset for purely topological embedding generation.
4. **Embedding Generation (`4_dicee_runner.py`)** Generates independent baseline representations using the DICEE framework (KECI algorithm).
5. **JSpace Alignment (`5_jspace_mapper.py`)** A custom PyTorch continuous learning model that minimizes a multi-objective loss function to project isolated embeddings into a unified joint space.
6. **Inference & Scoring (`6_inference.py`)** Calculates Euclidean distances between aligned entity vectors to predict valid structural links.
7. **Evaluation (`7_results_roc.py`)** Evaluates the JSpace similarity scores against baseline text metrics and generates ROC curve visualizations.

## Requirements
* `torch`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `google-genai`
* `dicee`
* `rdflib`

## Usage
Run the scripts sequentially. Ensure that your initial input files (`1_vertices`, `2_vertices`) and the reference links (`links.csv`, `gold_standard.csv`) are located in the root directory.
