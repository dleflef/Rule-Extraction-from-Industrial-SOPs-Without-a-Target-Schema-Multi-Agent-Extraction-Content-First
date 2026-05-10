# Agentic Knowledge Graph for Digital Twins

This repository presents the development of a Multi-Agent System (MAS) that automates the construction of Knowledge Graphs (KGs). The goal is to use these graphs as the knowledge foundation for industrial Digital Twins by combining static technical documentation with dynamic operational data.

## Project Overview

Building high-quality knowledge graphs is usually expensive, difficult to scale, and hard to maintain over time. This project tackles those challenges by using a coordinated system of intelligent agents that work together to extract, structure, and validate knowledge automatically.

### Objectives

- **Automated Construction:** Develop a multi-agent pipeline that builds knowledge graphs from heterogeneous technical sources such as PDFs, HTML files, and tables.
- **Conflict Resolution:** Apply majority voting and configurable confidence thresholds to improve data reliability.
- **Digital Twin Integration:** Create a proof-of-concept digital twin that connects structured specifications with simulated IoT data.
- **Performance Evaluation:** Compare this agent-based approach with traditional centralized NLP pipelines.

---

## System Architecture

The system is designed as a sequence of specialized agents that transform unstructured technical content into a structured and validated knowledge graph.

### Extraction Pipeline

1. **Document Parser**  
   Preprocesses and segments technical documentation from different formats.

2. **Entity Extractor**  
   Performs domain-specific Named Entity Recognition (NER) to identify components, processes, materials, and parameters.

3. **Relation Extractor**  
   Detects and classifies relationships such as `part_of`, `controls`, `monitors`, and `depends_on`.

4. **Ontology Alignment Agent**  
   Maps extracted entities and relations to a predefined domain ontology.

5. **Validation Agent**  
   Checks logical consistency and resolves conflicts between competing triples.

---

## Methodology

### Conflict Resolution

Each extracted triple is assigned a composite reliability score based on:

- The confidence score of the extracting agent  
- Ontological consistency  
- Agreement with existing knowledge graph entries  

This scoring mechanism helps filter unreliable information while preserving useful knowledge.

### Digital Twin Synchronization

The system supports bidirectional synchronization between the static knowledge graph and a dynamic simulation layer. Simulated IoT data updates the graph, while the structured model provides context for interpreting runtime events.

---

## Tech Stack (Preliminary)

- **Graph Database:** Neo4j (Cypher queries)
- **Agent Frameworks (under evaluation):** LangGraph, CrewAI, AutoGen
- **LLMs:** Nvidia Nemotron and other models for NER and relation extraction
- **Visualization:** Neo4j Bloom and a custom interface

---

