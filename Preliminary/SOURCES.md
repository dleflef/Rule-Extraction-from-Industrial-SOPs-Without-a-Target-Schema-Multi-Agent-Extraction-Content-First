# Sources and attribution for the Preliminary folder

Everything in `Preliminary/` is exploratory material kept from the early stages of
the project. None of it is part of the pipeline reported in this repository, and
none of it is used to produce any reported result.

Most of these files are not original work. They are course and workshop notebooks
worked through while learning the tools, retained here only as a record of that
preparation. This file records where each one comes from. Every notebook and
support module also carries the same attribution in its own header. Rights in the
original material remain with its authors, and the licence of each source governs
its reuse.

## DeepLearning.AI short courses

### Document AI: From OCR to Agentic Doc Extraction (with LandingAI)

<https://www.deeplearning.ai/courses/document-ai-from-ocr-to-agentic-doc-extraction>

| File | Origin |
| --- | --- |
| `notebooks/Agentic_Doc_Extraction/Document_processing_OCR.ipynb` | Lab 1 |
| `notebooks/Agentic_Doc_Extraction/PaddleOCR.ipynb` | Lab 2 |
| `notebooks/Agentic_Doc_Extraction/agentic_document_extraction.ipynb` | Lab 4, first part |
| `notebooks/Agentic_Doc_Extraction/agentic_document_extraction_2.ipynb` | Lab 4, second part |
| `notebooks/Agentic_Doc_Extraction/extraction_RAG.ipynb` | Lab 5 |
| `notebooks/Agentic_Doc_Extraction/helper.py` | Course support code; parts follow the LandingAI sample at <https://docs.landing.ai/ade/ade-python> |

### Preprocessing Unstructured Data for LLM Applications (with Unstructured)

Taught by Matthew Robinson.
<https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/>

| File | Origin |
| --- | --- |
| `notebooks/Agentic_Doc_Extraction/metadata_extraction_chunking.ipynb` | Lesson 3 |
| `notebooks/Agentic_Doc_Extraction/preprocessing_pdf.ipynb` | Lesson 4 |
| `notebooks/Agentic_Doc_Extraction/extracting_tables.ipynb` | Lesson 5 |
| `notebooks/Agentic_Doc_Extraction/RAG_bot.ipynb` | Lesson 6 |

### AI Agents in LangGraph (with LangChain and Tavily)

Taught by Harrison Chase and Rotem Weiss.
<https://www.deeplearning.ai/courses/ai-agents-in-langgraph>

| File | Origin |
| --- | --- |
| `notebooks/Data_Agents/build_agent.ipynb` | Lesson 1, which itself follows the ReAct pattern described at <https://til.simonwillison.net/llms/python-react-pattern> |
| `notebooks/Data_Agents/LangGraph_components.ipynb` | Lesson 2 |
| `notebooks/Data_Agents/agentic_search_tools.ipynb` | Lesson 3 |
| `notebooks/Data_Agents/persistence_streaming.ipynb` | Lesson 4 |
| `notebooks/Data_Agents/essay_writer.ipynb` | Lesson 6 |

### Building and Evaluating Data Agents (with Snowflake)

<https://www.deeplearning.ai/short-courses/building-and-evaluating-data-agents/>

| File | Origin |
| --- | --- |
| `notebooks/Data_Agents/multi_agent_workflow.ipynb` | Lesson 2 |
| `notebooks/Data_Agents/cortex_agent.ipynb` | Lesson 3 |
| `notebooks/Data_Agents/agent_performance.ipynb` | Lesson 4 |
| `notebooks/Data_Agents/measure_gpa.ipynb` | Lesson 5 |
| `notebooks/Data_Agents/improve_agent_GPA.ipynb` | Lesson 6 |

### Agentic Knowledge Graph Construction (with Neo4j)

Taught by Andreas Kollegger. Course:
<https://www.deeplearning.ai/courses/agentic-knowledge-graph-construction>
Accompanying code: <https://github.com/neo4j-contrib/agentic-kg> (MIT licence).

| File | Origin |
| --- | --- |
| `notebooks/Google_adk_intro.ipynb` | Lesson 3, first part |
| `notebooks/Google_adk_multi_agents.ipynb` | Lesson 3, second part |
| `notebooks/user_intent.ipynb` | Lesson 4 |
| `notebooks/schema_proposal_structured.ipynb` | Lesson 6 |
| `notebooks/schema_proposal_unstructured.ipynb` | Lesson 7 |
| `notebooks/kg_construction_markdown_files.ipynb` | Lesson 8 |
| `notebooks/py_helper/helper.py` | Course support code |
| `notebooks/py_helper/neo4j_for_adk.py` | Course support code |
| `notebooks/py_helper/tools.py` | Course support code |
| `notebooks/py_helper/construction_plan.json` | Course data file |

## NODES 2021 workshop, C. J. Sullivan

*Creating a Knowledge Graph with Neo4j: A Simple Machine Learning Approach*.
Materials: <https://github.com/cj2001/nodes2021_kg_workshop>. The upstream
repository states no licence, so these files are reproduced here for reference
only.

| File | Origin |
| --- | --- |
| `notebooks/graph_data_science.ipynb` | `notebooks/02-graph_data_science.ipynb` |
| `notebooks/wikidata_kg.ipynb` | `notebooks/03-wikidata_kg.ipynb` |
| `notebooks/wikidata_kg_data_science.ipynb` | `notebooks/04-wikidata_kg_data_science.ipynb` |
| `json_files/svo.json` | `json_files/svo.json` |
| `json_files/wiki.json` | `json_files/wiki.json` |

## Original material

`example1/` is an early prototype written for this project: a Streamlit front end
over a small LangGraph pipeline for industrial document analysis, together with
its sample data and a hallucination check. No external source was identified for
it. It relies on the usual third-party libraries, which carry their own licences,
and its structure is not taken from any of the courses listed above.
