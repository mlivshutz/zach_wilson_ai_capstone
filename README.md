# AI-Powered GitHub Knowledge Base System

## Overview

An intelligent, event-driven system that automatically ingests GitHub repositories into a vector database and provides RAG-powered chat capabilities. The system uses Google Sheets as a user interface, n8n for automation, Databricks for processing, and Zilliz Cloud for vector storage.

## 📦 Source Code

The complete source code for this system is available at: [https://github.com/mlivshutz/zach_wilson_ai_capstone.git](https://github.com/mlivshutz/zach_wilson_ai_capstone.git)

## 🏗️ System Architecture

### Core Components

- **Google Sheets**: User interface for repository management
- **n8n Automation**: Event-driven workflow orchestration
- **Databricks Workflows**: Asynchronous repository processing
- **FastAPI Application**: RAG-powered chat API and web interface
- **Zilliz Cloud**: Managed vector database with dual indices
- **Vercel**: Serverless deployment platform

### Data Flow

1. **Repository Addition**: User adds repo to Google Sheets → n8n triggers Databricks workflow → Repository files fetched from GitHub → Text chunked and embedded via OpenAI → Vectors stored in Zilliz Cloud
2. **Repository Deletion**: Similar flow but removes all vectors associated with the repository
3. **Chat Interaction**: User queries via web interface → FastAPI retrieves relevant vectors → OpenAI generates contextual response

## 🚀 Key Features

### Automated Repository Ingestion
- **Event-Driven**: Google Sheets changes automatically trigger processing
- **Asynchronous Processing**: Databricks handles heavy computation without blocking
- **Multi-Format Support**: Processes `.py`, `.md`, `.sql`, `.yml` files
- **Intelligent Chunking**: 2000-character chunks with 200-character overlap

### Hybrid Vector Search
- **Dense Index**: OpenAI embeddings (1536-dim) for semantic similarity
- **Sparse Index**: BM25-like term vectors for keyword matching
- **Dual Retrieval**: Combines both approaches for comprehensive search

### RAG-Powered Chat
- **Context-Aware Responses**: Retrieves relevant code snippets before generation
- **Source Attribution**: Shows which repositories informed each answer
- **Re-ranking**: LLM-based relevance scoring for better results
- **Real-time Interface**: Modern web UI with typing indicators

## 🛠️ Technical Stack

### Frontend
- **HTML5/CSS3**: Responsive web interface
- **Vanilla JavaScript**: Chat functionality and file uploads
- **Real-time Updates**: System health monitoring

### Backend
- **FastAPI**: Async Python web framework
- **Pydantic**: Request/response validation
- **Uvicorn**: ASGI server for production deployment

### AI/ML Pipeline
- **OpenAI API**: Text embeddings (`text-embedding-3-small`) and chat completions (`gpt-4.1-mini`)
- **Custom RAG Pipeline**: Retrieval → Re-ranking → Augmented Generation
- **Hybrid Search**: Dense + sparse vector retrieval

### Data Processing
- **Databricks**: Distributed notebook execution
- **PyGithub**: Repository file extraction
- **Text Chunking**: Overlap-based segmentation
- **MMH3 Hashing**: Deterministic document IDs

### Vector Storage
- **Zilliz Cloud**: Managed Milvus vector database
- **COSINE Similarity**: Dense vector search (IVF_FLAT index)
- **IP Similarity**: Sparse vector search (SPARSE_INVERTED_INDEX)

### Automation & Deployment
- **n8n**: Visual workflow automation
- **Google Sheets API**: User interface integration
- **Databricks Jobs API**: Async workflow triggers
- **Vercel**: Serverless FastAPI deployment

## 📊 Architecture Diagrams

The `documents/` folder contains detailed Mermaid diagrams:
- `01_system_overview.mmd`: Complete system architecture
- `02_data_flow_sequence.mmd`: Step-by-step workflow sequences
- `03_component_architecture.mmd`: C4 component relationships
- `04_technical_stack.mmd`: Technology stack breakdown

## 🔧 Setup & Configuration

### Prerequisites
- OpenAI API key
- Zilliz Cloud account and cluster
- GitHub Personal Access Token
- Databricks workspace
- n8n instance
- Google Sheets with appropriate webhooks

### Environment Variables
```env
OPENAI_API_KEY=your_openai_key
ZILLIZ_CLOUD_URI=https://your-cluster.zillizcloud.com
ZILLIZ_API_KEY=your_zilliz_key
GITHUB_PAT=your_github_token
```

### Installation
```bash
pip install -r requirements.txt
python index.py
```

## 🎯 Use Cases

### Code Knowledge Base
- **Repository Documentation**: Automatically index entire codebases
- **Code Search**: Find relevant functions, classes, and documentation
- **Technical Q&A**: Ask questions about implementation details

### Team Collaboration
- **Onboarding**: Help new developers understand existing codebases
- **Code Reviews**: Quickly find related code and documentation
- **Knowledge Sharing**: Centralized access to team repositories

### Research & Analysis
- **Pattern Discovery**: Find similar implementations across repositories
- **Best Practices**: Identify common approaches and standards
- **Dependency Analysis**: Understand inter-repository relationships

## 🔄 Workflow Examples

### Adding a Repository
1. Open Google Sheets interface
2. Add new row with `owner/repo` format
3. n8n automatically detects change and triggers Databricks
4. Repository is processed and added to vector database
5. Chat interface immediately has access to new content

### Querying the Knowledge Base
1. Access web interface at deployed URL
2. Ask questions about any indexed repository
3. System retrieves relevant code snippets and documentation
4. Receives AI-generated response with source attribution

## 📈 Performance & Scalability

- **Async Processing**: Non-blocking repository ingestion
- **Distributed Computing**: Databricks for scalable text processing
- **Managed Infrastructure**: Zilliz Cloud auto-scales with demand
- **Serverless Deployment**: Vercel handles traffic spikes automatically
- **Efficient Storage**: Dual indices optimize for both semantic and keyword search

## 🔐 Security Features

- **API Key Management**: Secure environment variable handling
- **Input Validation**: Pydantic models prevent injection attacks
- **Access Control**: Repository-level permissions via GitHub tokens
- **Prompt Injection Protection**: Keyword filtering for chat inputs

This system demonstrates a modern, cloud-native approach to building intelligent knowledge management systems with event-driven architecture and state-of-the-art AI capabilities.