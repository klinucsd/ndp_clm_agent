# California Landscape Metrics Analysis Agent

An intelligent AI agent system for analyzing California Landscape Metrics datasets through natural language queries. The system provides multiple interaction interfaces ranging from simple dataset search to advanced multi-agent coordination with interactive visualizations.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Available Interfaces](#available-interfaces)
- [MCP Server](#mcp-server)
- [Configuration](#configuration)
- [Example Queries](#example-queries)
- [Model Options](#model-options)
- [Advanced Features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains three progressively sophisticated AI agent implementations for California Landscape Metrics analysis:

1. **Simple Agent** (`simple_clm_agent.ipynb`) - Basic dataset search with conversation history
2. **Enhanced Agent** (`enhanced_clm_agent.ipynb`) - Full analysis capabilities with maps and charts
3. **Advanced Multi-Agent** (`advanced_clm_agent.ipynb`) - LangGraph-powered system with Data Commons integration

All agents use a FastMCP server that provides tools for:
- RAG-based semantic dataset discovery
- Zonal statistics computation
- Threshold-based area calculations
- Value distribution analysis
- Interactive map generation

## Features

### Core Capabilities
- **Semantic Dataset Search**: RAG-based vector search across 189+ California datasets
- **Zonal Statistics**: Compute mean, median, min, max, std for county boundaries
- **Threshold Analysis**: Calculate areas and percentages above/below thresholds
- **Distribution Analysis**: Generate histograms and value distributions
- **Interactive Visualizations**: WMS-based maps with legends and distribution charts
- **Conversation Memory**: Maintains context for follow-up questions
- **Multi-Model Support**: Works with OpenAI GPT-4o-mini or NRP Qwen3

### Advanced Features (advanced_clm_agent.ipynb)
- **Multi-Agent Coordination**: Intelligent routing between CLM and Data Commons agents
- **LangSmith Tracing**: Full observability with execution traces and cost tracking
- **Combined Analysis**: Integrates environmental and demographic data
- **Smart Routing**: Confidence-based agent selection for optimal results

## Repository Structure

```
├── simple_clm_agent.ipynb          # Basic dataset search agent
├── enhanced_clm_agent.ipynb        # Full-featured analysis agent with visualizations
├── advanced_clm_agent.ipynb        # Multi-agent system with LangGraph
├── clm_mcp_server.py               # FastMCP server with analysis tools
└── README.md                       # This file
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/california-landscape-metrics-agent.git
cd california-landscape-metrics-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

**For Simple Agent:**
```bash
pip install python-dotenv ipywidgets pydantic-ai fastmcp openai nest-asyncio
```

**For Enhanced Agent:**
```bash
pip install python-dotenv ipywidgets pydantic-ai fastmcp openai nest-asyncio folium matplotlib markdown
```

**For Advanced Multi-Agent:**
```bash
pip install langgraph langchain-openai langchain-core langsmith python-dotenv aiohttp folium matplotlib markdown nest-asyncio ipywidgets pandas numpy
```

### 4. Configure API Keys

Create a `.env` file in the project root:
```env
# Required for all agents
OPENAI_API_KEY=your_openai_key_here

# Optional: for NRP Qwen3 model
NRP_API_KEY=your_nrp_key_here

# Optional: for Data Commons integration (advanced agent only)
DC_API_KEY=your_data_commons_key_here

# Optional: for LangSmith tracing (advanced agent only)
LANGCHAIN_API_KEY=your_langsmith_key_here
```

## Available Interfaces

### 1. Simple Dataset Search Agent
**File:** `simple_clm_agent.ipynb`

**Best For:** Learning agent basics, quick dataset discovery

**Features:**
- Semantic dataset search
- Dataset metadata retrieval
- Conversation history
- Simple chat interface

**Example Usage:**
```python
# In notebook
MODEL = "openai"  # or "nrp"
# Run all cells, then ask questions like:
# "Find datasets about carbon turnover"
# "What are the units for this dataset?"
```

### 2. Enhanced Analysis Agent
**File:** `enhanced_clm_agent.ipynb`

**Best For:** Comprehensive data analysis, production use

**Features:**
- All simple agent features
- Zonal statistics computation
- Threshold-based analysis
- Interactive WMS maps with legends
- Distribution charts and histograms
- Multi-county comparisons
- Markdown-formatted responses

**Example Usage:**
```python
# In notebook
MODEL = "openai"  # or "nrp"
# Run all cells, then try queries like:
# "What is the average carbon turnover time in Los Angeles?"
# "Show me the unemployment distribution for San Diego and Orange counties"
# "Show me a map of annual burn probability"
```

### 3. Advanced Multi-Agent System
**File:** `advanced_clm_agent.ipynb`

**Best For:** Complex queries, research, multi-source analysis

**Features:**
- All enhanced agent features
- Intelligent agent routing (CLM vs Data Commons)
- LangGraph workflow orchestration
- LangSmith observability and tracing
- Multi-source data integration
- Cost tracking and optimization
- Real-time trace visualization

**Setup:**
1. Start Data Commons MCP server:
```bash
wget https://astral.sh/uv/install.sh
sh install.sh
export DC_API_KEY=your_data_commons_key
uv tool run datacommons-mcp serve http --port 3000 &
```

2. Configure LangSmith:
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
os.environ["LANGCHAIN_PROJECT"] = "your_project"
```

3. Run queries like:
```
# "What is the unemployment rate in San Diego?" → Routes to Data Commons
# "Show carbon turnover distribution in LA" → Routes to CLM
# "Compare burn probability and population density" → Uses both agents
```

## MCP Server

### Overview
The `clm_mcp_server.py` provides four main tools via FastMCP:

### Tools

#### 1. `search_datasets`
Semantic search across 189+ datasets using RAG-based vector search.

```python
result = search_datasets(
    query="carbon turnover time",
    top_k=3
)
```

#### 2. `compute_zonal_stats`
Calculate statistics for geographic features.

```python
result = compute_zonal_stats(
    wcs_base_url="https://sparcal.sdsc.edu/geoserver",
    wfs_base_url="https://sparcal.sdsc.edu/geoserver/boundary/wfs",
    wcs_coverage_id="rrk__cstocks_turnovertime_202009_202312_t1_v5",
    feature_id="boundary:ca_counties",
    filter_column="name",
    filter_value=["Los Angeles", "San Diego"],
    stats=["mean", "median", "min", "max", "std"]
)
```

#### 3. `zonal_count`
Count pixels above/below thresholds.

```python
result = zonal_count(
    wcs_base_url="https://sparcal.sdsc.edu/geoserver",
    wfs_base_url="https://sparcal.sdsc.edu/geoserver/boundary/wfs",
    wcs_coverage_id="rrk__cstocks_turnovertime_202009_202312_t1_v5",
    feature_id="boundary:ca_counties",
    filter_column="name",
    filter_value="San Diego",
    threshold=100.0
)
```

#### 4. `zonal_distribution`
Get value distributions for histogram/chart generation.

```python
result = zonal_distribution(
    wcs_base_url="https://sparcal.sdsc.edu/geoserver",
    wfs_base_url="https://sparcal.sdsc.edu/geoserver/boundary/wfs",
    wcs_coverage_id="rrk__cstocks_turnovertime_202009_202312_t1_v5",
    feature_id="boundary:ca_counties",
    filter_column="name",
    filter_value=["San Diego", "Los Angeles"],
    num_bins=10,
    global_bins=True
)
```

### Running the MCP Server
The server is hosted at `https://wenokn.fastmcp.app/mcp` by default. To run locally:

```bash
python clm_mcp_server.py
```

## Configuration

### GeoServer Endpoints
```python
CLM_CONFIG = {
    "wcs_base_url": "https://sparcal.sdsc.edu/geoserver",
    "wfs_base_url": "https://sparcal.sdsc.edu/geoserver/boundary/wfs",
    "feature_id": "boundary:ca_counties",
    "filter_column": "name"
}
```

### MCP Endpoints
- **CLM Server**: `https://wenokn.fastmcp.app/mcp`
- **Data Commons Server** (advanced only): `http://localhost:3000/mcp`

### Model Selection
Choose between OpenAI and NRP models in each notebook:

```python
MODEL = "openai"  # Fast, reliable, ~$0.001 per query
# or
MODEL = "nrp"     # Free, open-source, slightly slower
```

## Example Queries

### Simple Agent
```
- "Find datasets about carbon turnover"
- "What datasets are available for burn probability?"
- "What are the units for this dataset?"
- "Tell me more about this dataset"
```

### Enhanced Agent
```
- "What is the average carbon turnover time in Los Angeles?"
- "Find the maximum annual burn probability in San Diego county"
- "Which county has the highest burn probability?"
- "Show me the unemployment distribution for San Diego and Los Angeles"
- "Display a map of this dataset"
- "What percentage of San Diego has carbon turnover time above 100 years?"
- "Rank the top 5 counties by mean carbon turnover time"
```

### Advanced Multi-Agent
```
- "What is the population of Sacramento?" (→ Data Commons)
- "Show carbon turnover distribution in LA" (→ CLM)
- "Compare unemployment rate to burn probability in San Diego" (→ Both)
- "What counties have high fire risk and high population?" (→ Both)
```

## Model Options

### OpenAI GPT-4o-mini
- **Pros**: Fast (1-3s), reliable, well-tested
- **Cons**: Requires paid API key (~$0.001 per query)
- **Best for**: Production use, quick responses

### NRP Qwen3
- **Pros**: Free, open-source, privacy-friendly
- **Cons**: Slower (5-10s), requires NRP access
- **Best for**: Research, cost-sensitive applications

## Advanced Features

### LangSmith Tracing (Advanced Agent)
View detailed execution traces at https://smith.langchain.com/:
- Visual workflow graphs
- Token usage and costs
- Latency for each step
- All tool calls and responses
- Debug information

### Agent Routing Logic
The advanced agent uses confidence-based routing:

```
Query Analysis
    ↓
├─→ CLM Agent (California environmental data)
│   • Spatial patterns, distributions
│   • 30m × 30m resolution
│   • Fire, carbon, water, biodiversity
│
├─→ Data Commons Agent (Global demographics)
│   • Population, income, health
│   • Aggregated rates and totals
│   • Any location worldwide
│
└─→ Both Agents (Complex queries)
    • Combines environmental + demographic
    • Cross-references datasets
    • Synthesizes multi-source insights
```

### Visualization Capabilities
- **Maps**: Interactive WMS layers with legends and units
- **Charts**: Matplotlib-generated histograms and distributions
- **Comparisons**: Side-by-side multi-county analysis

## Contributing

Contributions are welcome! Areas for improvement:
1. Additional data sources beyond CLM and Data Commons
2. More sophisticated visualization options
3. Enhanced caching and performance optimization
4. Additional statistical analysis methods
5. Export capabilities (PDF, CSV, GeoJSON)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See individual notebook files for detailed guides

---

**Note**: This is a research project. Results should be validated before use in production or policy decisions.
