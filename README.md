# California Landscape Metrics Analysis Agent

This repository contains an AI agent designed to analyze California Landscape Metrics datasets. The agent provides tools to dynamically discover and query geospatial datasets, compute zonal statistics, and calculate areas based on thresholds for California counties. It includes a FastMCP server for dataset operations and an interactive Jupyter notebook chat interface for user queries.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the FastMCP Server](#running-the-fastmcp-server)
  - [Running the Command-Line Agent](#running-the-command-line-agent)
  - [Using the Interactive Chat Interface](#using-the-interactive-chat-interface)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Example Queries](#example-queries)
- [Contributing](#contributing)
- [License](#license)

## Overview
The California Landscape Metrics Analysis Agent leverages geospatial data services (WCS and WFS) to provide insights into metrics such as carbon turnover time, wildfire hazard, vegetation height, and more. It uses a FastMCP server for dataset operations and a Pydantic AI agent for natural language query processing. The interactive chat interface, implemented in a Jupyter notebook, allows users to ask questions and receive answers with dataset details.

## Features
- **Dynamic Dataset Discovery**: Searches for relevant datasets using a RAG-based vector search.
- **Zonal Statistics**: Computes statistics (mean, median, min, max, std) for raster data within county boundaries.
- **Threshold Analysis**: Calculates areas and percentages above or below specified thresholds for single counties.
- **Interactive Chat Interface**: A Jupyter notebook-based UI for querying datasets in natural language.
- **Scalable Processing**: Supports parallel processing with configurable retries and timeouts.

## Repository Structure
```
├── clm_mcp_server.py       # FastMCP server with tools for dataset search and geospatial analysis
├── clm_agent.py            # Pydantic AI agent for dynamic dataset discovery and query processing
├── ndp_clm_agent_chatbox.ipynb  # Jupyter notebook with interactive chat interface
└── README.md               # This file
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/california-landscape-metrics-agent.git
   cd california-landscape-metrics-agent
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install fastmcp pydantic-ai openai requests ipywidgets nest-asyncio python-dotenv
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root.
   - Add your OpenAI API key:
     ```env
     OPENAI_API_KEY=your-api-key-here
     ```

## Usage

### Running the FastMCP Server
The `clm_mcp_server.py` script runs a FastMCP server that provides tools for dataset search and geospatial analysis.

1. Start the server:
   ```bash
   python clm_mcp_server.py
   ```
   The server will run on `http://127.0.0.1:8800/mcp` by default.

### Running the Command-Line Agent
The `clm_agent.py` script provides a command-line interface to test the agent with predefined questions.

1. Ensure the FastMCP server is running (or update `mcp_client` in `clm_agent.py` to use a remote server: `Client("https://wenokn.fastmcp.app/mcp")`).
2. Run the agent:
   ```bash
   python clm_agent.py
   ```
   The script will process a set of example questions and display the results.

### Using the Interactive Chat Interface
The `ndp_clm_agent_chatbox.ipynb` Jupyter notebook provides an interactive chat interface.

1. Install Jupyter if not already installed:
   ```bash
   pip install jupyter
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `ndp_clm_agent_chatbox.ipynb` in the Jupyter interface.
4. Run all cells to set up the agent and display the chat interface.
5. Enter questions in the text area and click "Send" to get answers.

**Note**: Ensure the OpenAI API key is set in the environment or directly in the notebook before running.

## Dependencies
- Python 3.10+
- `fastmcp`: For MCP server and client functionality
- `pydantic-ai`: For AI agent implementation
- `openai`: For GPT-4o-mini model integration
- `requests`: For HTTP requests to geospatial services
- `ipywidgets`: For Jupyter notebook UI
- `nest-asyncio`: For nested asyncio in Jupyter
- `python-dotenv`: For environment variable management

## Configuration
- **GeoServer Endpoints**:
  - Web Coverage Service (WCS): `https://sparcal.sdsc.edu/geoserver`
  - Web Feature Service (WFS): `https://sparcal.sdsc.edu/geoserver/boundary/wfs`
  - Feature ID: `boundary:ca_counties`
  - Filter Column: `name`
- **MCP Server**: Defaults to `http://127.0.0.1:8800/mcp` (local) or `https://wenokn.fastmcp.app/mcp` (remote).
- **OpenAI API Key**: Set via `.env` file or environment variable `OPENAI_API_KEY`.

## Example Queries
- "What is the average carbon turnover time in Los Angeles County?"
- "Which county has the highest wildfire hazard?"
- "What is the burn probability in San Diego County?"
- "What is the average vegetation height in Mendocino County?"
- "Show me tree canopy cover statistics for Sacramento County."

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.