# ============================================================================
# LangGraph Multi-Agent System with LangSmith Tracing
# Converted from Pydantic AI notebook
# ============================================================================

"""
Cell 1: Installation and Setup
"""

# ============================================================================
# COMPLETE INSTALLATION COMMAND - Run this first!
# ============================================================================

!pip install langgraph langchain-openai langchain-core langsmith \
    python-dotenv aiohttp folium matplotlib markdown nest-asyncio \
    ipywidgets pandas numpy

"""
Cell 2: Environment Setup and Configuration
"""

from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence, Literal
import operator
from datetime import datetime

# Load environment variables
load_dotenv()

# ============================================================================
# LangSmith Configuration - THIS IS THE MAGIC! 🎯
# ============================================================================

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")  # Add to .env
os.environ["LANGCHAIN_PROJECT"] = "CLM-DataCommons-MultiAgent"

print("✅ LangSmith tracing enabled!")
print(f"   Project: {os.environ['LANGCHAIN_PROJECT']}")
print(f"   View traces at: https://smith.langchain.com/")

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DC_API_KEY = os.getenv('DC_API_KEY')
NRP_API_KEY = os.getenv('NRP_API_KEY')

# Base configuration for CLM GeoServer
CLM_CONFIG = {
    "wcs_base_url": "https://sparcal.sdsc.edu/geoserver",
    "wfs_base_url": "https://sparcal.sdsc.edu/geoserver/boundary/wfs",
    "feature_id": "boundary:ca_counties",
    "filter_column": "name"
}

# MCP Configuration
MCP_URL = "https://wenokn.fastmcp.app/mcp"
DC_MCP_URL = "http://localhost:3000/mcp"

print("✅ Configuration loaded")
print(f"   OpenAI: {'✓' if OPENAI_API_KEY else '✗'}")
print(f"   Data Commons: {'✓' if DC_API_KEY else '✗'}")
print(f"   NRP: {'✓' if NRP_API_KEY else '✗'}")

"""
Cell 3: State Definition (LangGraph Core Concept)
"""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import Literal

class AgentState(TypedDict):
    """
    The state of our multi-agent system.
    This gets passed between all nodes in the graph.
    """
    # Core conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Routing decision
    next_agent: Literal["clm", "dc", "both", "END"]
    
    # Agent responses
    clm_response: str
    dc_response: str
    
    # Visualizations
    map_data: dict | None
    distribution_data: dict | None
    
    # Metadata
    question: str
    routing_confidence: float
    routing_reasoning: str

print("✅ State schema defined")

"""
Cell 4: MCP Client Setup
"""

import aiohttp
import json
from typing import Dict, Any, List

class MCPClient:
    """MCP Client with automatic LangSmith tracing."""
    
    def __init__(self, url: str, name: str = "MCP"):
        self.url = url
        self.name = name
        self.session: aiohttp.ClientSession | None = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()
    
    async def _parse_sse_response(self, resp: aiohttp.ClientResponse) -> List[Dict]:
        """Parse server-sent events response."""
        messages = []
        buffer = ""
        
        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                buffer += line[6:] + "\n"
            elif line == "data: [DONE]":
                if buffer.strip():
                    try:
                        messages.append(json.loads(buffer.strip()))
                    except json.JSONDecodeError:
                        pass
                buffer = ""
                break
        
        if buffer.strip():
            try:
                messages.append(json.loads(buffer.strip()))
            except json.JSONDecodeError:
                pass
        
        return messages
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """
        Call MCP tool.
        LangSmith will automatically trace this as a tool call!
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        async with self.session.post(self.url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"MCP HTTP {resp.status}")
            
            msgs = await self._parse_sse_response(resp)
            if not msgs:
                raise RuntimeError("Empty MCP response")
            
            result = msgs[0].get("result", {})
            
            # Extract text content
            text_parts = [
                block.get("text", "")
                for block in result.get("content", [])
                if block.get("type") == "text"
            ]
            
            return "\n".join(text_parts) or json.dumps(result, indent=2)

# Initialize MCP clients
clm_mcp = MCPClient(MCP_URL, "CLM-MCP")
dc_mcp = MCPClient(DC_MCP_URL, "DC-MCP")

print("✅ MCP clients initialized")

"""
Cell 5: CLM Agent Tools (as LangChain Tools)
"""

from langchain_core.tools import tool
from langsmith import traceable

@tool
@traceable(name="search_clm_datasets")
async def search_clm_datasets(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search for California Landscape Metrics datasets.
    
    Args:
        query: Search query for datasets
        top_k: Number of top results to return
    """
    async with clm_mcp:
        result = await clm_mcp.call_tool(
            "search_datasets",
            {"query": query, "top_k": top_k}
        )
        
        # Parse result
        try:
            data = json.loads(result) if isinstance(result, str) else result
            if data.get('success') and data.get('datasets'):
                return {
                    'success': True,
                    'selected': data['datasets'][0],
                    'alternatives': data['datasets'][1:]
                }
        except:
            pass
        
        return {'success': False, 'error': 'No datasets found'}

@tool
@traceable(name="get_clm_statistics")
async def get_clm_statistics(
    coverage_id: str,
    counties: List[str] | None = None,
    stats: List[str] | None = None
) -> Dict[str, Any]:
    """
    Get statistical measures for CLM dataset.
    
    Args:
        coverage_id: WCS coverage ID for the dataset
        counties: List of county names (None for all)
        stats: Statistics to compute (default: mean, median, min, max, std)
    """
    if stats is None:
        stats = ["mean", "median", "min", "max", "std"]
    
    async with clm_mcp:
        result = await clm_mcp.call_tool(
            "compute_zonal_stats",
            {
                **CLM_CONFIG,
                "wcs_coverage_id": coverage_id,
                "filter_value": counties,
                "stats": stats
            }
        )
        
        try:
            return json.loads(result) if isinstance(result, str) else result
        except:
            return {'success': False, 'error': str(result)[:200]}

@tool
@traceable(name="get_clm_distribution")
async def get_clm_distribution(
    coverage_id: str,
    counties: List[str] | None = None,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Get value distribution for CLM dataset.
    
    Args:
        coverage_id: WCS coverage ID
        counties: County names (None for all)
        num_bins: Number of histogram bins
    """
    async with clm_mcp:
        result = await clm_mcp.call_tool(
            "zonal_distribution",
            {
                **CLM_CONFIG,
                "wcs_coverage_id": coverage_id,
                "filter_value": counties,
                "num_bins": num_bins,
                "global_bins": True,
                "categorical_threshold": 20
            }
        )
        
        try:
            data = json.loads(result) if isinstance(result, str) else result
            if data.get('success'):
                data['action'] = 'show_distribution'
            return data
        except:
            return {'success': False, 'error': str(result)[:200]}

# List of CLM tools
clm_tools = [search_clm_datasets, get_clm_statistics, get_clm_distribution]

print("✅ CLM tools defined")

"""
Cell 6: Data Commons Agent Tools
"""

@tool
@traceable(name="search_dc_indicators")
async def search_dc_indicators(
    query: str,
    places: List[str] | None = None,
    parent_place: str | None = None
) -> str:
    """
    Search for indicators in Google Data Commons.
    
    Args:
        query: Search query
        places: List of place names
        parent_place: Parent geographic area
    """
    async with dc_mcp:
        args = {
            "query": query,
            "include_topics": True,
            "maybe_bilateral": False
        }
        if places:
            args["places"] = places
        if parent_place:
            args["parent_place"] = parent_place
        
        return await dc_mcp.call_tool("search_indicators", args)

@tool
@traceable(name="get_dc_observations")
async def get_dc_observations(
    variable_dcid: str,
    place_dcid: str,
    child_place_type: str | None = None,
    date: str = "latest"
) -> str:
    """
    Get observations from Google Data Commons.
    
    Args:
        variable_dcid: Variable DCID
        place_dcid: Place DCID
        child_place_type: Type of child places
        date: Date (default: "latest")
    """
    async with dc_mcp:
        args = {
            "variable_dcid": variable_dcid,
            "place_dcid": place_dcid,
            "date": date
        }
        if child_place_type:
            args["child_place_type"] = child_place_type
        
        return await dc_mcp.call_tool("get_observations", args)

# List of DC tools
dc_tools = [search_dc_indicators, get_dc_observations]

print("✅ Data Commons tools defined")

"""
Cell 7: LangGraph Nodes - The Agent Logic
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================================
# Node 1: Router - Decides which agent(s) to use
# ============================================================================

@traceable(name="Router Node")
async def router_node(state: AgentState) -> AgentState:
    """
    Route the question to appropriate agent(s).
    This is automatically traced in LangSmith!
    """
    question = state["question"]
    
    # Define system message without template variables in JSON
    system_message = """You are a routing expert for a multi-agent system.

**Agent Capabilities:**

**CLM Agent** - California landscape/environmental data:
- 189 datasets: air quality, biodiversity, carbon, fire, water, poverty, unemployment
- 30m x 30m resolution spatial data
- Maps, distributions, county statistics
- California ONLY

**DC Agent** - Global demographic/economic data:
- Any location worldwide
- Population, income, health, economics
- Aggregated totals and rates

**Routing Rules:**
1. "distribution" / "map" / "spatial pattern" → CLM (if California topic)
2. "total count" / "how many people" → DC
3. "rate" / "percentage" without "distribution" → DC (actual demographic rate)
4. California environmental topics → CLM
5. Non-California locations → DC
6. If unsure → BOTH

Respond with JSON only (no markdown):
{{
    "agent": "clm" or "dc" or "both",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])
    
    response = await llm.ainvoke(
        router_prompt.format_messages(question=question)
    )
    
    # Parse routing decision
    try:
        import json
        decision = json.loads(response.content)
        next_agent = decision.get("agent", "both")
        confidence = decision.get("confidence", 0.5)
        reasoning = decision.get("reasoning", "")
    except:
        next_agent = "both"
        confidence = 0.5
        reasoning = "Failed to parse routing decision"
    
    print(f"🎯 Router Decision: {next_agent.upper()} (confidence: {confidence:.0%})")
    print(f"   Reasoning: {reasoning}")
    
    return {
        **state,
        "next_agent": next_agent,
        "routing_confidence": confidence,
        "routing_reasoning": reasoning,
        "messages": [response]
    }

# ============================================================================
# Node 2: CLM Agent - California Landscape Metrics
# ============================================================================

@traceable(name="CLM Agent Node")
async def clm_agent_node(state: AgentState) -> AgentState:
    """
    CLM Agent with tool calling.
    All tool calls are automatically traced in LangSmith!
    """
    question = state["question"]
    
    clm_prompt = f"""You are an expert in California Landscape Metrics datasets.

You have access to tools for:
1. Searching 189 CLM datasets
2. Computing statistics for counties
3. Getting value distributions

**Workflow:**
1. First call search_clm_datasets to find relevant dataset
2. Then use get_clm_statistics or get_clm_distribution as needed

**Important:**
- Always mention dataset name in response
- Clarify that CLM data is 30m x 30m spatial resolution
- For distributions, specify units and meaning

Answer this question: {question}"""
    
    # Create ReAct agent with tools
    # This gives us automatic tool calling with LangSmith tracing!
    clm_react_agent = create_react_agent(
        llm,
        clm_tools,
        state_modifier=clm_prompt
    )
    
    try:
        result = await clm_react_agent.ainvoke({
            "messages": [HumanMessage(content=question)]
        })
        
        response = result["messages"][-1].content
        
        # Check for visualizations in tool results
        map_data = None
        distribution_data = None
        
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                try:
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if tool_result.get("action") == "show_distribution":
                        distribution_data = tool_result
                except:
                    pass
        
        print(f"✅ CLM Agent completed")
        
        return {
            **state,
            "clm_response": response,
            "map_data": map_data,
            "distribution_data": distribution_data,
            "messages": state["messages"] + result["messages"]
        }
        
    except Exception as e:
        error_msg = f"CLM Agent error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "clm_response": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

# ============================================================================
# Node 3: Data Commons Agent
# ============================================================================

@traceable(name="DC Agent Node")
async def dc_agent_node(state: AgentState) -> AgentState:
    """
    Data Commons Agent with tool calling.
    Automatically traced in LangSmith!
    """
    question = state["question"]
    
    # Enhance question with location qualifier
    enhanced_question = question
    if not any(x in question.lower() for x in [', ca', ', usa', 'california', ' county']):
        # Try to extract location and add qualifier
        import re
        location_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
        if location_match:
            location = location_match.group(1)
            enhanced_question = question.replace(location, f"{location}, CA, USA")
    
    dc_prompt = f"""You are a precise data analyst using Google Data Commons.

**Workflow:**
1. Use search_dc_indicators to find relevant variable
2. Use get_dc_observations to get the data

**Rules:**
- Always qualify place names: "San Diego, CA, USA"
- Use date="latest" unless specified
- Provide actual demographic rates/totals

Answer this question: {enhanced_question}"""
    
    dc_react_agent = create_react_agent(
        llm,
        dc_tools,
        state_modifier=dc_prompt
    )
    
    try:
        result = await dc_react_agent.ainvoke({
            "messages": [HumanMessage(content=enhanced_question)]
        })
        
        response = result["messages"][-1].content
        
        print(f"✅ DC Agent completed")
        
        return {
            **state,
            "dc_response": response,
            "messages": state["messages"] + result["messages"]
        }
        
    except Exception as e:
        error_msg = f"DC Agent error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "dc_response": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

# ============================================================================
# Node 4: Combiner - Merge responses from both agents
# ============================================================================

@traceable(name="Combiner Node")
async def combiner_node(state: AgentState) -> AgentState:
    """
    Combine responses from CLM and DC agents.
    """
    question = state["question"]
    clm_resp = state.get("clm_response", "")
    dc_resp = state.get("dc_response", "")
    
    combiner_prompt = ChatPromptTemplate.from_messages([
        ("system", """You combine responses from specialized agents.

**Agent Types:**
- **CLM**: California spatial data (30m x 30m pixels) - shows geographic patterns
- **DC**: Demographic totals/rates - shows population statistics

**Your Task:**
1. Identify which responses are useful
2. Combine complementary information
3. Clarify differences (spatial patterns vs population rates)
4. Always mention dataset names

**Critical:**
- CLM "mean unemployment" = spatial average across pixels, NOT actual unemployment rate
- DC "unemployment rate" = actual demographic rate
- Explain this distinction when both agents respond"""),
        ("human", """Question: {question}

CLM Response: {clm_response}

DC Response: {dc_response}

Provide a clear, combined answer:""")
    ])
    
    response = await llm.ainvoke(
        combiner_prompt.format_messages(
            question=question,
            clm_response=clm_resp,
            dc_response=dc_resp
        )
    )
    
    print(f"✅ Combined responses")
    
    return {
        **state,
        "messages": state["messages"] + [response]
    }

print("✅ LangGraph nodes defined")

"""
Cell 8: Build the LangGraph
"""

from langgraph.graph import StateGraph, END

# ============================================================================
# Graph Construction - This is where the magic happens! 🎨
# ============================================================================

def build_agent_graph():
    """
    Build the multi-agent workflow graph.
    LangSmith will visualize this as a beautiful flowchart!
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("clm_agent", clm_agent_node)
    workflow.add_node("dc_agent", dc_agent_node)
    workflow.add_node("combiner", combiner_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Conditional routing based on router decision
    def route_after_router(state: AgentState) -> str:
        next_agent = state.get("next_agent", "both")
        if next_agent == "clm":
            return "clm_agent"
        elif next_agent == "dc":
            return "dc_agent"
        else:  # "both"
            return "clm_agent"  # Will run both in sequence
    
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "clm_agent": "clm_agent",
            "dc_agent": "dc_agent"
        }
    )
    
    # After CLM agent
    def route_after_clm(state: AgentState) -> str:
        if state.get("next_agent") == "both" and not state.get("dc_response"):
            return "dc_agent"
        elif state.get("next_agent") == "both":
            return "combiner"
        else:
            return END
    
    workflow.add_conditional_edges(
        "clm_agent",
        route_after_clm,
        {
            "dc_agent": "dc_agent",
            "combiner": "combiner",
            END: END
        }
    )
    
    # After DC agent
    def route_after_dc(state: AgentState) -> str:
        if state.get("next_agent") == "both" and state.get("clm_response"):
            return "combiner"
        else:
            return END
    
    workflow.add_conditional_edges(
        "dc_agent",
        route_after_dc,
        {
            "combiner": "combiner",
            END: END
        }
    )
    
    # After combiner
    workflow.add_edge("combiner", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Build the graph
agent_graph = build_agent_graph()

print("✅ LangGraph compiled!")
print("   View graph structure at: https://smith.langchain.com/")

"""
Cell 9: Run Function with Full Tracing
"""

@traceable(name="Multi-Agent Query")
async def run_multi_agent_query(question: str) -> Dict[str, Any]:
    """
    Run a question through the multi-agent system.
    Everything is automatically traced in LangSmith!
    
    Args:
        question: User's question
        
    Returns:
        Dictionary with output, visualizations, and metadata
    """
    print(f"\n{'='*80}")
    print(f"📝 Question: {question}")
    print(f"{'='*80}\n")
    
    # Initialize state
    initial_state = {
        "messages": [],
        "question": question,
        "next_agent": "both",
        "clm_response": "",
        "dc_response": "",
        "map_data": None,
        "distribution_data": None,
        "routing_confidence": 0.0,
        "routing_reasoning": ""
    }
    
    # Run the graph
    # LangSmith will create a beautiful trace showing:
    # - Router decision
    # - Which agents ran
    # - All tool calls
    # - Token usage
    # - Latency for each step
    final_state = await agent_graph.ainvoke(initial_state)
    
    # Extract final response
    final_message = final_state["messages"][-1]
    output = final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    result = {
        'output': output,
        'map_data': final_state.get('map_data'),
        'distribution_data': final_state.get('distribution_data'),
        'routing': {
            'agent': final_state.get('next_agent', 'unknown'),
            'confidence': final_state.get('routing_confidence', 0.0),
            'reasoning': final_state.get('routing_reasoning', '')
        },
        'clm_response': final_state.get('clm_response', ''),
        'dc_response': final_state.get('dc_response', '')
    }
    
    print(f"\n{'='*80}")
    print(f"✅ Query completed!")
    print(f"   Routed to: {result['routing']['agent'].upper()}")
    print(f"   Check trace at: https://smith.langchain.com/")
    print(f"{'='*80}\n")
    
    return result

print("✅ Query function ready")

"""
Cell 10: Chat Interface (Same as before, but with LangSmith!)
"""

import nest_asyncio
nest_asyncio.apply()

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import folium
from folium import WmsTileLayer
import matplotlib.pyplot as plt
import io
import base64
import markdown
import html as html_module

class LangGraphChatInterface:
    """Chat interface for LangGraph multi-agent system."""
    
    def __init__(self):
        self.messages_container = []
        self.conversation_history = []
        
        # Output area
        self.output_area = widgets.VBox(
            layout=widgets.Layout(
                border='1px solid #ddd',
                height='calc(100vh - 350px)',
                min_height='400px',
                overflow_y='auto',
                padding='10px',
                margin='0 0 10px 0'
            )
        )
        
        # Input controls
        self.input_box = widgets.Textarea(
            placeholder='Ask about California landscape metrics or general data...',
            layout=widgets.Layout(width='100%', height='100px', margin='10px 0')
        )
        
        self.send_button = widgets.Button(
            description='Send',
            button_style='primary',
            icon='paper-plane',
            layout=widgets.Layout(width='100px', margin='0 5px 0 0')
        )
        
        self.clear_button = widgets.Button(
            description='Clear',
            button_style='warning',
            icon='trash',
            layout=widgets.Layout(width='100px', margin='0 5px 0 0')
        )
        
        self.trace_button = widgets.Button(
            description='View Trace',
            button_style='info',
            icon='chart-line',
            layout=widgets.Layout(width='120px', margin='0 5px 0 0')
        )
        
        self.status_label = widgets.HTML(
            value="✅ Ready",
            layout=widgets.Layout(margin='0 0 0 10px')
        )
        
        # Event handlers
        self.send_button.on_click(self.on_send_clicked)
        self.clear_button.on_click(self.on_clear_clicked)
        self.trace_button.on_click(self.on_trace_clicked)
        
        # Layout
        button_box = widgets.HBox([
            self.send_button,
            self.clear_button,
            self.trace_button,
            self.status_label
        ])
        
        self.interface = widgets.VBox([
            widgets.HTML(value="""
                <h3>🎯 LangGraph Multi-Agent System</h3>
                <p style='color: #666; font-size: 0.9em;'>
                    <strong>✨ Powered by LangSmith Tracing!</strong><br>
                    <strong>CLM Agent:</strong> California environmental data<br>
                    <strong>Data Commons Agent:</strong> Global demographics<br>
                    <strong>🔍 View traces:</strong> <a href="https://smith.langchain.com/" target="_blank">smith.langchain.com</a>
                </p>
            """),
            self.output_area,
            self.input_box,
            button_box
        ], layout=widgets.Layout(width='100%', max_width='1200px', margin='0 auto'))
        
        # Welcome message
        self._add_message(
            "Welcome to LangGraph Multi-Agent System! 🎉\n\n"
            "**New Features:**\n"
            "- 🔍 **LangSmith Tracing**: All interactions automatically logged\n"
            "- 📊 **Visual Workflow**: See agent coordination in real-time\n"
            "- 💰 **Cost Tracking**: Monitor token usage and costs\n"
            "- 🐛 **Debug Mode**: Step through agent reasoning\n\n"
            "**Try asking:**\n"
            "- What is the carbon turnover time in Los Angeles?\n"
            "- Show distribution of unemployment in San Diego\n"
            "- What is the population of Sacramento?\n\n"
            "Click **View Trace** after each query to see the execution flow!",
            "system"
        )
    
    def _create_distribution_chart(self, distribution_data):
        """Create distribution chart."""
        # Same implementation as before
        try:
            data = distribution_data.get('data', [])
            dist_type = distribution_data.get('distribution_type', 'continuous')
            dataset_info = distribution_data.get('dataset_info', {})
            title = dataset_info.get('title', 'Value Distribution')
            units = dataset_info.get('units', '')
            filter_column = 'name'
            
            if not data:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            
            counties = sorted(list(set([d.get(filter_column) for d in data if filter_column in d])))
            if not counties:
                return None
            
            colors = plt.cm.tab10(range(len(counties)))
            
            if dist_type == 'categorical':
                import numpy as np
                values = sorted(list(set([d['value'] for d in data])))
                x = np.arange(len(values))
                width = 0.8 / len(counties) if len(counties) > 1 else 0.5
                
                for i, county in enumerate(counties):
                    county_data = [d for d in data if d.get(filter_column) == county]
                    counts = []
                    for val in values:
                        matching = [d['count'] for d in county_data if d['value'] == val]
                        counts.append(matching[0] if matching else 0)
                    
                    offset = (i - len(counties)/2) * width + width/2
                    ax.bar(x + offset, counts, width, label=county, alpha=0.7, color=colors[i])
                
                ax.set_xlabel('Value', fontsize=11)
                ax.set_ylabel('Count (pixels)', fontsize=11)
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in values])
                ax.legend(fontsize=10, loc='best')
                ax.set_title(f'{title}\nCategorical Distribution', fontsize=12, fontweight='bold', pad=10)
                
            else:  # continuous
                bins = distribution_data.get('bins', [])
                if not bins:
                    return None
                
                bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
                bin_width = bins[1] - bins[0] if len(bins) > 1 else 1
                bar_width = bin_width * 0.8 / len(counties) if len(counties) > 1 else bin_width * 0.7
                
                for i, county in enumerate(counties):
                    county_data = [d for d in data if d.get(filter_column) == county]
                    county_data = sorted(county_data, key=lambda x: x.get('bin_index', 0))
                    counts = [d['count'] for d in county_data]
                    
                    if len(counties) > 1:
                        offset = (i - len(counties)/2) * bar_width + bar_width/2
                        positions = [bc + offset for bc in bin_centers]
                    else:
                        positions = bin_centers
                    
                    ax.bar(positions, counts, bar_width, label=county, alpha=0.7, color=colors[i])
                
                xlabel = f'Value Range ({units})' if units else 'Value Range'
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel('Count (pixels)', fontsize=11)
                
                if len(counties) > 1:
                    ax.legend(fontsize=10, loc='best')
                
                ax.set_title(f'{title}\nValue Distribution', fontsize=12, fontweight='bold', pad=10)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None
    
    def _create_map(self, map_data, style_name=None):
        """Create Folium map."""
        try:
            wms_url = map_data.get('wms_base_url', '')
            layer_name = map_data.get('wms_layer_name', '')
            title = map_data.get('title', 'Dataset')
            
            # California bounds
            bounds = [[32.5, -124.5], [42.0, -114.0]]
            center_lat = (bounds[0][0] + bounds[1][0]) / 2
            center_lon = (bounds[0][1] + bounds[1][1]) / 2
            
            m = folium.Map(
                location=[center_lat, center_lon],
                tiles='OpenStreetMap',
                control_scale=True
            )
            m.fit_bounds(bounds)
            
            if wms_url and layer_name:
                wms_params = {
                    'url': wms_url + '/wms',
                    'layers': layer_name,
                    'name': title,
                    'fmt': 'image/png',
                    'transparent': True,
                    'overlay': True,
                    'control': True,
                    'version': '1.1.0'
                }
                
                if style_name:
                    wms_params['styles'] = style_name
                
                wms = WmsTileLayer(**wms_params)
                wms.add_to(m)
                folium.LayerControl().add_to(m)
            
            return m
        except Exception as e:
            print(f"Error creating map: {e}")
            return None
    
    def _add_message(self, text, role="user", map_data=None, distribution_data=None, routing_info=None):
        """Add message to chat."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if role == "user":
            color = "#007bff"
            icon = "👤"
            label = "You"
            bg_color = "#e7f3ff"
        elif role == "assistant":
            color = "#28a745"
            icon = "🤖"
            label = "Assistant"
            bg_color = "#e8f5e9"
        else:
            color = "#6c757d"
            icon = "ℹ️"
            label = "System"
            bg_color = "#f8f9fa"
        
        # Add routing badge
        routing_badge = ""
        if routing_info:
            agent = routing_info.get('agent', 'unknown').upper()
            confidence = routing_info.get('confidence', 0)
            agent_colors = {
                'CLM': '#ff6b6b',
                'DC': '#4ecdc4',
                'BOTH': '#95e1d3'
            }
            badge_color = agent_colors.get(agent, '#999')
            routing_badge = f"""
                <span style='background-color: {badge_color}; color: white; padding: 2px 8px; 
                             border-radius: 12px; font-size: 0.75em; font-weight: bold; margin-left: 8px;'>
                    {agent} ({confidence:.0%})
                </span>
            """
        
        # Convert markdown
        if role == "assistant":
            try:
                html_content = markdown.markdown(str(text), extensions=['extra', 'nl2br', 'sane_lists'])
            except:
                html_content = html_module.escape(str(text)).replace('\n', '<br>')
        else:
            html_content = html_module.escape(str(text)).replace('\n', '<br>')
        
        message_html = widgets.HTML(
            value=f"""
            <div style='margin: 10px 0; padding: 12px; background-color: {bg_color}; 
                        border-radius: 8px; border-left: 4px solid {color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <div>
                        <strong style='color: {color};'>{icon} {label}</strong>
                        {routing_badge}
                    </div>
                    <span style='color: #999; font-size: 0.85em;'>{timestamp}</span>
                </div>
                <div style='line-height: 1.6;'>{html_content}</div>
            </div>
            """
        )
        
        self.messages_container.append(message_html)
        
        # Add visualizations
        if distribution_data:
            img_base64 = self._create_distribution_chart(distribution_data)
            if img_base64:
                chart_html = f"""
                <div style='width: 98%; margin: 10px 0;'>
                    <div style='width: 100%; border: 1px solid #ddd; border-radius: 8px; 
                                padding: 10px; background-color: white;'>
                        <img src="data:image/png;base64,{img_base64}" 
                             style="width: 100%; height: auto;" alt="Distribution Chart">
                    </div>
                </div>
                """
                self.messages_container.append(widgets.HTML(value=chart_html))
        
        elif map_data:
            layer_name = map_data.get('wms_layer_name', '')
            style_name = f"{layer_name}_std" if layer_name else None
            folium_map = self._create_map(map_data, style_name=style_name)
            
            if folium_map:
                map_html = f"""
                <div style='width: 98%; margin: 10px 0;'>
                    <div style='width: 100%; height: 300px; border: 1px solid #ddd; 
                                border-radius: 8px; overflow: hidden;'>
                        {folium_map._repr_html_()}
                    </div>
                </div>
                """
                self.messages_container.append(widgets.HTML(value=map_html))
        
        self.output_area.children = tuple(self.messages_container)
    
    def on_send_clicked(self, button):
        """Handle send button click."""
        question = self.input_box.value.strip()
        if not question:
            return
        
        self._add_message(question, "user")
        self.input_box.value = ""
        self.send_button.disabled = True
        self.input_box.disabled = True
        self.status_label.value = "<span style='color: orange;'>⏳ Processing...</span>"
        
        try:
            import asyncio
            
            # Run the query
            result = asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(run_multi_agent_query(question), timeout=180)
            )
            
            # Store in history
            self.conversation_history.append({
                'question': question,
                'result': result,
                'timestamp': datetime.now()
            })
            
            answer = result.get('output', 'No response')
            routing_info = result.get('routing')
            map_data = result.get('map_data')
            distribution_data = result.get('distribution_data')
            
            self._add_message(
                answer,
                "assistant",
                map_data=map_data,
                distribution_data=distribution_data,
                routing_info=routing_info
            )
            
            self.status_label.value = "<span style='color: green;'>✅ Ready</span>"
            
        except asyncio.TimeoutError:
            self._add_message("Request timed out after 3 minutes.", "system")
            self.status_label.value = "<span style='color: red;'>❌ Timeout</span>"
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            self._add_message(error_msg, "system")
            self.status_label.value = "<span style='color: red;'>❌ Error</span>"
        finally:
            self.send_button.disabled = False
            self.input_box.disabled = False
    
    def on_clear_clicked(self, button):
        """Clear chat history."""
        self.messages_container = []
        self.output_area.children = tuple(self.messages_container)
        self._add_message(
            "Chat cleared. Start asking questions!",
            "system"
        )
    
    def on_trace_clicked(self, button):
        """Show LangSmith trace link."""
        if not self.conversation_history:
            self._add_message(
                "No queries yet! Ask a question first, then click 'View Trace' to see the execution details.",
                "system"
            )
            return
        
        last_query = self.conversation_history[-1]
        trace_msg = f"""**🔍 LangSmith Trace Available**

View detailed trace at: [LangSmith Dashboard](https://smith.langchain.com/)

**What you'll see:**
- 📊 Visual workflow graph
- ⏱️ Timing for each step
- 💰 Token usage and costs
- 🔧 All tool calls and responses
- 🐛 Debug information

**Last Query:**
- Question: {last_query['question']}
- Routed to: {last_query['result']['routing']['agent'].upper()}
- Confidence: {last_query['result']['routing']['confidence']:.0%}

All traces are automatically saved and searchable in LangSmith!"""
        
        self._add_message(trace_msg, "system")
    
    def display(self):
        """Display the interface."""
        clear_output(wait=True)
        display(HTML("""
        <style>
            .jp-Cell-outputArea { max-height: none !important; }
            .output_scroll { max-height: none !important; overflow-y: visible !important; }
        </style>
        """))
        display(self.interface)

# Create and display chat interface
chat = LangGraphChatInterface()
chat.display()

print("✅ Chat interface ready!")
print("🎯 All queries will be traced in LangSmith!")

"""
Cell 11: Example Queries and Testing
"""

# Example queries to test the system
example_queries = [
    "What is the carbon turnover time in Los Angeles?",
    "Show me the unemployment distribution in San Diego",
    "What is the population of Sacramento?",
    "Compare burn probability between San Diego and Los Angeles",
    "What is the median household income in San Francisco?",
]

async def test_queries():
    """Test the system with example queries."""
    print("🧪 Testing Multi-Agent System\n")
    
    for i, question in enumerate(example_queries[:2], 1):  # Test first 2
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(example_queries[:2])}")
        print(f"{'='*80}")
        
        result = await run_multi_agent_query(question)
        
        print(f"\n✅ Result:")
        print(f"   Output: {result['output'][:200]}...")
        print(f"   Agent: {result['routing']['agent'].upper()}")
        print(f"   Confidence: {result['routing']['confidence']:.0%}")
        
        if result.get('map_data'):
            print(f"   📍 Map data available")
        if result.get('distribution_data'):
            print(f"   📊 Distribution data available")
        
        print(f"\n🔍 View trace: https://smith.langchain.com/")
        
        # Small delay between queries
        await asyncio.sleep(2)
    
    print(f"\n{'='*80}")
    print("✅ Testing complete!")
    print("🔍 Check LangSmith for detailed traces of all queries")
    print(f"{'='*80}\n")

# Uncomment to run tests:
# await test_queries()

print("✅ Example queries ready")
print("\nYou can test with:")
print("  await test_queries()")

"""
Cell 12: Visualization Tools
"""

from langsmith import Client as LangSmithClient

def get_trace_stats():
    """Get statistics from LangSmith traces."""
    try:
        client = LangSmithClient()
        project_name = os.environ.get("LANGCHAIN_PROJECT", "CLM-DataCommons-MultiAgent")
        
        # Get recent runs
        runs = list(client.list_runs(project_name=project_name, limit=10))
        
        if not runs:
            print("No traces found yet. Run some queries first!")
            return
        
        print(f"\n📊 Trace Statistics for Project: {project_name}")
        print(f"{'='*80}")
        
        total_tokens = 0
        total_cost = 0
        agents_used = {'clm': 0, 'dc': 0, 'both': 0}
        
        for run in runs:
            if hasattr(run, 'total_tokens'):
                total_tokens += run.total_tokens or 0
            
            # Estimate cost (rough calculation)
            if hasattr(run, 'total_tokens') and run.total_tokens:
                # GPT-4o-mini: $0.150 / 1M input, $0.600 / 1M output
                estimated_cost = (run.total_tokens / 1_000_000) * 0.375  # Average
                total_cost += estimated_cost
        
        print(f"Recent Queries: {len(runs)}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Estimated Cost: ${total_cost:.4f}")
        
        print(f"\n🔗 View all traces:")
        print(f"   https://smith.langchain.com/projects/{project_name}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error getting trace stats: {e}")
        print("Make sure LANGCHAIN_API_KEY is set correctly.")

# Get trace statistics
# get_trace_stats()

print("✅ Visualization tools ready")
print("\nYou can view stats with:")
print("  get_trace_stats()")

"""
Cell 13: Summary and Next Steps
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    🎉 LangGraph Setup Complete! 🎉                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

✨ What You Got:

1. **LangSmith Automatic Tracing**
   - Every LLM call logged
   - All MCP tool calls tracked
   - Visual workflow graphs
   - Token usage & cost tracking
   
2. **Multi-Agent Workflow**
   - Router decides which agent(s) to use
   - CLM Agent for California data
   - Data Commons Agent for global data
   - Combiner merges responses

3. **Built-in Tools**
   - search_clm_datasets
   - get_clm_statistics
   - get_clm_distribution
   - search_dc_indicators
   - get_dc_observations

4. **Interactive Chat Interface**
   - Send queries
   - View results with visualizations
   - Access trace links

═══════════════════════════════════════════════════════════════════════════

🚀 Try It Out:

1. Use the chat interface above
2. Ask: "What is the carbon turnover time in Los Angeles?"
3. Click "View Trace" to see the execution flow
4. Visit https://smith.langchain.com/ for detailed analysis

═══════════════════════════════════════════════════════════════════════════

📊 Key Differences from Pydantic AI:

| Feature | Pydantic AI | LangGraph |
|---------|-------------|-----------|
| Tracing | Manual | Automatic ✅ |
| Visualization | None | Graph view ✅ |
| Cost tracking | Manual | Built-in ✅ |
| Workflow graph | Hidden | Visual ✅ |
| MCP tracking | Manual | Automatic ✅ |
| Team sharing | No | Yes ✅ |

═══════════════════════════════════════════════════════════════════════════

🎯 LangSmith Benefits:

✓ See EXACTLY how agents coordinate
✓ Debug failures step-by-step
✓ Optimize token usage and costs
✓ Share traces with team
✓ Export data for research papers
✓ Monitor production deployments

═══════════════════════════════════════════════════════════════════════════

📚 Resources:

- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- LangSmith Docs: https://docs.smith.langchain.com/
- Your traces: https://smith.langchain.com/

═══════════════════════════════════════════════════════════════════════════

Happy querying! 🚀
""")