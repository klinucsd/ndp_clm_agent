"""
Extended Pydantic AI Agent for California Landscape Metrics Analysis
Uses MCP tools to dynamically discover and query datasets based on questions.
"""

import asyncio
from pydantic_ai import Agent, RunContext
from fastmcp import Client
from typing import Optional
from dataclasses import dataclass
import os

# Base configuration - feature service remains the same
BASE_CONFIG = {
    "wcs_base_url": "https://sparcal.sdsc.edu/geoserver",
    "wfs_base_url": "https://sparcal.sdsc.edu/geoserver/boundary/wfs",
    "feature_id": "boundary:ca_counties",
    "filter_column": "name"
}

# Initialize MCP client
mcp_client = Client("http://127.0.0.1:8800/mcp")
# mcp_client = Client("https://wenokn.fastmcp.app/mcp")

@dataclass
class AgentContext:
    """Context to store discovered dataset information."""
    current_coverage_id: Optional[str] = None
    current_dataset_info: Optional[dict] = None


def create_agent():
    """Create and configure the Pydantic AI agent with dynamic dataset discovery."""
    
    agent = Agent(
        model='openai:gpt-4o',
        deps_type=AgentContext,
        system_prompt="""You are an expert in analyzing California Landscape Metrics datasets.

You have access to these tools:
1. search_and_select_dataset: Search for the most relevant dataset based on the question
2. get_county_statistics: Compute statistics for one or more counties
3. get_area_above_threshold: Calculate percentage/area above a threshold for a single county
4. get_area_below_threshold: Calculate percentage/area below a threshold for a single county

**WORKFLOW:**
1. ALWAYS start by calling search_and_select_dataset with the user's question to find the right dataset
2. Once the dataset is selected, use the other tools to answer the question
3. Include the dataset name and units in your final answer for context

**For statistical questions:**
- Use get_county_statistics for mean, median, min, max, std
- Pass counties=None to get all counties
- For rankings, get all counties and sort results

**For threshold questions:**
- Use get_area_above_threshold or get_area_below_threshold
- These work on single counties only

**IMPORTANT - For questions about thresholds across many counties:**
- First get all counties with get_county_statistics(counties=None)
- Select a SMALL SAMPLE (5-10 counties) to check thresholds
- Report sampled results, noting it's a sample
- DO NOT check all 58 counties individually

**Answer format:**
- Be precise with numbers and include units from the dataset
- Provide clear, concise answers
- Mention the dataset being used

Fixed parameters (auto-configured):
- wcs_base_url, wfs_base_url, feature_id, filter_column
""",
    )

    @agent.tool
    async def search_and_select_dataset(
        ctx: RunContext[AgentContext],
        question: str,
        top_k: int = 3
    ) -> dict:
        """
        Search for and select the most relevant dataset for answering the question.
        This should be called FIRST before any analysis.
        
        Args:
            question: The user's question
            top_k: Number of top results to consider (default: 3)
        
        Returns:
            Information about the selected dataset including coverage_id and metadata
        """
        async with mcp_client:
            result = await mcp_client.call_tool(
                "search_datasets",
                {
                    "query": question,
                    "top_k": top_k
                }
            )
            
            data = result.data
            if data.get('success') and data.get('datasets'):
                # Select the first (most relevant) dataset
                best_dataset = data['datasets'][0]
                
                # Update context with selected dataset
                ctx.deps.current_coverage_id = best_dataset['wcs_coverage_id']
                ctx.deps.current_dataset_info = best_dataset
                
                return {
                    'success': True,
                    'selected_dataset': best_dataset,
                    'alternatives': data['datasets'][1:] if len(data['datasets']) > 1 else [],
                    'message': f"Selected: {best_dataset['title']}"
                }
            else:
                return {
                    'success': False,
                    'message': 'No suitable datasets found',
                    'error': data.get('error', 'Unknown error')
                }

    @agent.tool
    async def get_county_statistics(
        ctx: RunContext[AgentContext],
        counties: Optional[list[str]] = None,
        stats: list[str] = None
    ) -> dict:
        """
        Get statistics for one or more counties using the currently selected dataset.
        Must call search_and_select_dataset first.
        
        Args:
            counties: List of county names or None for all counties
            stats: Statistics to compute (default: mean, median, min, max, std)
        
        Returns:
            Dictionary with statistics for each county
        """
        if not ctx.deps.current_coverage_id:
            return {
                'success': False,
                'error': 'No dataset selected. Call search_and_select_dataset first.'
            }
        
        if stats is None:
            stats = ["mean", "median", "min", "max", "std"]
        
        async with mcp_client:
            result = await mcp_client.call_tool(
                "compute_zonal_stats",
                {
                    **BASE_CONFIG,
                    "wcs_coverage_id": ctx.deps.current_coverage_id,
                    "filter_value": counties,
                    "stats": stats,
                    "max_workers": 8
                }
            )
            
            # Add dataset info to response
            response = result.data
            if response.get('success'):
                response['dataset_info'] = {
                    'title': ctx.deps.current_dataset_info.get('title'),
                    'units': ctx.deps.current_dataset_info.get('data_units')
                }
            return response

    @agent.tool
    async def get_area_above_threshold(
        ctx: RunContext[AgentContext],
        county: str,
        threshold: float
    ) -> dict:
        """
        Calculate percentage and area above a threshold for a single county.
        Must call search_and_select_dataset first.
        
        Args:
            county: County name (e.g., "San Diego")
            threshold: Threshold value
        
        Returns:
            Dictionary with pixel counts, percentage, and area
        """
        if not ctx.deps.current_coverage_id:
            return {
                'success': False,
                'error': 'No dataset selected. Call search_and_select_dataset first.'
            }
        
        async with mcp_client:
            result = await mcp_client.call_tool(
                "zonal_count",
                {
                    **BASE_CONFIG,
                    "wcs_coverage_id": ctx.deps.current_coverage_id,
                    "filter_value": county,
                    "threshold": threshold
                }
            )
            
            data = result.data
            if data.get('success'):
                stats = data['data']
                valid = stats['valid_pixels']
                above = stats['above_threshold_pixels']
                pixel_area = stats['pixel_area_square_meters']
                
                percentage = (above / valid * 100) if valid > 0 else 0
                area_sq_m = above * pixel_area
                area_sq_km = area_sq_m / 1_000_000
                
                return {
                    'success': True,
                    'county': county,
                    'threshold': threshold,
                    'valid_pixels': valid,
                    'above_threshold_pixels': above,
                    'percentage': percentage,
                    'area_square_meters': area_sq_m,
                    'area_square_km': area_sq_km,
                    'dataset_info': {
                        'title': ctx.deps.current_dataset_info.get('title'),
                        'units': ctx.deps.current_dataset_info.get('data_units')
                    }
                }
            return data

    @agent.tool
    async def get_area_below_threshold(
        ctx: RunContext[AgentContext],
        county: str,
        threshold: float
    ) -> dict:
        """
        Calculate percentage and area below a threshold for a single county.
        Must call search_and_select_dataset first.
        
        Args:
            county: County name (e.g., "San Diego")
            threshold: Threshold value
        
        Returns:
            Dictionary with pixel counts, percentage, and area
        """
        if not ctx.deps.current_coverage_id:
            return {
                'success': False,
                'error': 'No dataset selected. Call search_and_select_dataset first.'
            }
        
        async with mcp_client:
            result = await mcp_client.call_tool(
                "zonal_count",
                {
                    **BASE_CONFIG,
                    "wcs_coverage_id": ctx.deps.current_coverage_id,
                    "filter_value": county,
                    "threshold": threshold
                }
            )
            
            data = result.data
            if data.get('success'):
                stats = data['data']
                valid = stats['valid_pixels']
                above = stats['above_threshold_pixels']
                below = valid - above
                pixel_area = stats['pixel_area_square_meters']
                
                percentage = (below / valid * 100) if valid > 0 else 0
                area_sq_m = below * pixel_area
                area_sq_km = area_sq_m / 1_000_000
                
                return {
                    'success': True,
                    'county': county,
                    'threshold': threshold,
                    'valid_pixels': valid,
                    'below_threshold_pixels': below,
                    'percentage': percentage,
                    'area_square_meters': area_sq_m,
                    'area_square_km': area_sq_km,
                    'dataset_info': {
                        'title': ctx.deps.current_dataset_info.get('title'),
                        'units': ctx.deps.current_dataset_info.get('data_units')
                    }
                }
            return data
    
    return agent


async def answer_question(agent, question: str, timeout: int = 300) -> str:
    """Answer a question with dynamic dataset discovery."""
    try:
        # Create fresh context for each question
        ctx = AgentContext()
        
        # Run with timeout
        result = await asyncio.wait_for(
            agent.run(question, deps=ctx),
            timeout=timeout
        )
        return result.output if hasattr(result, 'output') else str(result)
    except asyncio.TimeoutError:
        return f"Error: Question timed out after {timeout} seconds."
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)[:200]}"


async def main():
    """Test the agent with diverse questions across different datasets."""
    
    agent = create_agent()
    
    questions = [
        # Carbon turnover time questions
        'What is the average carbon turnover time in Los Angeles County?',
        'Which county has the highest carbon turnover time?',
        
        # Wildfire/burn probability questions
        'What is the burn probability in San Diego County?',
        'Which counties have the highest wildfire hazard?',
        
        # Vegetation questions
        'What is the average vegetation height in Mendocino County?',
        
        # Carbon stock questions
        'What are the carbon stocks in Sonoma County?',
        
        # General landscape metrics
        'What is the average elevation in Alpine County?',
        'Show me tree canopy cover statistics for Sacramento County?',
    ]
    
    print("=" * 80)
    print("California Landscape Metrics Analysis - Dynamic Dataset Discovery")
    print("=" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 80}")
        print(f"Question {i}: {question}")
        print('-' * 80)
        
        try:
            answer = await answer_question(agent, question, timeout=180)
            print(f"Answer: {answer}")
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except Exception as e:
            print(f"Error: {type(e).__name__}: {str(e)[:200]}")
        
        print()
    
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
    else:
        asyncio.run(main())
