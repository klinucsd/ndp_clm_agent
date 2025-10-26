from fastmcp import FastMCP
from typing import Optional, List, Union
import requests
import json

# Initialize FastMCP server
mcp = FastMCP("Geospatial Analysis with Dataset Discovery for California Landscape Metrics")


@mcp.tool()
def search_datasets(
    query: str,
    top_k: int = 3,
    rag_endpoint: str = "https://sparcal.sdsc.edu/api/v1/Utility/clm"
) -> dict:
    """
    Search for relevant WCS datasets using RAG-based vector search.
    
    This tool uses semantic search to find the most relevant California Landscape
    Metrics datasets for a given query. It returns dataset metadata including
    WCS coverage IDs, WMS layer names, descriptions, and data units.
    
    Args:
        query: Natural language query describing what data you need
               (e.g., "carbon turnover time", "wildfire hazard", "burn probability")
        top_k: Number of top results to return (default: 3)
        rag_endpoint: RAG search endpoint URL (default: sparcal.sdsc.edu endpoint)
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating if search succeeded
        - datasets: List of relevant datasets with metadata
        - count: Number of datasets returned
        - query: Original query string
    
    Example:
        result = search_datasets(
            query="What is the average carbon turnover time?",
            top_k=3
        )
        
        # Use the first result's coverage_id for analysis
        if result['success']:
            best_dataset = result['datasets'][0]
            coverage_id = best_dataset['wcs_coverage_id']
            print(f"Using: {best_dataset['title']}")
    """
    try:
        response = requests.get(
            rag_endpoint,
            params={"search_terms": query},
            timeout=10
        )
        response.raise_for_status()
        results = response.json()
        
        if not isinstance(results, list):
            return {
                'success': False,
                'message': 'Unexpected response format from RAG endpoint',
                'query': query
            }
        
        # Extract relevant information from top k results
        datasets = []
        for pkg in results[:top_k]:
            # Find WCS resource
            wcs_resource = None
            for resource in pkg.get('resources', []):
                if resource.get('format') == 'WCS':
                    wcs_resource = resource
                    break

            # Find WMS resource
            wms_resource = None
            for resource in pkg.get('resources', []):
                if resource.get('format') == 'WMS':
                    wms_resource = resource
                    break
                    
            if not wcs_resource:
                continue
            
            # Extract extras
            extras = {e['key']: e['value'] for e in pkg.get('extras', [])}
            
            dataset_info = {
                'wcs_coverage_id': wcs_resource.get('wcs_coverage_id'),
                'wcs_base_url': wcs_resource.get('url'),
                'wcs_projection': wcs_resource.get('wcs_srs'),
                'wms_layer_name': wms_resource.get('wms_layer'),
                'wms_base_url': wms_resource.get('url'),
                'wms_projection': wcs_resource.get('wms_srs'),
                'title': pkg.get('title', ''),
                'description': pkg.get('notes', '')[:300],  # Truncate
                'data_units': extras.get('data_units', 'unknown'),
                'pillar': extras.get('pillar', 'unknown'),
                'element': extras.get('element', 'unknown'),
                'tags': [tag['name'] for tag in pkg.get('tags', [])]
            }
            
            datasets.append(dataset_info)
        
        return {
            'success': True,
            'datasets': datasets,
            'count': len(datasets),
            'query': query,
            'message': f'Found {len(datasets)} relevant datasets'
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to search datasets: {str(e)}',
            'query': query
        }


@mcp.tool()
def compute_zonal_stats(
    wcs_base_url: str,
    wfs_base_url: str,
    wcs_coverage_id: str,
    feature_id: str,
    filter_column: str,
    filter_value: Optional[Union[str, List[str]]] = None,
    stats: List[str] = ["mean", "min", "max", "std"],
    max_retries: int = 3,
    timeout: int = 30,
    max_workers: int = 16,
    api_endpoint: str = "https://sparcal.sdsc.edu/api/v1/Utility/compute_zonal_stats"
) -> dict:
    """
    Compute zonal statistics for features from WFS using raster data from WCS.
    
    This tool calls a FastAPI endpoint that processes geographic features and computes
    statistics like mean, min, max, and standard deviation for raster data within each feature.
    
    Args:
        wcs_base_url: Base URL for the Web Coverage Service (e.g., "https://sparcal.sdsc.edu/geoserver")
        wfs_base_url: Base URL for the Web Feature Service (e.g., "https://sparcal.sdsc.edu/geoserver/boundary/wfs")
        wcs_coverage_id: Coverage identifier for the raster layer (e.g., "rrk__cstocks_turnovertime_202009_202312_t1_v5")
        feature_id: Feature type identifier (e.g., "boundary:ca_counties")
        filter_column: Column name to filter features (e.g., "name")
        filter_value: Single value or list of values to filter features. If None, processes all features
        stats: List of statistics to compute. Options: "mean", "min", "max", "std", "sum", "count", "median"
        max_retries: Maximum number of retry attempts for failed requests (1-10)
        timeout: Request timeout in seconds (10-120)
        max_workers: Number of parallel workers for processing (1-24)
        api_endpoint: FastAPI endpoint URL
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating if operation succeeded
        - data: List of dictionaries with statistics for each feature
        - failed_features: List of features that failed processing (if any)
        - total_features: Total number of features attempted
        - processed_features: Number of successfully processed features
        - processing_time_seconds: Time taken to process
        - message: Status message
    
    Example:
        result = compute_zonal_stats(
            wcs_base_url="https://sparcal.sdsc.edu/geoserver",
            wfs_base_url="https://sparcal.sdsc.edu/geoserver/boundary/wfs",
            wcs_coverage_id="rrk__cstocks_turnovertime_202009_202312_t1_v5",
            feature_id="boundary:ca_counties",
            filter_column="name",
            filter_value=["Los Angeles", "Orange"],
            stats=["mean", "min", "max"]
        )
    """
    payload = {
        "wcs_base_url": wcs_base_url,
        "wfs_base_url": wfs_base_url,
        "wcs_coverage_id": wcs_coverage_id,
        "feature_id": feature_id,
        "filter_column": filter_column,
        "filter_value": filter_value,
        "stats": stats,
        "max_retries": max_retries,
        "timeout": timeout,
        "max_workers": max_workers
    }
    
    try:
        response = requests.post(api_endpoint, json=payload, timeout=max(timeout, 60))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to call API endpoint: {str(e)}"
        }


# @mcp.tool()
# def zonal_count(
#     wcs_base_url: str,
#     wfs_base_url: str,
#     wcs_coverage_id: str,
#     feature_id: str,
#     filter_column: str,
#     filter_value: str,
#     threshold: float,
#     max_retries: int = 3,
#     timeout: int = 30,
#     api_endpoint: str = "https://sparcal.sdsc.edu/api/v1/Utility/zonal_count"
# ) -> dict:
#     """
#     Compute pixel counts above a threshold for a feature from WFS using raster data from WCS.
    
#     This tool calls a FastAPI endpoint that counts how many pixels in a raster layer
#     have values exceeding a specified threshold within a geographic feature boundary.
    
#     Args:
#         wcs_base_url: Base URL for the Web Coverage Service (e.g., "https://sparcal.sdsc.edu/geoserver")
#         wfs_base_url: Base URL for the Web Feature Service (e.g., "https://sparcal.sdsc.edu/geoserver/boundary/wfs")
#         wcs_coverage_id: Coverage identifier for the raster layer (e.g., "rrk__cstocks_turnovertime_202009_202312_t1_v5")
#         feature_id: Feature type identifier (e.g., "boundary:ca_counties")
#         filter_column: Column name to filter features (e.g., "name")
#         filter_value: Single value to identify the feature (e.g., "San Diego")
#         threshold: Threshold value for counting pixels
#         max_retries: Maximum number of retry attempts for failed requests (1-10)
#         timeout: Request timeout in seconds (10-120)
#         api_endpoint: FastAPI endpoint URL
    
#     Returns:
#         Dictionary containing:
#         - success: Boolean indicating if operation succeeded
#         - data: Dictionary with:
#             - filter_column: Name of the feature
#             - valid_pixels: Total count of valid (non-nodata) pixels
#             - above_threshold_pixels: Count of pixels above the threshold
#             - pixel_area_square_meters: Area of each pixel in square meters
#         - processing_time_seconds: Time taken to process
#         - message: Status message
    
#     Example:
#         result = zonal_count(
#             wcs_base_url="https://sparcal.sdsc.edu/geoserver",
#             wfs_base_url="https://sparcal.sdsc.edu/geoserver/boundary/wfs",
#             wcs_coverage_id="rrk__cstocks_turnovertime_202009_202312_t1_v5",
#             feature_id="boundary:ca_counties",
#             filter_column="name",
#             filter_value="San Diego",
#             threshold=100.0
#         )
        
#         # Calculate percentage and area
#         if result['success']:
#             data = result['data']
#             percentage = (data['above_threshold_pixels'] / data['valid_pixels'] * 100)
#             area_sq_m = data['above_threshold_pixels'] * data['pixel_area_square_meters']
#             print(f"Percentage: {percentage:.2f}%")
#             print(f"Area: {area_sq_m:.2f} square meters")
#     """
#     payload = {
#         "wcs_base_url": wcs_base_url,
#         "wfs_base_url": wfs_base_url,
#         "wcs_coverage_id": wcs_coverage_id,
#         "feature_id": feature_id,
#         "filter_column": filter_column,
#         "filter_value": filter_value,
#         "threshold": threshold,
#         "max_retries": max_retries,
#         "timeout": timeout
#     }
    
#     try:
#         response = requests.post(api_endpoint, json=payload, timeout=max(timeout, 60))
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "message": f"Failed to call API endpoint: {str(e)}"
#         }

@mcp.tool()
def zonal_count(
    wcs_base_url: str,
    wfs_base_url: str,
    wcs_coverage_id: str,
    feature_id: str,
    filter_column: str,
    filter_value: Optional[Union[str, List[str]]] = None,
    threshold: float = 100.0,
    max_retries: int = 3,
    timeout: int = 30,
    max_workers: int = 16,
    api_endpoint: str = "https://sparcal.sdsc.edu/api/v1/Utility/zonal_count_batch"
) -> dict:
    """
    Compute pixel counts above a threshold for features from WFS using raster data from WCS.
    
    This tool calls a FastAPI endpoint that counts how many pixels in a raster layer
    have values exceeding a specified threshold within geographic feature boundaries.
    Supports processing single or multiple features in parallel.
    
    Args:
        wcs_base_url: Base URL for the Web Coverage Service (e.g., "https://sparcal.sdsc.edu/geoserver")
        wfs_base_url: Base URL for the Web Feature Service (e.g., "https://sparcal.sdsc.edu/geoserver/boundary/wfs")
        wcs_coverage_id: Coverage identifier for the raster layer (e.g., "rrk__cstocks_turnovertime_202009_202312_t1_v5")
        feature_id: Feature type identifier (e.g., "boundary:ca_counties")
        filter_column: Column name to filter features (e.g., "name")
        filter_value: Single value, list of values, or None to process all features (e.g., "San Diego" or ["San Diego", "Los Angeles"])
        threshold: Threshold value for counting pixels (default: 100.0)
        max_retries: Maximum number of retry attempts for failed requests (1-10)
        timeout: Request timeout in seconds (10-120)
        max_workers: Number of parallel workers for processing (1-24)
        api_endpoint: FastAPI endpoint URL
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating if operation succeeded
        - data: List of dictionaries, each with:
            - filter_column: Name of the feature
            - valid_pixels: Total count of valid (non-nodata) pixels
            - above_threshold_pixels: Count of pixels above the threshold
            - pixel_area_square_meters: Area of each pixel in square meters
            - threshold: The threshold value used
        - failed_features: List of features that failed processing (if any)
        - total_features: Total number of features attempted
        - processed_features: Number of successfully processed features
        - processing_time_seconds: Time taken to process
        - message: Status message
    
    Examples:
        # Process a single county
        result = zonal_count(
            wcs_base_url="https://sparcal.sdsc.edu/geoserver",
            wfs_base_url="https://sparcal.sdsc.edu/geoserver/boundary/wfs",
            wcs_coverage_id="rrk__cstocks_turnovertime_202009_202312_t1_v5",
            feature_id="boundary:ca_counties",
            filter_column="name",
            filter_value="San Diego",
            threshold=100.0
        )
        
        # Process multiple counties
        result = zonal_count(
            wcs_base_url="https://sparcal.sdsc.edu/geoserver",
            wfs_base_url="https://sparcal.sdsc.edu/geoserver/boundary/wfs",
            wcs_coverage_id="rrk__cstocks_turnovertime_202009_202312_t1_v5",
            feature_id="boundary:ca_counties",
            filter_column="name",
            filter_value=["San Diego", "Los Angeles", "Orange"],
            threshold=100.0,
            max_workers=8
        )
        
        # Process all counties
        result = zonal_count(
            wcs_base_url="https://sparcal.sdsc.edu/geoserver",
            wfs_base_url="https://sparcal.sdsc.edu/geoserver/boundary/wfs",
            wcs_coverage_id="rrk__cstocks_turnovertime_202009_202312_t1_v5",
            feature_id="boundary:ca_counties",
            filter_column="name",
            filter_value=None,  # Process all features
            threshold=100.0,
            max_workers=16
        )
        
        # Calculate percentages and areas from results
        if result['success']:
            for data in result['data']:
                percentage = (data['above_threshold_pixels'] / data['valid_pixels'] * 100) if data['valid_pixels'] > 0 else 0
                area_sq_m = data['above_threshold_pixels'] * data['pixel_area_square_meters']
                print(f"{data[filter_column]}: {percentage:.2f}% ({area_sq_m:.2f} sq m)")
    """
    payload = {
        "wcs_base_url": wcs_base_url,
        "wfs_base_url": wfs_base_url,
        "wcs_coverage_id": wcs_coverage_id,
        "feature_id": feature_id,
        "filter_column": filter_column,
        "filter_value": filter_value,
        "threshold": threshold,
        "max_retries": max_retries,
        "timeout": timeout,
        "max_workers": max_workers
    }
    
    try:
        response = requests.post(api_endpoint, json=payload, timeout=max(timeout, 60))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to call API endpoint: {str(e)}"
        }
