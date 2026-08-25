"""
API Retrieval Module for SimpleCoder

This module provides functionality to retrieve relevant API endpoints based on task descriptions.
It uses TF-IDF based similarity search to find the most relevant API endpoints from multiple
API databases (MediaWiki, YouTube, ORCID).

The module supports:
- Loading API databases from JSON files
- Semantic search using TF-IDF vectorization
- Generating LLM-friendly API context for prompts
- Automatic detection of authentication requirements
"""

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIEndpointInfo:
    """Lightweight representation of an API endpoint for retrieval purposes."""
    api_name: str           # Name of the parent API (e.g., "YouTube Data API")
    action: str             # Action identifier (e.g., "videos.list")
    name: str               # Human-readable name
    description: str        # Full description
    http_method: str        # GET, POST, etc.
    url_pattern: str        # URL pattern
    requires_auth: bool     # Whether authentication is required
    auth_type: Optional[str] = None  # Type of authentication (e.g., "api_key", "oauth")
    parameters: List[Dict] = None    # List of parameters
    examples: List[Dict] = None      # Example requests
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.examples is None:
            self.examples = []


class APIRetriever:
    """
    Retrieves relevant API endpoints based on task descriptions.
    
    Uses TF-IDF based similarity to find the most relevant endpoints from
    loaded API databases.
    """
    
    def __init__(self, api_db_paths: Optional[List[str]] = None):
        """
        Initialize the API retriever.
        
        Args:
            api_db_paths: List of paths to API database JSON files.
                         If None, will try to load from default locations.
        """
        self.endpoints: List[APIEndpointInfo] = []
        self.api_databases: Dict[str, Dict] = {}
        self._tfidf_matrix = None
        self._vectorizer = None
        self._documents = []
        
        # Load API databases
        if api_db_paths:
            for path in api_db_paths:
                self._load_api_database(path)
        else:
            self._load_default_databases()
        
        # Build search index
        if self.endpoints:
            self._build_search_index()
    
    def _load_default_databases(self):
        """Load API databases from default locations."""
        # Try to find the init_api_database directory
        possible_paths = [
            # Relative to this file (scripts/api_retrieval.py -> scripts/init_api_database/)
            Path(__file__).parent / "init_api_database",
            # From current working directory
            Path.cwd() / "scripts" / "init_api_database",
            Path.cwd() / "examples" / "open_deep_research" / "scripts" / "init_api_database",
        ]
        
        db_files = ["mediawiki_api.json", "youtube_api.json", "orcid_api.json"]
        
        for base_path in possible_paths:
            if base_path.exists():
                for db_file in db_files:
                    db_path = base_path / db_file
                    if db_path.exists():
                        self._load_api_database(str(db_path))
                break
    
    def _load_api_database(self, path: str):
        """
        Load an API database from a JSON file.
        
        Args:
            path: Path to the JSON file
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            api_name = data.get("name", "Unknown API")
            self.api_databases[api_name] = data
            
            # Determine auth type from API name
            auth_type = None
            if "youtube" in api_name.lower():
                auth_type = "api_key"  # YouTube uses API key
            elif "orcid" in api_name.lower():
                auth_type = "bearer_token"  # ORCID uses Bearer token
            
            # Extract endpoints
            for ep in data.get("endpoints", []):
                endpoint = APIEndpointInfo(
                    api_name=api_name,
                    action=ep.get("action", ""),
                    name=ep.get("name", ""),
                    description=ep.get("description", ""),
                    http_method=ep.get("http_method", "GET"),
                    url_pattern=ep.get("url_pattern", ""),
                    requires_auth=ep.get("requires_auth", False),
                    auth_type=auth_type if ep.get("requires_auth", False) else None,
                    parameters=ep.get("parameters", []),
                    examples=ep.get("examples", [])
                )
                self.endpoints.append(endpoint)
            
            logger.info(f"Loaded {len(data.get('endpoints', []))} endpoints from {api_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load API database from {path}: {e}")
    
    def _build_search_index(self):
        """Build TF-IDF search index from endpoints."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create documents for each endpoint
            self._documents = []
            for ep in self.endpoints:
                # Combine relevant text fields for search
                doc = f"{ep.name} {ep.action} {ep.description} {ep.api_name}"
                # Add parameter names and descriptions
                for param in ep.parameters:
                    doc += f" {param.get('name', '')} {param.get('description', '')}"
                self._documents.append(doc.lower())
            
            # Build TF-IDF vectorizer
            self._vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(self._documents)
            
        except ImportError:
            logger.warning("sklearn not available, falling back to simple keyword search")
            self._vectorizer = None
    
    def search(self, query: str, top_k: int = 5) -> List[APIEndpointInfo]:
        """
        Search for relevant API endpoints based on a query.
        
        Args:
            query: The search query (typically a task description)
            top_k: Maximum number of results to return
            
        Returns:
            List of relevant APIEndpointInfo objects
        """
        if not self.endpoints:
            return []
        
        if self._vectorizer is not None:
            return self._tfidf_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)
    
    def _tfidf_search(self, query: str, top_k: int) -> List[APIEndpointInfo]:
        """Search using TF-IDF similarity."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Vectorize query
        query_vec = self._vectorizer.transform([query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Filter out zero-similarity results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Threshold to filter irrelevant results
                results.append(self.endpoints[idx])
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[APIEndpointInfo]:
        """Simple keyword-based search fallback."""
        query_terms = set(query.lower().split())
        
        scored_endpoints = []
        for ep in self.endpoints:
            # Create searchable text
            text = f"{ep.name} {ep.action} {ep.description}".lower()
            
            # Count matching terms
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored_endpoints.append((score, ep))
        
        # Sort by score and return top-k
        scored_endpoints.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored_endpoints[:top_k]]
    
    def generate_api_context(
        self, 
        endpoints: List[APIEndpointInfo],
        include_examples: bool = True
    ) -> str:
        """
        Generate LLM-friendly API context from a list of endpoints.
        
        Args:
            endpoints: List of API endpoints to include
            include_examples: Whether to include example requests
            
        Returns:
            Formatted string suitable for LLM prompts
        """
        if not endpoints:
            return ""
        
        lines = ["## Relevant API Endpoints\n"]
        lines.append("The following API endpoints may be useful for your task:\n")
        
        for ep in endpoints:
            lines.append(f"### {ep.name} ({ep.api_name})")
            lines.append(f"- **Action/Endpoint:** `{ep.action}`")
            lines.append(f"- **HTTP Method:** {ep.http_method}")
            lines.append(f"- **URL:** `{ep.url_pattern}`")
            lines.append(f"- **Description:** {ep.description}")
            
            if ep.requires_auth:
                if ep.auth_type == "api_key":
                    lines.append(f"- **Authentication:** Requires API Key (available as `YOUTUBE_API_KEY` variable)")
                elif ep.auth_type == "bearer_token":
                    lines.append(f"- **Authentication:** Requires Bearer Token")
                else:
                    lines.append(f"- **Authentication:** Required")
            else:
                lines.append("- **Authentication:** Not required")
            
            # Add key parameters
            if ep.parameters:
                required_params = [p for p in ep.parameters if p.get("required", False)]
                if required_params:
                    lines.append("- **Required Parameters:**")
                    for p in required_params[:5]:  # Limit to 5 required params
                        lines.append(f"  - `{p.get('name')}` ({p.get('type', 'string')}): {p.get('description', '')[:100]}")
            
            # Add examples if requested
            if include_examples and ep.examples:
                lines.append("- **Example:**")
                for ex in ep.examples[:1]:  # Only show first example
                    lines.append(f"  ```")
                    lines.append(f"  {ex.get('url', ex.get('description', ''))}")
                    lines.append(f"  ```")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def get_auth_requirements(self, endpoints: List[APIEndpointInfo]) -> Dict[str, bool]:
        """
        Check authentication requirements for a list of endpoints.
        
        Args:
            endpoints: List of API endpoints to check
            
        Returns:
            Dictionary with auth type keys and boolean values indicating if needed
        """
        requirements = {
            "youtube_api_key": False,
            "orcid_token": False,
            "mediawiki_auth": False
        }
        
        for ep in endpoints:
            if ep.requires_auth:
                if "youtube" in ep.api_name.lower():
                    requirements["youtube_api_key"] = True
                elif "orcid" in ep.api_name.lower():
                    requirements["orcid_token"] = True
                elif "mediawiki" in ep.api_name.lower():
                    requirements["mediawiki_auth"] = True
        
        return requirements


def get_api_context_for_task(task: str, top_k: int = 3) -> Tuple[str, Dict[str, bool]]:
    """
    Convenience function to get API context for a given task.
    
    Args:
        task: The task description
        top_k: Maximum number of relevant endpoints to return
        
    Returns:
        Tuple of (api_context_string, auth_requirements_dict)
    """
    retriever = APIRetriever()
    endpoints = retriever.search(task, top_k=top_k)
    context = retriever.generate_api_context(endpoints)
    auth_reqs = retriever.get_auth_requirements(endpoints)
    return context, auth_reqs


# Singleton instance for reuse
_global_retriever: Optional[APIRetriever] = None


def get_global_retriever() -> APIRetriever:
    """Get or create a global APIRetriever instance."""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = APIRetriever()
    return _global_retriever


if __name__ == "__main__":
    # Test the API retriever
    retriever = APIRetriever()
    print(f"Loaded {len(retriever.endpoints)} endpoints from {len(retriever.api_databases)} APIs")
    
    # Test search
    test_queries = [
        "search for youtube videos",
        "get wikipedia page content",
        "find ORCID researcher profile",
        "list youtube comments"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        endpoints = retriever.search(query, top_k=3)
        print(retriever.generate_api_context(endpoints))
        print(f"Auth requirements: {retriever.get_auth_requirements(endpoints)}")
