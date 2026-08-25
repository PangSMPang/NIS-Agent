"""
MediaWiki API Database Schema Definition
用于让LLM理解和调用MediaWiki API的结构化数据模型

This module defines the data structures for storing API endpoint information
in a format that is easy for LLMs to parse and understand.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class HttpMethod(str, Enum):
    """HTTP请求方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class ParameterType(str, Enum):
    """参数数据类型"""
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    FILE = "file"
    ARRAY = "array"
    ENUM = "enum"


class ParameterLocation(str, Enum):
    """参数位置"""
    QUERY = "query"
    BODY = "body"
    HEADER = "header"
    PATH = "path"


@dataclass
class Parameter:
    """API参数定义"""
    name: str
    type: str
    required: bool
    description: str
    location: str = "query"
    default: Optional[str] = None
    enum_values: Optional[List[str]] = None
    example: Optional[str] = None
    deprecated: bool = False
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ResponseField:
    """响应字段定义"""
    name: str
    type: str
    description: str
    nullable: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Response:
    """API响应定义"""
    status_code: int
    description: str
    example: Optional[Dict[str, Any]] = None
    fields: List[ResponseField] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['fields'] = [f.to_dict() for f in self.fields] if self.fields else []
        return d


@dataclass
class APIEndpoint:
    """API端点定义"""
    action: str
    name: str
    description: str
    
    http_method: str
    url_pattern: str
    
    requires_auth: bool = False
    required_rights: List[str] = field(default_factory=list)
    requires_token: bool = False
    token_type: Optional[str] = None
    
    parameters: List[Parameter] = field(default_factory=list)
    
    responses: List[Response] = field(default_factory=list)
    
    category: str = "general"
    documentation_url: Optional[str] = None
    examples: List[Dict[str, str]] = field(default_factory=list)
    notes: Optional[str] = None
    deprecated: bool = False
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['parameters'] = [p.to_dict() for p in self.parameters]
        d['responses'] = [r.to_dict() for r in self.responses]
        return d
    
    def to_llm_prompt(self) -> str:
        """生成LLM友好的端点描述"""
        lines = [
            f"## API: {self.name}",
            f"**Action:** `{self.action}`",
            f"**Description:** {self.description}",
            f"**HTTP Method:** {self.http_method}",
            f"**URL:** `{self.url_pattern}`",
            f"**Requires Authentication:** {'Yes' if self.requires_auth else 'No'}",
        ]
        
        if self.requires_token:
            lines.append(f"**Requires Token:** {self.token_type or 'csrf'}")
        
        if self.required_rights:
            lines.append(f"**Required Rights:** {', '.join(self.required_rights)}")
        
        if self.parameters:
            lines.append("\n**Parameters:**")
            lines.append("| Name | Type | Required | Description |")
            lines.append("|------|------|----------|-------------|")
            for p in self.parameters:
                req = "Yes" if p.required else "No"
                lines.append(f"| `{p.name}` | {p.type} | {req} | {p.description} |")
        
        if self.examples:
            lines.append("\n**Examples:**")
            for ex in self.examples:
                lines.append(f"- {ex.get('description', 'Example')}:")
                lines.append(f"  ```")
                lines.append(f"  {ex.get('url', '')}")
                lines.append(f"  ```")
        
        for resp in self.responses:
            if resp.example:
                status = "Success" if resp.status_code == 200 else "Error"
                lines.append(f"\n**{status} Response ({resp.status_code}):**")
                lines.append("```json")
                lines.append(json.dumps(resp.example, indent=2))
                lines.append("```")
        
        return "\n".join(lines)


@dataclass
class APIDatabase:
    """API数据库"""
    name: str
    version: str
    base_url: str
    description: str
    
    auth_methods: List[Dict[str, str]] = field(default_factory=list)
    
    endpoints: List[APIEndpoint] = field(default_factory=list)
    
    global_parameters: List[Parameter] = field(default_factory=list)
    
    error_codes: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "base_url": self.base_url,
            "description": self.description,
            "auth_methods": self.auth_methods,
            "global_parameters": [p.to_dict() for p in self.global_parameters],
            "endpoints": [e.to_dict() for e in self.endpoints],
            "error_codes": self.error_codes
        }
    
    def to_json(self, indent: int = 2) -> str:
        """导出为JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, filepath: str):
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'APIDatabase':
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'APIDatabase':
        """从字典创建"""
        db = cls(
            name=data["name"],
            version=data["version"],
            base_url=data["base_url"],
            description=data["description"],
            auth_methods=data.get("auth_methods", []),
            error_codes=data.get("error_codes", [])
        )
        
        for p in data.get("global_parameters", []):
            db.global_parameters.append(Parameter(**p))
        
        for ep in data.get("endpoints", []):
            params = [Parameter(**p) for p in ep.get("parameters", [])]
            responses = []
            for r in ep.get("responses", []):
                fields = [ResponseField(**f) for f in r.get("fields", [])]
                responses.append(Response(
                    status_code=r["status_code"],
                    description=r["description"],
                    example=r.get("example"),
                    fields=fields
                ))
            
            endpoint = APIEndpoint(
                action=ep["action"],
                name=ep["name"],
                description=ep["description"],
                http_method=ep["http_method"],
                url_pattern=ep["url_pattern"],
                requires_auth=ep.get("requires_auth", False),
                required_rights=ep.get("required_rights", []),
                requires_token=ep.get("requires_token", False),
                token_type=ep.get("token_type"),
                parameters=params,
                responses=responses,
                category=ep.get("category", "general"),
                documentation_url=ep.get("documentation_url"),
                examples=ep.get("examples", []),
                notes=ep.get("notes"),
                deprecated=ep.get("deprecated", False)
            )
            db.endpoints.append(endpoint)
        
        return db
    
    def search_by_action(self, action: str) -> Optional[APIEndpoint]:
        """按action搜索端点"""
        for ep in self.endpoints:
            if ep.action.lower() == action.lower():
                return ep
        return None
    
    def search_by_category(self, category: str) -> List[APIEndpoint]:
        """按分类搜索端点"""
        return [ep for ep in self.endpoints if ep.category.lower() == category.lower()]
    
    def search_by_keyword(self, keyword: str) -> List[APIEndpoint]:
        """按关键词搜索端点"""
        keyword = keyword.lower()
        results = []
        for ep in self.endpoints:
            if (keyword in ep.name.lower() or 
                keyword in ep.description.lower() or
                keyword in ep.action.lower()):
                results.append(ep)
        return results
    
    def get_endpoint_summary(self) -> str:
        """生成端点摘要"""
        lines = [
            f"# {self.name} API Documentation",
            f"**Version:** {self.version}",
            f"**Base URL:** `{self.base_url}`",
            f"\n{self.description}",
            "\n## Available Endpoints by Category\n"
        ]
        
        categories = {}
        for ep in self.endpoints:
            cat = ep.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ep)
        
        for cat, eps in sorted(categories.items()):
            lines.append(f"### {cat.title()}")
            for ep in eps:
                method = ep.http_method
                auth = "🔐" if ep.requires_auth else ""
                lines.append(f"- `{ep.action}` - {ep.name} [{method}] {auth}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_llm_context(self, actions: Optional[List[str]] = None) -> str:
        """
        生成LLM可读的API上下文
        
        Args:
            actions: 指定要包含的action列表，None表示全部
        
        Returns:
            LLM友好的API文档字符串
        """
        lines = [
            f"# {self.name}",
            f"\n## Overview",
            f"{self.description}",
            f"\n**Base URL:** `{self.base_url}`",
            f"**Version:** {self.version}",
        ]
        
        if self.auth_methods:
            lines.append("\n## Authentication")
            for auth in self.auth_methods:
                lines.append(f"- **{auth.get('name', 'Unknown')}**: {auth.get('description', '')}")
        
        if self.global_parameters:
            lines.append("\n## Global Parameters")
            lines.append("These parameters can be used with any endpoint:")
            lines.append("| Name | Type | Description |")
            lines.append("|------|------|-------------|")
            for p in self.global_parameters:
                lines.append(f"| `{p.name}` | {p.type} | {p.description} |")
        
        lines.append("\n## Endpoints\n")
        
        endpoints_to_include = self.endpoints
        if actions:
            endpoints_to_include = [ep for ep in self.endpoints if ep.action in actions]
        
        for ep in endpoints_to_include:
            lines.append(ep.to_llm_prompt())
            lines.append("\n---\n")
        
        if self.error_codes:
            lines.append("\n## Error Codes")
            lines.append("| Code | Description |")
            lines.append("|------|-------------|")
            for err in self.error_codes:
                lines.append(f"| `{err.get('code', '')}` | {err.get('description', '')} |")
        
        return "\n".join(lines)


if __name__ == "__main__":
    test_param = Parameter(
        name="title",
        type="string",
        required=True,
        description="Page title to edit",
        example="Test_Page"
    )
    
    test_response = Response(
        status_code=200,
        description="Success response",
        example={"edit": {"result": "Success", "pageid": 12345}}
    )
    
    test_endpoint = APIEndpoint(
        action="edit",
        name="Edit Page",
        description="Create and edit pages",
        http_method="POST",
        url_pattern="/w/api.php?action=edit",
        requires_auth=True,
        requires_token=True,
        token_type="csrf",
        parameters=[test_param],
        responses=[test_response],
        category="page_operations"
    )
    
    print("Test endpoint to_llm_prompt():")
    print(test_endpoint.to_llm_prompt())
