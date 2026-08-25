"""
MediaWiki API Query Tool for LLM
提供给LLM使用的API查询工具

这个工具可以：
1. 按action名称查找API端点
2. 按功能分类搜索端点
3. 按关键词搜索端点
4. 生成API调用示例
5. 输出LLM友好的API文档
"""

import json
import os
from typing import Optional, List, Dict, Any
from mediawiki_api_schema import APIDatabase, APIEndpoint


class MediaWikiAPITool:
    """MediaWiki API查询工具"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化工具
        
        Args:
            db_path: API数据库JSON文件路径，默认为同目录下的mediawiki_api.json
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "mediawiki_api.json"
            )
        
        self.db = APIDatabase.load(db_path)
        self._build_index()
    
    def _build_index(self):
        """构建索引以加速搜索"""
        self._action_index = {}
        self._category_index = {}
        
        for ep in self.db.endpoints:
            self._action_index[ep.action.lower()] = ep
            
            cat = ep.category.lower()
            if cat not in self._category_index:
                self._category_index[cat] = []
            self._category_index[cat].append(ep)
    
    def get_api_overview(self) -> str:
        """
        获取API概览
        
        Returns:
            API概览字符串
        """
        return f"""# {self.db.name}

**Version:** {self.db.version}
**Base URL:** `{self.db.base_url}`

{self.db.description}

## Available Categories
{chr(10).join(f"- **{cat}**: {len(eps)} endpoints" for cat, eps in sorted(self._category_index.items()))}

## Authentication Methods
{chr(10).join(f"- **{auth['name']}**: {auth['description']}" for auth in self.db.auth_methods)}
"""
    
    def find_endpoint(self, action: str) -> Optional[str]:
        """
        按action名称查找端点
        
        Args:
            action: API action名称，如 "edit", "query", "login"
        
        Returns:
            端点的LLM友好描述，如果未找到返回None
        """
        action = action.lower()
        
        if action in self._action_index:
            return self._action_index[action].to_llm_prompt()
        
        matches = []
        for key, ep in self._action_index.items():
            if action in key or key in action:
                matches.append(ep)
        
        if matches:
            if len(matches) == 1:
                return matches[0].to_llm_prompt()
            else:
                result = f"Found {len(matches)} matching endpoints:\n\n"
                for ep in matches:
                    result += f"### {ep.name} (`{ep.action}`)\n"
                    result += f"{ep.description}\n\n"
                return result
        
        return None
    
    def list_endpoints_by_category(self, category: str) -> str:
        """
        按分类列出端点
        
        Args:
            category: 分类名称
        
        Returns:
            分类下所有端点的描述
        """
        category = category.lower()
        
        if category not in self._category_index:
            for cat in self._category_index:
                if category in cat or cat in category:
                    category = cat
                    break
            else:
                available = ", ".join(sorted(self._category_index.keys()))
                return f"Category '{category}' not found. Available categories: {available}"
        
        endpoints = self._category_index[category]
        result = f"# {category.title()} Endpoints ({len(endpoints)} total)\n\n"
        
        for ep in endpoints:
            result += ep.to_llm_prompt()
            result += "\n\n---\n\n"
        
        return result
    
    def search(self, keyword: str) -> str:
        """
        按关键词搜索端点
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            匹配的端点列表
        """
        results = self.db.search_by_keyword(keyword)
        
        if not results:
            return f"No endpoints found matching '{keyword}'"
        
        output = f"# Search Results for '{keyword}' ({len(results)} found)\n\n"
        
        for ep in results:
            output += f"## {ep.name}\n"
            output += f"**Action:** `{ep.action}`\n"
            output += f"**Category:** {ep.category}\n"
            output += f"**Description:** {ep.description}\n"
            output += f"**HTTP Method:** {ep.http_method}\n"
            if ep.requires_auth:
                output += "**Requires Authentication:** Yes\n"
            output += "\n---\n\n"
        
        return output
    
    def generate_api_call_example(
        self,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        base_url: str = "https://en.wikipedia.org/w/api.php"
    ) -> str:
        """
        生成API调用示例
        
        Args:
            action: API action名称
            params: 参数字典
            base_url: 基础URL
        
        Returns:
            API调用示例代码
        """
        endpoint = self._action_index.get(action.lower())
        if not endpoint:
            return f"Unknown action: {action}"
        
        url_params = {"action": action.split("+")[0], "format": "json"}
        if params:
            url_params.update(params)
        
        if "+" in action:
            parts = action.split("+")
            if parts[0] == "query":
                if parts[1] in ["tokens"]:
                    url_params["meta"] = parts[1]
                elif parts[1] in ["search"]:
                    url_params["list"] = parts[1]
        
        param_str = "&".join(f"{k}={v}" for k, v in url_params.items())
        full_url = f"{base_url}?{param_str}"
        
        example = f"""# {endpoint.name}
# {endpoint.description}

## URL
```
{full_url}
```

## Python Example
```python
import requests

url = "{base_url}"
params = {json.dumps(url_params, indent=4)}

response = requests.{'post' if endpoint.http_method == 'POST' else 'get'}(url, {'data' if endpoint.http_method == 'POST' else 'params'}=params)
data = response.json()
print(data)
```

## cURL Example
```bash
curl {'-X POST ' if endpoint.http_method == 'POST' else ''}"{full_url}"
```
"""
        
        if endpoint.requires_token:
            example += f"""
## Note: Token Required
This endpoint requires a {endpoint.token_type or 'csrf'} token. First get a token:
```
{base_url}?action=query&meta=tokens&type={endpoint.token_type or 'csrf'}&format=json
```
Then include the token in your request as the `token` parameter.
"""
        
        return example
    
    def get_authentication_guide(self) -> str:
        """
        获取认证指南
        
        Returns:
            认证指南文档
        """
        return """# MediaWiki API Authentication Guide

## Overview
MediaWiki API uses cookie-based authentication with CSRF token protection for write operations.

## Step-by-Step Login Process

### 1. Get Login Token
```
GET /w/api.php?action=query&meta=tokens&type=login&format=json
```

Response:
```json
{
    "query": {
        "tokens": {
            "logintoken": "abc123+\\\\"
        }
    }
}
```

### 2. Login Request
```
POST /w/api.php?action=login&lgname=USERNAME&lgpassword=PASSWORD&lgtoken=TOKEN&format=json
```

Response (Success):
```json
{
    "login": {
        "result": "Success",
        "lguserid": 12345,
        "lgusername": "YourUsername"
    }
}
```

### 3. Get CSRF Token (for write operations)
After login, get a CSRF token for editing and other write operations:
```
GET /w/api.php?action=query&meta=tokens&type=csrf&format=json
```

## Bot Passwords (Recommended)
For bots and automated scripts, use Special:BotPasswords instead of main account:
1. Go to Special:BotPasswords on your wiki
2. Create a bot password with required permissions
3. Login format: `username@botname` with the generated password

## Token Types
| Token Type | Used For |
|------------|----------|
| csrf | Most write operations (edit, delete, move, etc.) |
| login | Login requests |
| createaccount | Account creation |
| patrol | Patrolling revisions |
| rollback | Rollback operations |
| userrights | Changing user rights |
| watch | Watch/unwatch pages |

## Session Management
- Cookies are used to maintain session
- Store and send cookies with each request
- Sessions expire after a period of inactivity
"""
    
    def get_quick_reference(self) -> str:
        """
        获取快速参考卡片
        
        Returns:
            常用操作快速参考
        """
        return """# MediaWiki API Quick Reference

## Common Operations

### Read Page Content
```
GET /w/api.php?action=query&titles=PAGE_TITLE&prop=revisions&rvprop=content&rvslots=main&format=json
```

### Search Pages
```
GET /w/api.php?action=query&list=search&srsearch=KEYWORD&format=json
```

### Parse Wikitext to HTML
```
GET /w/api.php?action=parse&page=PAGE_TITLE&format=json
```

### Get Page Info
```
GET /w/api.php?action=query&titles=PAGE_TITLE&prop=info&format=json
```

### Edit Page (requires auth + token)
```
POST /w/api.php?action=edit&title=PAGE_TITLE&text=CONTENT&summary=SUMMARY&token=TOKEN&format=json
```

### Get CSRF Token
```
GET /w/api.php?action=query&meta=tokens&type=csrf&format=json
```

### Get User Info
```
GET /w/api.php?action=query&meta=userinfo&uiprop=rights|editcount&format=json
```

### Get Recent Changes
```
GET /w/api.php?action=query&list=recentchanges&rclimit=50&format=json
```

### OpenSearch (Title Suggestions)
```
GET /w/api.php?action=opensearch&search=QUERY&limit=10&format=json
```

## Format Parameters
- `format=json` - JSON output (recommended)
- `formatversion=2` - Modern format (cleaner arrays)

## Rate Limiting
- Set `maxlag` parameter to avoid overloading servers
- Example: `&maxlag=5` - only proceed if lag < 5 seconds

## Wikipedia Endpoints
- English: https://en.wikipedia.org/w/api.php
- Chinese: https://zh.wikipedia.org/w/api.php
- Japanese: https://ja.wikipedia.org/w/api.php
"""
    
    def list_all_actions(self) -> str:
        """
        列出所有可用的actions
        
        Returns:
            所有actions的列表
        """
        output = "# All Available API Actions\n\n"
        
        for cat in sorted(self._category_index.keys()):
            output += f"## {cat.replace('_', ' ').title()}\n\n"
            output += "| Action | Name | Method | Auth Required |\n"
            output += "|--------|------|--------|---------------|\n"
            
            for ep in self._category_index[cat]:
                auth = "✓" if ep.requires_auth else ""
                output += f"| `{ep.action}` | {ep.name} | {ep.http_method} | {auth} |\n"
            
            output += "\n"
        
        return output
    
    def get_error_codes(self) -> str:
        """
        获取错误码列表
        
        Returns:
            错误码文档
        """
        output = "# MediaWiki API Error Codes\n\n"
        output += "| Code | Description |\n"
        output += "|------|-------------|\n"
        
        for err in self.db.error_codes:
            output += f"| `{err['code']}` | {err['description']} |\n"
        
        output += """
## Error Response Format
```json
{
    "error": {
        "code": "error-code",
        "info": "Error description",
        "docref": "Documentation reference URL"
    }
}
```

## Common Solutions
- `badtoken`: Get a fresh CSRF token before the operation
- `permissiondenied`: Check user rights, may need to login
- `ratelimited`: Wait and retry, add `maxlag` parameter
- `readonly`: Wiki is in maintenance mode, wait and retry
"""
        
        return output


def interactive_mode():
    """交互式模式"""
    tool = MediaWikiAPITool()
    
    print("=" * 60)
    print("MediaWiki API Query Tool")
    print("=" * 60)
    print("\nCommands:")
    print("  find <action>     - Find endpoint by action name")
    print("  category <name>   - List endpoints by category")
    print("  search <keyword>  - Search endpoints")
    print("  example <action>  - Generate API call example")
    print("  overview          - Show API overview")
    print("  actions           - List all actions")
    print("  auth              - Authentication guide")
    print("  quick             - Quick reference")
    print("  errors            - Error codes")
    print("  quit              - Exit")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "quit" or cmd == "exit":
            print("Goodbye!")
            break
        elif cmd == "find":
            if arg:
                result = tool.find_endpoint(arg)
                print(result or f"Endpoint '{arg}' not found")
            else:
                print("Usage: find <action>")
        elif cmd == "category":
            if arg:
                print(tool.list_endpoints_by_category(arg))
            else:
                print("Available categories:", ", ".join(sorted(tool._category_index.keys())))
        elif cmd == "search":
            if arg:
                print(tool.search(arg))
            else:
                print("Usage: search <keyword>")
        elif cmd == "example":
            if arg:
                print(tool.generate_api_call_example(arg))
            else:
                print("Usage: example <action>")
        elif cmd == "overview":
            print(tool.get_api_overview())
        elif cmd == "actions":
            print(tool.list_all_actions())
        elif cmd == "auth":
            print(tool.get_authentication_guide())
        elif cmd == "quick":
            print(tool.get_quick_reference())
        elif cmd == "errors":
            print(tool.get_error_codes())
        else:
            print(f"Unknown command: {cmd}")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        tool = MediaWikiAPITool()
        cmd = sys.argv[1].lower()
        
        if cmd == "find" and len(sys.argv) > 2:
            result = tool.find_endpoint(sys.argv[2])
            print(result or f"Endpoint '{sys.argv[2]}' not found")
        elif cmd == "category" and len(sys.argv) > 2:
            print(tool.list_endpoints_by_category(sys.argv[2]))
        elif cmd == "search" and len(sys.argv) > 2:
            print(tool.search(sys.argv[2]))
        elif cmd == "overview":
            print(tool.get_api_overview())
        elif cmd == "actions":
            print(tool.list_all_actions())
        elif cmd == "auth":
            print(tool.get_authentication_guide())
        elif cmd == "quick":
            print(tool.get_quick_reference())
        elif cmd == "errors":
            print(tool.get_error_codes())
        else:
            print("Usage: python api_query_tool.py <command> [args]")
            print("Commands: find, category, search, overview, actions, auth, quick, errors")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
