"""
MediaWiki API Database Builder
构建MediaWiki Action API的结构化数据库（仅查询端点，无需认证）

运行此脚本将生成:
1. mediawiki_api.json - 完整的API数据库
2. mediawiki_api_llm.md - LLM友好的Markdown文档
"""

from mediawiki_api_schema import (
    APIDatabase, APIEndpoint, Parameter, Response, ResponseField
)
import json
import os


def build_mediawiki_api_database() -> APIDatabase:
    """构建MediaWiki API数据库（仅包含无需认证的查询端点）"""
    
    db = APIDatabase(
        name="MediaWiki Action API (Read-Only)",
        version="1.43",
        base_url="https://{wiki-domain}/w/api.php",
        description="""The MediaWiki Action API is a web service that provides programmatic access 
to wiki content. This database contains only READ-ONLY endpoints that do NOT require authentication.
All requests are made to the api.php endpoint with an 'action' parameter specifying the operation.

Common wiki endpoints:
- Wikipedia (English): https://en.wikipedia.org/w/api.php
- MediaWiki.org: https://www.mediawiki.org/w/api.php
- Wikimedia Commons: https://commons.wikimedia.org/w/api.php

NOTE: This is a read-only subset of the API. No authentication is required for these endpoints."""
    )
    
    db.auth_methods = [
        {
            "name": "No Authentication Required",
            "description": "All endpoints in this database are read-only and do not require authentication"
        }
    ]
    
    db.global_parameters = [
        Parameter(
            name="format",
            type="enum",
            required=False,
            description="Output format for the response",
            default="json",
            enum_values=["json", "jsonfm", "xml", "xmlfm", "php", "none", "rawfm"]
        ),
        Parameter(
            name="formatversion",
            type="enum",
            required=False,
            description="Output format version. Use 2 for cleaner output",
            default="1",
            enum_values=["1", "2", "latest"]
        ),
        Parameter(
            name="errorformat",
            type="enum",
            required=False,
            description="Format for error messages",
            default="bc",
            enum_values=["bc", "html", "wikitext", "plaintext", "raw", "none"]
        ),
        Parameter(
            name="maxlag",
            type="integer",
            required=False,
            description="Maximum replication lag in seconds. Useful for bots to avoid overloading"
        ),
        Parameter(
            name="assert",
            type="enum",
            required=False,
            description="Verify user state before performing action",
            enum_values=["anon", "user", "bot"]
        ),
        Parameter(
            name="curtimestamp",
            type="boolean",
            required=False,
            description="Include current timestamp in the result"
        )
    ]
    
    db.error_codes = [
        {"code": "unknownaction", "description": "Unrecognized action parameter value"},
        {"code": "missingtitle", "description": "The specified page does not exist"},
        {"code": "ratelimited", "description": "Action rate limited, try again later"},
        {"code": "invalidtitle", "description": "Invalid page title provided"},
        {"code": "nosuchpageid", "description": "There is no page with ID specified"},
        {"code": "nosuchrevid", "description": "There is no revision with ID specified"},
        {"code": "badcontinue", "description": "Invalid continuation parameter"}
    ]
    
    
    
    # action=query (main)
    db.endpoints.append(APIEndpoint(
        action="query",
        name="Query",
        description="Fetch data from and about MediaWiki. The main read operation for getting page content, metadata, and lists.",
        http_method="GET",
        url_pattern="/w/api.php?action=query",
        requires_auth=False,
        required_rights=["read"],
        category="query",
        documentation_url="https://www.mediawiki.org/wiki/API:Query",
        parameters=[
            Parameter(
                name="titles",
                type="string",
                required=False,
                description="Pipe-separated list of page titles to query",
                example="Albert_Einstein|Isaac_Newton"
            ),
            Parameter(
                name="pageids",
                type="string",
                required=False,
                description="Pipe-separated list of page IDs",
                example="12345|67890"
            ),
            Parameter(
                name="revids",
                type="string",
                required=False,
                description="Pipe-separated list of revision IDs"
            ),
            Parameter(
                name="prop",
                type="string",
                required=False,
                description="Properties to fetch: revisions, categories, links, images, info, etc.",
                example="revisions|categories|info"
            ),
            Parameter(
                name="list",
                type="string",
                required=False,
                description="Lists to query: allpages, search, recentchanges, usercontribs, etc.",
                example="search"
            ),
            Parameter(
                name="meta",
                type="string",
                required=False,
                description="Meta information: siteinfo, tokens, userinfo, etc.",
                example="siteinfo|userinfo"
            ),
            Parameter(
                name="generator",
                type="string",
                required=False,
                description="Use a list/prop module as a generator of pages"
            ),
            Parameter(
                name="redirects",
                type="boolean",
                required=False,
                description="Automatically resolve redirects"
            ),
            Parameter(
                name="converttitles",
                type="boolean",
                required=False,
                description="Convert titles to other variants if needed"
            ),
            Parameter(
                name="export",
                type="boolean",
                required=False,
                description="Export pages in XML format"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Query successful",
                example={
                    "query": {
                        "pages": {
                            "12345": {
                                "pageid": 12345,
                                "ns": 0,
                                "title": "Example Page",
                                "revisions": [
                                    {
                                        "revid": 67890,
                                        "parentid": 67889,
                                        "user": "ExampleUser",
                                        "timestamp": "2024-01-15T12:00:00Z",
                                        "slots": {
                                            "main": {
                                                "contentmodel": "wikitext",
                                                "contentformat": "text/x-wiki",
                                                "content": "Page content here..."
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            )
        ],
        examples=[
            {
                "description": "Get page content",
                "url": "/w/api.php?action=query&titles=Example&prop=revisions&rvprop=content&rvslots=main&format=json"
            },
            {
                "description": "Search for pages",
                "url": "/w/api.php?action=query&list=search&srsearch=keyword&format=json"
            },
            {
                "description": "Get user info",
                "url": "/w/api.php?action=query&meta=userinfo&uiprop=rights|editcount&format=json"
            }
        ]
    ))
    
    # action=parse
    db.endpoints.append(APIEndpoint(
        action="parse",
        name="Parse Wikitext",
        description="Parse wikitext content and return HTML. Can parse page content or arbitrary wikitext.",
        http_method="GET",
        url_pattern="/w/api.php?action=parse",
        requires_auth=False,
        required_rights=["read"],
        category="query",
        documentation_url="https://www.mediawiki.org/wiki/API:Parsing_wikitext",
        parameters=[
            Parameter(
                name="page",
                type="string",
                required=False,
                description="Page title to parse",
                example="Main_Page"
            ),
            Parameter(
                name="pageid",
                type="integer",
                required=False,
                description="Page ID to parse"
            ),
            Parameter(
                name="text",
                type="string",
                required=False,
                description="Raw wikitext to parse"
            ),
            Parameter(
                name="title",
                type="string",
                required=False,
                description="Page title for context when parsing raw text"
            ),
            Parameter(
                name="revid",
                type="integer",
                required=False,
                description="Revision ID to parse"
            ),
            Parameter(
                name="prop",
                type="string",
                required=False,
                description="Properties to return: text, langlinks, categories, links, templates, etc.",
                default="text|langlinks|categories|links|templates|images|externallinks|sections|revid|displaytitle|iwlinks|properties|parsewarnings"
            ),
            Parameter(
                name="section",
                type="string",
                required=False,
                description="Section number to parse (0 for lead section)"
            ),
            Parameter(
                name="preview",
                type="boolean",
                required=False,
                description="Parse in preview mode"
            ),
            Parameter(
                name="disabletoc",
                type="boolean",
                required=False,
                description="Disable table of contents in output"
            ),
            Parameter(
                name="disableeditsection",
                type="boolean",
                required=False,
                description="Disable edit section links"
            ),
            Parameter(
                name="contentformat",
                type="string",
                required=False,
                description="Content format of the text parameter"
            ),
            Parameter(
                name="contentmodel",
                type="string",
                required=False,
                description="Content model of the text parameter"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Parse successful",
                example={
                    "parse": {
                        "title": "Pet door",
                        "pageid": 12345,
                        "text": {
                            "*": "<div class=\"mw-parser-output\">...</div>"
                        },
                        "categories": [
                            {"*": "Cats", "sortkey": ""}
                        ],
                        "sections": [
                            {
                                "toclevel": 1,
                                "level": "2",
                                "line": "History",
                                "number": "1",
                                "index": "1"
                            }
                        ]
                    }
                }
            )
        ],
        examples=[
            {
                "description": "Parse a page and get HTML",
                "url": "/w/api.php?action=parse&page=Pet_door&format=json"
            },
            {
                "description": "Parse raw wikitext",
                "url": "/w/api.php?action=parse&text={{PAGENAME}}&title=Test&format=json"
            }
        ]
    ))
    
    
    # action=opensearch
    db.endpoints.append(APIEndpoint(
        action="opensearch",
        name="OpenSearch",
        description="Search wiki page titles using OpenSearch protocol. Returns suggestions.",
        http_method="GET",
        url_pattern="/w/api.php?action=opensearch",
        requires_auth=False,
        required_rights=["read"],
        category="search",
        documentation_url="https://www.mediawiki.org/wiki/API:Opensearch",
        parameters=[
            Parameter(
                name="search",
                type="string",
                required=True,
                description="Search string"
            ),
            Parameter(
                name="namespace",
                type="string",
                required=False,
                description="Namespaces to search (pipe-separated)",
                default="0"
            ),
            Parameter(
                name="limit",
                type="integer",
                required=False,
                description="Maximum results to return",
                default="10"
            ),
            Parameter(
                name="suggest",
                type="boolean",
                required=False,
                description="Enable search suggestions",
                deprecated=True
            ),
            Parameter(
                name="redirects",
                type="enum",
                required=False,
                description="How to handle redirects",
                enum_values=["return", "resolve"]
            ),
            Parameter(
                name="warningsaserror",
                type="boolean",
                required=False,
                description="Treat warnings as errors"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Search successful (OpenSearch format)",
                example=[
                    "Cat",
                    ["Cat", "Caterpillar", "Category"],
                    ["A cat is a small domesticated carnivorous mammal...", "", ""],
                    ["https://en.wikipedia.org/wiki/Cat", "", ""]
                ]
            )
        ],
        examples=[
            {
                "description": "Search for pages starting with 'cat'",
                "url": "/w/api.php?action=opensearch&search=cat&limit=10&format=json"
            }
        ]
    ))
    
    # action=query&list=search
    db.endpoints.append(APIEndpoint(
        action="query+search",
        name="Full-text Search",
        description="Search for pages by title or content text. More powerful than opensearch.",
        http_method="GET",
        url_pattern="/w/api.php?action=query&list=search",
        requires_auth=False,
        required_rights=["read"],
        category="search",
        documentation_url="https://www.mediawiki.org/wiki/API:Search",
        parameters=[
            Parameter(
                name="srsearch",
                type="string",
                required=True,
                description="Search query. Supports search operators like intitle:, incategory:, prefix:",
                example="machine learning intitle:introduction"
            ),
            Parameter(
                name="srnamespace",
                type="string",
                required=False,
                description="Namespaces to search (pipe-separated)",
                default="0"
            ),
            Parameter(
                name="srlimit",
                type="integer",
                required=False,
                description="Maximum results",
                default="10"
            ),
            Parameter(
                name="sroffset",
                type="integer",
                required=False,
                description="Offset for pagination"
            ),
            Parameter(
                name="srwhat",
                type="enum",
                required=False,
                description="What to search",
                enum_values=["title", "text", "nearmatch"]
            ),
            Parameter(
                name="srinfo",
                type="string",
                required=False,
                description="Metadata to return",
                enum_values=["totalhits", "suggestion", "rewrittenquery"]
            ),
            Parameter(
                name="srprop",
                type="string",
                required=False,
                description="Properties to return: size, wordcount, timestamp, snippet, etc.",
                example="size|wordcount|timestamp|snippet"
            ),
            Parameter(
                name="srsort",
                type="enum",
                required=False,
                description="Sort order",
                enum_values=["relevance", "just_match", "none", "incoming_links_asc", "incoming_links_desc", 
                            "last_edit_asc", "last_edit_desc", "create_timestamp_asc", "create_timestamp_desc"]
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Search successful",
                example={
                    "query": {
                        "searchinfo": {
                            "totalhits": 1234
                        },
                        "search": [
                            {
                                "ns": 0,
                                "title": "Machine learning",
                                "pageid": 12345,
                                "size": 50000,
                                "wordcount": 5000,
                                "snippet": "...<span class=\"searchmatch\">Machine</span> <span class=\"searchmatch\">learning</span>...",
                                "timestamp": "2024-01-15T12:00:00Z"
                            }
                        ]
                    }
                }
            )
        ],
        examples=[
            {
                "description": "Search for 'python programming'",
                "url": "/w/api.php?action=query&list=search&srsearch=python%20programming&srprop=snippet&format=json"
            }
        ]
    ))
    
    
    # action=expandtemplates
    db.endpoints.append(APIEndpoint(
        action="expandtemplates",
        name="Expand Templates",
        description="Expand templates in wikitext. Useful for testing template output.",
        http_method="GET",
        url_pattern="/w/api.php?action=expandtemplates",
        requires_auth=False,
        category="utility",
        documentation_url="https://www.mediawiki.org/wiki/API:Expandtemplates",
        parameters=[
            Parameter(
                name="text",
                type="string",
                required=True,
                description="Wikitext to expand"
            ),
            Parameter(
                name="title",
                type="string",
                required=False,
                description="Page title for context"
            ),
            Parameter(
                name="revid",
                type="integer",
                required=False,
                description="Revision ID for context"
            ),
            Parameter(
                name="prop",
                type="string",
                required=False,
                description="Properties to return: wikitext, categories, properties, ttl, etc."
            ),
            Parameter(
                name="includecomments",
                type="boolean",
                required=False,
                description="Include HTML comments in output"
            ),
            Parameter(
                name="generatexml",
                type="boolean",
                required=False,
                description="Generate XML parse tree",
                deprecated=True
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Expansion successful",
                example={
                    "expandtemplates": {
                        "wikitext": "Expanded template content here"
                    }
                }
            )
        ]
    ))
    
    # action=compare
    db.endpoints.append(APIEndpoint(
        action="compare",
        name="Compare Revisions",
        description="Get the difference between two page revisions.",
        http_method="GET",
        url_pattern="/w/api.php?action=compare",
        requires_auth=False,
        category="utility",
        documentation_url="https://www.mediawiki.org/wiki/API:Compare",
        parameters=[
            Parameter(
                name="fromtitle",
                type="string",
                required=False,
                description="First page title"
            ),
            Parameter(
                name="fromid",
                type="integer",
                required=False,
                description="First page ID"
            ),
            Parameter(
                name="fromrev",
                type="integer",
                required=False,
                description="First revision ID"
            ),
            Parameter(
                name="fromtext",
                type="string",
                required=False,
                description="Raw text for comparison"
            ),
            Parameter(
                name="totitle",
                type="string",
                required=False,
                description="Second page title"
            ),
            Parameter(
                name="toid",
                type="integer",
                required=False,
                description="Second page ID"
            ),
            Parameter(
                name="torev",
                type="integer",
                required=False,
                description="Second revision ID"
            ),
            Parameter(
                name="totext",
                type="string",
                required=False,
                description="Raw text for comparison"
            ),
            Parameter(
                name="prop",
                type="string",
                required=False,
                description="Properties: diff, diffsize, rel, ids, title, user, comment, parsedcomment, size"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Comparison successful",
                example={
                    "compare": {
                        "fromid": 12345,
                        "fromrevid": 67889,
                        "fromns": 0,
                        "fromtitle": "Test Page",
                        "toid": 12345,
                        "torevid": 67890,
                        "tons": 0,
                        "totitle": "Test Page",
                        "*": "<table class=\"diff\">...</table>"
                    }
                }
            )
        ]
    ))
    
    # action=feedcontributions
    db.endpoints.append(APIEndpoint(
        action="feedcontributions",
        name="User Contributions Feed",
        description="Get a user's contributions as an RSS/Atom feed.",
        http_method="GET",
        url_pattern="/w/api.php?action=feedcontributions",
        requires_auth=False,
        category="utility",
        parameters=[
            Parameter(
                name="user",
                type="string",
                required=True,
                description="Username"
            ),
            Parameter(
                name="namespace",
                type="integer",
                required=False,
                description="Namespace to filter"
            ),
            Parameter(
                name="feedformat",
                type="enum",
                required=False,
                description="Feed format",
                default="rss",
                enum_values=["rss", "atom"]
            ),
            Parameter(
                name="year",
                type="integer",
                required=False,
                description="Year to filter"
            ),
            Parameter(
                name="month",
                type="integer",
                required=False,
                description="Month to filter"
            ),
            Parameter(
                name="tagfilter",
                type="string",
                required=False,
                description="Filter by tag"
            ),
            Parameter(
                name="deletedonly",
                type="boolean",
                required=False,
                description="Show deleted contributions only"
            ),
            Parameter(
                name="toponly",
                type="boolean",
                required=False,
                description="Show only latest revision per page"
            ),
            Parameter(
                name="newonly",
                type="boolean",
                required=False,
                description="Show only page creations"
            ),
            Parameter(
                name="hideminor",
                type="boolean",
                required=False,
                description="Hide minor edits"
            ),
            Parameter(
                name="showsizediff",
                type="boolean",
                required=False,
                description="Show size difference"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="RSS/Atom feed returned"
            )
        ]
    ))
    
    return db


def main():
    """主函数：生成数据库文件"""
    print("Building MediaWiki API Database...")
    
    db = build_mediawiki_api_database()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    json_path = os.path.join(output_dir, "mediawiki_api.json")
    db.save(json_path)
    print(f"✓ Saved JSON database to: {json_path}")
    
    md_path = os.path.join(output_dir, "mediawiki_api_llm.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(db.generate_llm_context())
    print(f"✓ Saved LLM documentation to: {md_path}")
    
    summary_path = os.path.join(output_dir, "mediawiki_api_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(db.get_endpoint_summary())
    print(f"✓ Saved endpoint summary to: {summary_path}")
    
    print(f"\n📊 Database Statistics:")
    print(f"   - Total endpoints: {len(db.endpoints)}")
    print(f"   - Categories: {len(set(e.category for e in db.endpoints))}")
    print(f"   - Global parameters: {len(db.global_parameters)}")
    print(f"   - Error codes: {len(db.error_codes)}")
    
    categories = {}
    for ep in db.endpoints:
        cat = ep.category
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print("\n📁 Endpoints by Category:")
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count}")


if __name__ == "__main__":
    main()
