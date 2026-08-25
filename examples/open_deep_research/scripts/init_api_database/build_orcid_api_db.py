"""
ORCID Public API Database Builder
构建ORCID Public API的结构化数据库（仅包含数据读取和搜索功能）

ORCID (Open Researcher and Contributor ID) 是一个用于唯一标识研究人员的系统。
此API数据库仅包含用于搜索和读取公开研究者数据的端点。

运行此脚本将生成:
1. orcid_api.json - 完整的API数据库
2. orcid_api_llm.md - LLM友好的Markdown文档
"""

from mediawiki_api_schema import (
    APIDatabase, APIEndpoint, Parameter, Response, ResponseField
)
import json
import os


def build_orcid_api_database() -> APIDatabase:
    """构建ORCID Public API数据库"""
    
    db = APIDatabase(
        name="ORCID Public API",
        version="v3.0",
        base_url="https://pub.orcid.org/v3.0",
        description="""ORCID (Open Researcher and Contributor ID) Public API provides read-only access 
to public researcher data in the ORCID Registry.

Use this API to:
- Search for researchers by name, affiliation, works, etc.
- Read public profile data from ORCID records
- Retrieve publication lists, employment history, education, etc.

Base URLs:
- Production: https://pub.orcid.org/v3.0
- Sandbox (testing): https://pub.sandbox.orcid.org/v3.0

Documentation: https://info.orcid.org/documentation/api-tutorials/

Note: This database only includes read/search endpoints. Write operations require Member API."""
    )
    
    db.auth_methods = [
        {
            "name": "Public API (Read-only)",
            "description": "Register for free Public API credentials to get a Client ID and Client Secret. Use OAuth 2.0 client credentials flow to obtain a /read-public access token."
        },
        {
            "name": "Access Token",
            "description": "Include the access token in the Authorization header: 'Authorization: Bearer YOUR_ACCESS_TOKEN'"
        },
        {
            "name": "No Auth (Limited)",
            "description": "Some search endpoints can be accessed directly via browser without authentication, but with rate limits."
        }
    ]
    
    db.global_parameters = [
        Parameter(
            name="Authorization",
            type="string",
            required=False,
            description="Bearer token for authenticated requests. Format: 'Bearer ACCESS_TOKEN'",
            location="header"
        ),
        Parameter(
            name="Accept",
            type="string",
            required=False,
            description="Response format. Default is XML.",
            default="application/json",
            enum_values=["application/json", "application/xml", "application/vnd.orcid+json", "application/vnd.orcid+xml", "text/csv"]
        )
    ]
    
    db.error_codes = [
        {"code": "301", "description": "Moved Permanently - The ORCID iD has been deprecated"},
        {"code": "400", "description": "Bad Request - Invalid query syntax or parameters"},
        {"code": "401", "description": "Unauthorized - Missing or invalid access token"},
        {"code": "403", "description": "Forbidden - Insufficient permissions to access resource"},
        {"code": "404", "description": "Not Found - The ORCID record does not exist"},
        {"code": "409", "description": "Conflict - The record has been deactivated"},
        {"code": "429", "description": "Too Many Requests - Rate limit exceeded"},
        {"code": "500", "description": "Internal Server Error"},
        {"code": "503", "description": "Service Unavailable - Server maintenance"}
    ]
    
    
    
    db.endpoints.append(APIEndpoint(
        action="search",
        name="Search ORCID Registry",
        description="Search for ORCID records using Solr/Lucene query syntax. Returns matching ORCID iDs. Use this to find researchers by name, affiliation, works, keywords, etc.",
        http_method="GET",
        url_pattern="/v3.0/search/",
        requires_auth=False,
        category="search",
        documentation_url="https://info.orcid.org/documentation/api-tutorials/api-tutorial-searching-the-orcid-registry/",
        parameters=[
            Parameter(
                name="q",
                type="string",
                required=True,
                description="Search query using Solr/Lucene syntax. Supports field:value pairs and Boolean operators (AND, OR).",
                example="family-name:Einstein AND given-names:Albert"
            ),
            Parameter(
                name="start",
                type="integer",
                required=False,
                description="Starting position for pagination (0-based)",
                default="0"
            ),
            Parameter(
                name="rows",
                type="integer",
                required=False,
                description="Number of results to return (max 1000)",
                default="1000"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Search results with matching ORCID iDs",
                example={
                    "result": [
                        {
                            "orcid-identifier": {
                                "uri": "https://orcid.org/0000-0001-2345-6789",
                                "path": "0000-0001-2345-6789",
                                "host": "orcid.org"
                            }
                        }
                    ],
                    "num-found": 150
                }
            )
        ],
        examples=[
            {
                "description": "Search by family name",
                "url": "/v3.0/search/?q=family-name:Einstein"
            },
            {
                "description": "Search by name and affiliation",
                "url": "/v3.0/search/?q=family-name:Smith+AND+affiliation-org-name:Harvard"
            },
            {
                "description": "Search by DOI",
                "url": "/v3.0/search/?q=doi-self:10.1038/nature12373"
            },
            {
                "description": "Search with pagination",
                "url": "/v3.0/search/?q=keyword:machine+learning&start=0&rows=100"
            }
        ],
        notes="""Searchable fields include:
- orcid: ORCID iD
- given-names, family-name, credit-name, other-names: Name fields
- email: Public email addresses
- keyword: Keywords/interests
- affiliation-org-name: Organization names
- ringgold-org-id, grid-org-id, ror-org-id: Organization identifiers
- doi-self, pmid, isbn: Work identifiers
- text: Full-text search across all fields
- profile-last-modified-date: Filter by modification date

Public API is limited to 10,000 results."""
    ))
    
    db.endpoints.append(APIEndpoint(
        action="expanded-search",
        name="Expanded Search",
        description="Search with expanded results including name, email, and institution details directly in the response (no need for additional API calls).",
        http_method="GET",
        url_pattern="/v3.0/expanded-search/",
        requires_auth=False,
        category="search",
        documentation_url="https://info.orcid.org/documentation/api-tutorials/api-tutorial-searching-the-orcid-registry/",
        parameters=[
            Parameter(
                name="q",
                type="string",
                required=True,
                description="Search query using Solr/Lucene syntax",
                example="affiliation-org-name:MIT"
            ),
            Parameter(
                name="start",
                type="integer",
                required=False,
                description="Starting position for pagination",
                default="0"
            ),
            Parameter(
                name="rows",
                type="integer",
                required=False,
                description="Number of results to return",
                default="100"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Expanded search results with researcher details",
                example={
                    "expanded-result": [
                        {
                            "orcid-id": "0000-0001-2345-6789",
                            "given-names": "Albert",
                            "family-names": "Einstein",
                            "credit-name": "A. Einstein",
                            "other-name": ["Al Einstein"],
                            "email": [],
                            "institution-name": ["Princeton University", "ETH Zurich"]
                        }
                    ],
                    "num-found": 1
                }
            )
        ],
        examples=[
            {
                "description": "Search researchers at an organization",
                "url": "/v3.0/expanded-search/?q=affiliation-org-name:Stanford"
            },
            {
                "description": "Search by GRID ID",
                "url": "/v3.0/expanded-search/?q=grid-org-id:grid.5509.9"
            }
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="csv-search",
        name="CSV Search",
        description="Search and return results in CSV format. Useful for bulk data export and spreadsheet integration.",
        http_method="GET",
        url_pattern="/v3.0/csv-search/",
        requires_auth=False,
        category="search",
        documentation_url="https://info.orcid.org/documentation/api-tutorials/api-tutorial-searching-the-orcid-registry/",
        parameters=[
            Parameter(
                name="q",
                type="string",
                required=True,
                description="Search query",
                example="affiliation-org-name:ORCID"
            ),
            Parameter(
                name="fl",
                type="string",
                required=False,
                description="Comma-separated list of fields to include in output",
                example="orcid,given-names,family-name,current-institution-affiliation-name"
            ),
            Parameter(
                name="start",
                type="integer",
                required=False,
                description="Starting position",
                default="0"
            ),
            Parameter(
                name="rows",
                type="integer",
                required=False,
                description="Number of results",
                default="100"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="CSV formatted search results"
            )
        ],
        examples=[
            {
                "description": "Search with specific output fields",
                "url": "/v3.0/csv-search/?q=ringgold-org-id:385488&fl=orcid,given-names,family-name,current-institution-affiliation-name"
            }
        ],
        notes="""Available CSV fields:
- orcid
- email
- given-name
- family-name
- given-and-family-names
- credit-name
- other-name
- current-institution-affiliation-name
- past-institution-affiliation-name"""
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="record",
        name="Read Full Record",
        description="Retrieve the complete public ORCID record for a researcher, including all sections (biography, works, employment, education, etc.).",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/record",
        requires_auth=True,
        category="read",
        documentation_url="https://info.orcid.org/documentation/api-tutorials/api-tutorial-read-data-on-a-record/",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD in format 0000-0001-2345-6789",
                location="path",
                example="0000-0001-2345-6789"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Complete ORCID record",
                example={
                    "orcid-identifier": {
                        "uri": "https://orcid.org/0000-0001-2345-6789",
                        "path": "0000-0001-2345-6789"
                    },
                    "person": {
                        "name": {
                            "given-names": {"value": "Albert"},
                            "family-name": {"value": "Einstein"}
                        },
                        "biography": {"content": "Theoretical physicist..."}
                    },
                    "activities-summary": {
                        "works": {"group": []},
                        "employments": {"affiliation-group": []},
                        "educations": {"affiliation-group": []}
                    }
                }
            ),
            Response(
                status_code=404,
                description="ORCID record not found",
                example={
                    "response-code": 404,
                    "developer-message": "404 Not Found: No ORCID record found for 0000-0001-2345-6789"
                }
            )
        ],
        examples=[
            {
                "description": "Get full record",
                "url": "/v3.0/0000-0001-2345-6789/record"
            }
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="person",
        name="Read Person Details",
        description="Retrieve personal information including name, biography, keywords, researcher URLs, and external identifiers.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/person",
        requires_auth=True,
        category="read",
        documentation_url="https://info.orcid.org/documentation/api-tutorials/api-tutorial-read-data-on-a-record/",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path",
                example="0000-0001-2345-6789"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Person details",
                example={
                    "name": {
                        "given-names": {"value": "Albert"},
                        "family-name": {"value": "Einstein"},
                        "credit-name": {"value": "A. Einstein"}
                    },
                    "biography": {
                        "content": "Theoretical physicist known for the theory of relativity."
                    },
                    "keywords": {
                        "keyword": [
                            {"content": "physics"},
                            {"content": "relativity"}
                        ]
                    },
                    "researcher-urls": {
                        "researcher-url": [
                            {
                                "url-name": "Personal Website",
                                "url": {"value": "https://example.com"}
                            }
                        ]
                    }
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="works",
        name="Read Works/Publications",
        description="Retrieve the list of works (publications, datasets, etc.) associated with an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/works",
        requires_auth=True,
        category="read",
        documentation_url="https://info.orcid.org/documentation/api-tutorials/api-tutorial-read-data-on-a-record/",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path",
                example="0000-0001-2345-6789"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="List of works grouped by identifier",
                example={
                    "group": [
                        {
                            "work-summary": [
                                {
                                    "put-code": 12345,
                                    "title": {"title": {"value": "On the Electrodynamics of Moving Bodies"}},
                                    "type": "journal-article",
                                    "publication-date": {
                                        "year": {"value": "1905"},
                                        "month": {"value": "06"}
                                    },
                                    "external-ids": {
                                        "external-id": [
                                            {
                                                "external-id-type": "doi",
                                                "external-id-value": "10.1002/andp.19053221004"
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    ],
                    "path": "/0000-0001-2345-6789/works"
                }
            )
        ],
        examples=[
            {
                "description": "Get all works for a researcher",
                "url": "/v3.0/0000-0001-2345-6789/works"
            }
        ],
        notes="Works are grouped by identifier (DOI, PMID, etc.). Each group may contain multiple entries from different sources."
    ))
    
    db.endpoints.append(APIEndpoint(
        action="work",
        name="Read Single Work",
        description="Retrieve detailed information about a specific work using its put-code.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/work/{put-code}",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            ),
            Parameter(
                name="put-code",
                type="integer",
                required=True,
                description="The unique identifier for the work item",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Detailed work information",
                example={
                    "put-code": 12345,
                    "title": {
                        "title": {"value": "On the Electrodynamics of Moving Bodies"},
                        "subtitle": None
                    },
                    "journal-title": {"value": "Annalen der Physik"},
                    "type": "journal-article",
                    "publication-date": {"year": {"value": "1905"}},
                    "external-ids": {
                        "external-id": [
                            {"external-id-type": "doi", "external-id-value": "10.1002/andp.19053221004"}
                        ]
                    },
                    "contributors": {
                        "contributor": [
                            {"contributor-name": {"value": "Albert Einstein"}}
                        ]
                    }
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="employments",
        name="Read Employment History",
        description="Retrieve employment/work history from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/employments",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Employment history",
                example={
                    "affiliation-group": [
                        {
                            "summaries": [
                                {
                                    "employment-summary": {
                                        "put-code": 54321,
                                        "department-name": "Physics Department",
                                        "role-title": "Professor",
                                        "start-date": {"year": {"value": "1914"}},
                                        "end-date": {"year": {"value": "1933"}},
                                        "organization": {
                                            "name": "Humboldt University of Berlin",
                                            "address": {"city": "Berlin", "country": "DE"}
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="educations",
        name="Read Education History",
        description="Retrieve education/academic qualifications from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/educations",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Education history",
                example={
                    "affiliation-group": [
                        {
                            "summaries": [
                                {
                                    "education-summary": {
                                        "department-name": "Physics",
                                        "role-title": "Ph.D.",
                                        "start-date": {"year": {"value": "1900"}},
                                        "end-date": {"year": {"value": "1905"}},
                                        "organization": {
                                            "name": "University of Zurich",
                                            "address": {"city": "Zurich", "country": "CH"}
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="fundings",
        name="Read Funding/Grants",
        description="Retrieve funding and grants information from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/fundings",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Funding information",
                example={
                    "group": [
                        {
                            "funding-summary": [
                                {
                                    "put-code": 11111,
                                    "title": {"title": {"value": "Research Grant for Relativity Studies"}},
                                    "type": "grant",
                                    "start-date": {"year": {"value": "1910"}},
                                    "organization": {
                                        "name": "Swiss National Science Foundation"
                                    }
                                }
                            ]
                        }
                    ]
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="peer-reviews",
        name="Read Peer Reviews",
        description="Retrieve peer review activities from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/peer-reviews",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Peer review activities",
                example={
                    "group": [
                        {
                            "peer-review-group": [
                                {
                                    "peer-review-summary": [
                                        {
                                            "reviewer-role": "reviewer",
                                            "review-type": "review",
                                            "completion-date": {"year": {"value": "2023"}},
                                            "convening-organization": {
                                                "name": "Nature Publishing Group"
                                            }
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="distinctions",
        name="Read Distinctions/Awards",
        description="Retrieve distinctions, honors, and awards from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/distinctions",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Distinctions and awards",
                example={
                    "affiliation-group": [
                        {
                            "summaries": [
                                {
                                    "distinction-summary": {
                                        "role-title": "Nobel Prize in Physics",
                                        "start-date": {"year": {"value": "1921"}},
                                        "organization": {
                                            "name": "Royal Swedish Academy of Sciences"
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="memberships",
        name="Read Memberships",
        description="Retrieve professional memberships and society affiliations from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/memberships",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Professional memberships"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="services",
        name="Read Services",
        description="Retrieve professional service activities (editorial boards, committees, etc.) from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/services",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Service activities"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="qualifications",
        name="Read Qualifications",
        description="Retrieve professional qualifications and certifications from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/qualifications",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Qualifications and certifications"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="invited-positions",
        name="Read Invited Positions",
        description="Retrieve invited positions (visiting professorships, etc.) from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/invited-positions",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Invited positions"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="research-resources",
        name="Read Research Resources",
        description="Retrieve research resources (equipment, facilities, etc.) from an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/research-resources",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Research resources"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="external-identifiers",
        name="Read External Identifiers",
        description="Retrieve external identifiers (Scopus ID, ResearcherID, etc.) linked to an ORCID record.",
        http_method="GET",
        url_pattern="/v3.0/{orcid-id}/external-identifiers",
        requires_auth=True,
        category="read",
        parameters=[
            Parameter(
                name="orcid-id",
                type="string",
                required=True,
                description="The ORCID iD",
                location="path"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="External identifiers",
                example={
                    "external-identifier": [
                        {
                            "external-id-type": "Scopus Author ID",
                            "external-id-value": "12345678",
                            "external-id-url": {"value": "https://www.scopus.com/authid/detail.uri?authorId=12345678"}
                        },
                        {
                            "external-id-type": "ResearcherID",
                            "external-id-value": "A-1234-5678"
                        }
                    ]
                }
            )
        ]
    ))
    
    return db


def main():
    """主函数：生成数据库文件"""
    print("Building ORCID Public API Database...")
    
    db = build_orcid_api_database()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    json_path = os.path.join(output_dir, "orcid_api.json")
    db.save(json_path)
    print(f"✓ Saved JSON database to: {json_path}")
    
    md_path = os.path.join(output_dir, "orcid_api_llm.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(db.generate_llm_context())
    print(f"✓ Saved LLM documentation to: {md_path}")
    
    summary_path = os.path.join(output_dir, "orcid_api_summary.md")
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
