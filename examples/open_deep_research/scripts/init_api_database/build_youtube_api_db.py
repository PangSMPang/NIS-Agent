"""
YouTube Data API v3 Database Builder
构建YouTube Data API的结构化数据库

运行此脚本将生成:
1. youtube_api.json - 完整的API数据库
2. youtube_api_llm.md - LLM友好的Markdown文档
"""

from mediawiki_api_schema import (
    APIDatabase, APIEndpoint, Parameter, Response, ResponseField
)
import json
import os


def build_youtube_api_database() -> APIDatabase:
    """构建YouTube Data API数据库"""
    
    db = APIDatabase(
        name="YouTube Data API v3",
        version="v3",
        base_url="https://www.googleapis.com/youtube/v3",
        description="""YouTube Data API v3 lets you incorporate YouTube functionality into your own application.
You can use the API to fetch search results, manage playlists, upload videos, get comments, and more.

Authentication:
- API Key: For public data access (read-only)
- OAuth 2.0: For user-specific data and write operations

Base URL: https://www.googleapis.com/youtube/v3

Documentation: https://developers.google.com/youtube/v3/docs"""
    )
    
    db.auth_methods = [
        {
            "name": "API Key",
            "description": "Use the 'key' parameter for public data access. Get your API key from Google Cloud Console."
        },
        {
            "name": "OAuth 2.0",
            "description": "Required for operations that access private user data or modify data. Use 'access_token' parameter or 'Authorization: Bearer' header."
        }
    ]
    
    db.global_parameters = [
        Parameter(
            name="key",
            type="string",
            required=False,
            description="API key from Google Cloud Console. Required if not using OAuth 2.0."
        ),
        Parameter(
            name="access_token",
            type="string",
            required=False,
            description="OAuth 2.0 access token for authenticated requests."
        ),
        Parameter(
            name="part",
            type="string",
            required=True,
            description="Comma-separated list of resource parts to include in the response. Required for most endpoints."
        ),
        Parameter(
            name="fields",
            type="string",
            required=False,
            description="Selector specifying which fields to include in a partial response."
        ),
        Parameter(
            name="prettyPrint",
            type="boolean",
            required=False,
            description="Returns response with indentations and line breaks.",
            default="true"
        ),
        Parameter(
            name="quotaUser",
            type="string",
            required=False,
            description="Lets you enforce per-user quotas."
        ),
        Parameter(
            name="callback",
            type="string",
            required=False,
            description="JSONP callback function name."
        )
    ]
    
    db.error_codes = [
        {"code": "400", "description": "Bad Request - Invalid parameter value or missing required parameter"},
        {"code": "401", "description": "Unauthorized - Invalid or missing authentication credentials"},
        {"code": "403", "description": "Forbidden - Access denied, quota exceeded, or API not enabled"},
        {"code": "404", "description": "Not Found - The requested resource could not be found"},
        {"code": "409", "description": "Conflict - Request cannot be completed due to a conflict"},
        {"code": "500", "description": "Internal Server Error - Unexpected server error"},
        {"code": "quotaExceeded", "description": "The request cannot be completed because you have exceeded your quota"},
        {"code": "rateLimitExceeded", "description": "Too many requests in a given time period"},
        {"code": "dailyLimitExceeded", "description": "Daily limit exceeded"},
        {"code": "videoNotFound", "description": "The video identified by the videoId parameter could not be found"},
        {"code": "channelNotFound", "description": "The channel identified by the channelId parameter could not be found"},
        {"code": "commentNotFound", "description": "The comment identified by the id parameter could not be found"},
        {"code": "forbidden", "description": "The request is not properly authorized"},
        {"code": "invalidChannelId", "description": "The channelId parameter specified an invalid channel ID"},
        {"code": "invalidVideoId", "description": "The videoId parameter specified an invalid video ID"}
    ]
    
    
    
    db.endpoints.append(APIEndpoint(
        action="videos.list",
        name="List Videos",
        description="Returns a list of videos that match the API request parameters. Can retrieve video details by ID, or get popular videos.",
        http_method="GET",
        url_pattern="/youtube/v3/videos",
        requires_auth=False,
        category="videos",
        documentation_url="https://developers.google.com/youtube/v3/docs/videos/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Comma-separated list of video resource parts: snippet, contentDetails, statistics, status, player, topicDetails, recordingDetails, fileDetails, processingDetails, suggestions, liveStreamingDetails, localizations",
                example="snippet,contentDetails,statistics"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated list of video IDs",
                example="dQw4w9WgXcQ,jNQXAC9IVRw"
            ),
            Parameter(
                name="chart",
                type="enum",
                required=False,
                description="Return videos in specified chart",
                enum_values=["mostPopular"]
            ),
            Parameter(
                name="myRating",
                type="enum",
                required=False,
                description="Return videos rated by authenticated user (requires OAuth)",
                enum_values=["like", "dislike"]
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum number of items to return (0-50)",
                default="5"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            ),
            Parameter(
                name="regionCode",
                type="string",
                required=False,
                description="ISO 3166-1 alpha-2 country code",
                example="US"
            ),
            Parameter(
                name="videoCategoryId",
                type="string",
                required=False,
                description="Filter by video category ID"
            ),
            Parameter(
                name="hl",
                type="string",
                required=False,
                description="Language for localized resource properties",
                example="en"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with video list",
                example={
                    "kind": "youtube#videoListResponse",
                    "etag": "abc123",
                    "pageInfo": {
                        "totalResults": 1,
                        "resultsPerPage": 1
                    },
                    "items": [
                        {
                            "kind": "youtube#video",
                            "etag": "def456",
                            "id": "dQw4w9WgXcQ",
                            "snippet": {
                                "publishedAt": "2009-10-25T06:57:33Z",
                                "channelId": "UCuAXFkgsw1L7xaCfnd5JJOw",
                                "title": "Rick Astley - Never Gonna Give You Up",
                                "description": "The official video...",
                                "thumbnails": {
                                    "default": {"url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg"}
                                },
                                "channelTitle": "Rick Astley",
                                "categoryId": "10"
                            },
                            "contentDetails": {
                                "duration": "PT3M33S",
                                "dimension": "2d",
                                "definition": "hd"
                            },
                            "statistics": {
                                "viewCount": "1500000000",
                                "likeCount": "15000000",
                                "commentCount": "3000000"
                            }
                        }
                    ]
                }
            )
        ],
        examples=[
            {
                "description": "Get video details by ID",
                "url": "/youtube/v3/videos?part=snippet,statistics&id=dQw4w9WgXcQ&key=YOUR_API_KEY"
            },
            {
                "description": "Get popular videos in US",
                "url": "/youtube/v3/videos?part=snippet&chart=mostPopular&regionCode=US&maxResults=10&key=YOUR_API_KEY"
            }
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="videos.insert",
        name="Upload Video",
        description="Uploads a video to YouTube.",
        http_method="POST",
        url_pattern="/youtube/v3/videos",
        requires_auth=True,
        required_rights=["upload"],
        category="videos",
        documentation_url="https://developers.google.com/youtube/v3/docs/videos/insert",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to set: snippet, status, recordingDetails",
                example="snippet,status"
            ),
            Parameter(
                name="notifySubscribers",
                type="boolean",
                required=False,
                description="Whether to notify subscribers",
                default="true"
            ),
            Parameter(
                name="autoLevels",
                type="boolean",
                required=False,
                description="Auto-adjust video brightness and color"
            ),
            Parameter(
                name="stabilize",
                type="boolean",
                required=False,
                description="Apply video stabilization"
            )
        ],
        notes="Request body should include video metadata and the video file as multipart upload."
    ))
    
    db.endpoints.append(APIEndpoint(
        action="videos.update",
        name="Update Video",
        description="Updates a video's metadata.",
        http_method="PUT",
        url_pattern="/youtube/v3/videos",
        requires_auth=True,
        category="videos",
        documentation_url="https://developers.google.com/youtube/v3/docs/videos/update",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to update",
                example="snippet,status"
            )
        ],
        notes="Request body must include video ID and the parts to update."
    ))
    
    db.endpoints.append(APIEndpoint(
        action="videos.delete",
        name="Delete Video",
        description="Deletes a YouTube video.",
        http_method="DELETE",
        url_pattern="/youtube/v3/videos",
        requires_auth=True,
        category="videos",
        documentation_url="https://developers.google.com/youtube/v3/docs/videos/delete",
        parameters=[
            Parameter(
                name="id",
                type="string",
                required=True,
                description="Video ID to delete"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="videos.rate",
        name="Rate Video",
        description="Add a like or dislike rating to a video or remove a rating.",
        http_method="POST",
        url_pattern="/youtube/v3/videos/rate",
        requires_auth=True,
        category="videos",
        documentation_url="https://developers.google.com/youtube/v3/docs/videos/rate",
        parameters=[
            Parameter(
                name="id",
                type="string",
                required=True,
                description="Video ID to rate"
            ),
            Parameter(
                name="rating",
                type="enum",
                required=True,
                description="Rating to apply",
                enum_values=["like", "dislike", "none"]
            )
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="search.list",
        name="Search",
        description="Returns a collection of search results that match the query parameters. Search for videos, channels, and playlists.",
        http_method="GET",
        url_pattern="/youtube/v3/search",
        requires_auth=False,
        category="search",
        documentation_url="https://developers.google.com/youtube/v3/docs/search/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Must be 'snippet'",
                default="snippet"
            ),
            Parameter(
                name="q",
                type="string",
                required=False,
                description="Search query term",
                example="python tutorial"
            ),
            Parameter(
                name="type",
                type="enum",
                required=False,
                description="Restrict to specific resource type",
                enum_values=["video", "channel", "playlist"]
            ),
            Parameter(
                name="channelId",
                type="string",
                required=False,
                description="Search within a specific channel"
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum results (0-50)",
                default="5"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            ),
            Parameter(
                name="order",
                type="enum",
                required=False,
                description="Sort order for results",
                default="relevance",
                enum_values=["date", "rating", "relevance", "title", "videoCount", "viewCount"]
            ),
            Parameter(
                name="publishedAfter",
                type="timestamp",
                required=False,
                description="Filter by publish date (RFC 3339)",
                example="2024-01-01T00:00:00Z"
            ),
            Parameter(
                name="publishedBefore",
                type="timestamp",
                required=False,
                description="Filter by publish date (RFC 3339)"
            ),
            Parameter(
                name="regionCode",
                type="string",
                required=False,
                description="ISO 3166-1 alpha-2 country code",
                example="US"
            ),
            Parameter(
                name="relevanceLanguage",
                type="string",
                required=False,
                description="Prefer results in this language",
                example="en"
            ),
            Parameter(
                name="safeSearch",
                type="enum",
                required=False,
                description="Safe search filter",
                enum_values=["moderate", "none", "strict"]
            ),
            Parameter(
                name="videoDuration",
                type="enum",
                required=False,
                description="Filter by video duration",
                enum_values=["any", "long", "medium", "short"]
            ),
            Parameter(
                name="videoDefinition",
                type="enum",
                required=False,
                description="Filter by video definition",
                enum_values=["any", "high", "standard"]
            ),
            Parameter(
                name="videoType",
                type="enum",
                required=False,
                description="Filter by video type",
                enum_values=["any", "episode", "movie"]
            ),
            Parameter(
                name="videoCaption",
                type="enum",
                required=False,
                description="Filter by caption availability",
                enum_values=["any", "closedCaption", "none"]
            ),
            Parameter(
                name="videoCategoryId",
                type="string",
                required=False,
                description="Filter by category ID"
            ),
            Parameter(
                name="videoLicense",
                type="enum",
                required=False,
                description="Filter by video license",
                enum_values=["any", "creativeCommon", "youtube"]
            ),
            Parameter(
                name="eventType",
                type="enum",
                required=False,
                description="Filter by live event type",
                enum_values=["completed", "live", "upcoming"]
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful search results",
                example={
                    "kind": "youtube#searchListResponse",
                    "etag": "abc123",
                    "nextPageToken": "CAUQAA",
                    "pageInfo": {
                        "totalResults": 1000000,
                        "resultsPerPage": 5
                    },
                    "items": [
                        {
                            "kind": "youtube#searchResult",
                            "id": {
                                "kind": "youtube#video",
                                "videoId": "abc123xyz"
                            },
                            "snippet": {
                                "publishedAt": "2024-01-15T10:00:00Z",
                                "channelId": "UCxxx",
                                "title": "Python Tutorial for Beginners",
                                "description": "Learn Python programming...",
                                "thumbnails": {
                                    "default": {"url": "https://i.ytimg.com/vi/abc123xyz/default.jpg"}
                                },
                                "channelTitle": "Coding Channel",
                                "liveBroadcastContent": "none"
                            }
                        }
                    ]
                }
            )
        ],
        examples=[
            {
                "description": "Search for videos",
                "url": "/youtube/v3/search?part=snippet&q=python%20tutorial&type=video&maxResults=10&key=YOUR_API_KEY"
            },
            {
                "description": "Search for channels",
                "url": "/youtube/v3/search?part=snippet&q=coding&type=channel&key=YOUR_API_KEY"
            },
            {
                "description": "Search with filters",
                "url": "/youtube/v3/search?part=snippet&q=music&type=video&videoDuration=short&order=viewCount&key=YOUR_API_KEY"
            }
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="commentThreads.list",
        name="List Comment Threads",
        description="Returns a list of comment threads that match the API request parameters. Use this to get top-level comments on a video or channel.",
        http_method="GET",
        url_pattern="/youtube/v3/commentThreads",
        requires_auth=False,
        category="comments",
        documentation_url="https://developers.google.com/youtube/v3/docs/commentThreads/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to include: id, snippet, replies",
                example="snippet,replies"
            ),
            Parameter(
                name="videoId",
                type="string",
                required=False,
                description="Return comments for this video ID",
                example="dQw4w9WgXcQ"
            ),
            Parameter(
                name="channelId",
                type="string",
                required=False,
                description="Return comments for this channel ID"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated list of comment thread IDs"
            ),
            Parameter(
                name="allThreadsRelatedToChannelId",
                type="string",
                required=False,
                description="Return all comments associated with this channel (requires OAuth)"
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum results (1-100)",
                default="20"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            ),
            Parameter(
                name="order",
                type="enum",
                required=False,
                description="Sort order",
                default="time",
                enum_values=["time", "relevance"]
            ),
            Parameter(
                name="searchTerms",
                type="string",
                required=False,
                description="Filter comments containing these terms"
            ),
            Parameter(
                name="moderationStatus",
                type="enum",
                required=False,
                description="Filter by moderation status (requires OAuth)",
                enum_values=["heldForReview", "likelySpam", "published"]
            ),
            Parameter(
                name="textFormat",
                type="enum",
                required=False,
                description="Text format for comment text",
                default="html",
                enum_values=["html", "plainText"]
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with comment threads",
                example={
                    "kind": "youtube#commentThreadListResponse",
                    "etag": "abc123",
                    "nextPageToken": "QURTSl9pMU...",
                    "pageInfo": {
                        "totalResults": 5000,
                        "resultsPerPage": 20
                    },
                    "items": [
                        {
                            "kind": "youtube#commentThread",
                            "id": "UgzYx...",
                            "snippet": {
                                "videoId": "dQw4w9WgXcQ",
                                "topLevelComment": {
                                    "kind": "youtube#comment",
                                    "id": "UgzYx...",
                                    "snippet": {
                                        "videoId": "dQw4w9WgXcQ",
                                        "textDisplay": "Great video!",
                                        "textOriginal": "Great video!",
                                        "authorDisplayName": "User Name",
                                        "authorProfileImageUrl": "https://...",
                                        "authorChannelUrl": "https://...",
                                        "authorChannelId": {"value": "UCxxx"},
                                        "likeCount": 100,
                                        "publishedAt": "2024-01-15T10:00:00Z",
                                        "updatedAt": "2024-01-15T10:00:00Z"
                                    }
                                },
                                "canReply": True,
                                "totalReplyCount": 5,
                                "isPublic": True
                            },
                            "replies": {
                                "comments": []
                            }
                        }
                    ]
                }
            )
        ],
        examples=[
            {
                "description": "Get comments for a video",
                "url": "/youtube/v3/commentThreads?part=snippet,replies&videoId=dQw4w9WgXcQ&maxResults=20&key=YOUR_API_KEY"
            },
            {
                "description": "Search comments",
                "url": "/youtube/v3/commentThreads?part=snippet&videoId=VIDEO_ID&searchTerms=great&key=YOUR_API_KEY"
            }
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="commentThreads.insert",
        name="Create Comment Thread",
        description="Creates a new top-level comment on a video or channel.",
        http_method="POST",
        url_pattern="/youtube/v3/commentThreads",
        requires_auth=True,
        category="comments",
        documentation_url="https://developers.google.com/youtube/v3/docs/commentThreads/insert",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Must include 'snippet'",
                default="snippet"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Comment created successfully",
                example={
                    "kind": "youtube#commentThread",
                    "id": "UgzYx...",
                    "snippet": {
                        "videoId": "VIDEO_ID",
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": "My comment text",
                                "authorDisplayName": "My Name"
                            }
                        }
                    }
                }
            )
        ],
        notes="Request body must include snippet.videoId (or channelId) and snippet.topLevelComment.snippet.textOriginal"
    ))
    
    db.endpoints.append(APIEndpoint(
        action="comments.list",
        name="List Comments",
        description="Returns a list of comments that match the API request parameters. Use this to get replies to a comment.",
        http_method="GET",
        url_pattern="/youtube/v3/comments",
        requires_auth=False,
        category="comments",
        documentation_url="https://developers.google.com/youtube/v3/docs/comments/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to include: id, snippet",
                example="snippet"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated list of comment IDs"
            ),
            Parameter(
                name="parentId",
                type="string",
                required=False,
                description="Return replies to this comment ID"
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum results (1-100)",
                default="20"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            ),
            Parameter(
                name="textFormat",
                type="enum",
                required=False,
                description="Text format",
                default="html",
                enum_values=["html", "plainText"]
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with comments",
                example={
                    "kind": "youtube#commentListResponse",
                    "items": [
                        {
                            "kind": "youtube#comment",
                            "id": "UgzYx...",
                            "snippet": {
                                "textDisplay": "Reply text",
                                "authorDisplayName": "User",
                                "likeCount": 5,
                                "publishedAt": "2024-01-15T10:00:00Z"
                            }
                        }
                    ]
                }
            )
        ],
        examples=[
            {
                "description": "Get replies to a comment",
                "url": "/youtube/v3/comments?part=snippet&parentId=COMMENT_ID&key=YOUR_API_KEY"
            }
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="comments.insert",
        name="Reply to Comment",
        description="Creates a reply to an existing comment.",
        http_method="POST",
        url_pattern="/youtube/v3/comments",
        requires_auth=True,
        category="comments",
        documentation_url="https://developers.google.com/youtube/v3/docs/comments/insert",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Must include 'snippet'",
                default="snippet"
            )
        ],
        notes="Request body must include snippet.parentId and snippet.textOriginal"
    ))
    
    db.endpoints.append(APIEndpoint(
        action="comments.update",
        name="Update Comment",
        description="Modifies a comment.",
        http_method="PUT",
        url_pattern="/youtube/v3/comments",
        requires_auth=True,
        category="comments",
        documentation_url="https://developers.google.com/youtube/v3/docs/comments/update",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Must include 'snippet'",
                default="snippet"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="comments.delete",
        name="Delete Comment",
        description="Deletes a comment.",
        http_method="DELETE",
        url_pattern="/youtube/v3/comments",
        requires_auth=True,
        category="comments",
        documentation_url="https://developers.google.com/youtube/v3/docs/comments/delete",
        parameters=[
            Parameter(
                name="id",
                type="string",
                required=True,
                description="Comment ID to delete"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="comments.setModerationStatus",
        name="Set Comment Moderation Status",
        description="Sets the moderation status of one or more comments.",
        http_method="POST",
        url_pattern="/youtube/v3/comments/setModerationStatus",
        requires_auth=True,
        category="comments",
        documentation_url="https://developers.google.com/youtube/v3/docs/comments/setModerationStatus",
        parameters=[
            Parameter(
                name="id",
                type="string",
                required=True,
                description="Comma-separated list of comment IDs"
            ),
            Parameter(
                name="moderationStatus",
                type="enum",
                required=True,
                description="New moderation status",
                enum_values=["heldForReview", "published", "rejected"]
            ),
            Parameter(
                name="banAuthor",
                type="boolean",
                required=False,
                description="Ban the comment author from making future comments",
                default="false"
            )
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="channels.list",
        name="List Channels",
        description="Returns a collection of channel resources that match the request criteria.",
        http_method="GET",
        url_pattern="/youtube/v3/channels",
        requires_auth=False,
        category="channels",
        documentation_url="https://developers.google.com/youtube/v3/docs/channels/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to include: snippet, contentDetails, statistics, topicDetails, status, brandingSettings, contentOwnerDetails, localizations",
                example="snippet,statistics"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated list of channel IDs"
            ),
            Parameter(
                name="forUsername",
                type="string",
                required=False,
                description="Return channel for this username"
            ),
            Parameter(
                name="forHandle",
                type="string",
                required=False,
                description="Return channel for this handle (@username)",
                example="@Google"
            ),
            Parameter(
                name="mine",
                type="boolean",
                required=False,
                description="Return authenticated user's channel (requires OAuth)"
            ),
            Parameter(
                name="managedByMe",
                type="boolean",
                required=False,
                description="Return channels managed by user (requires OAuth)"
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum results (0-50)",
                default="5"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            ),
            Parameter(
                name="hl",
                type="string",
                required=False,
                description="Language for localized properties",
                example="en"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with channel data",
                example={
                    "kind": "youtube#channelListResponse",
                    "items": [
                        {
                            "kind": "youtube#channel",
                            "id": "UCxxx",
                            "snippet": {
                                "title": "Channel Name",
                                "description": "Channel description...",
                                "customUrl": "@channelname",
                                "publishedAt": "2010-01-01T00:00:00Z",
                                "thumbnails": {
                                    "default": {"url": "https://..."}
                                },
                                "country": "US"
                            },
                            "statistics": {
                                "viewCount": "1000000000",
                                "subscriberCount": "10000000",
                                "hiddenSubscriberCount": False,
                                "videoCount": "500"
                            }
                        }
                    ]
                }
            )
        ],
        examples=[
            {
                "description": "Get channel by ID",
                "url": "/youtube/v3/channels?part=snippet,statistics&id=UC_x5XG1OV2P6uZZ5FSM9Ttw&key=YOUR_API_KEY"
            },
            {
                "description": "Get channel by handle",
                "url": "/youtube/v3/channels?part=snippet,statistics&forHandle=@Google&key=YOUR_API_KEY"
            }
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="channels.update",
        name="Update Channel",
        description="Updates a channel's metadata. Only brandingSettings and invideoPromotion can be updated.",
        http_method="PUT",
        url_pattern="/youtube/v3/channels",
        requires_auth=True,
        category="channels",
        documentation_url="https://developers.google.com/youtube/v3/docs/channels/update",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to update: brandingSettings, invideoPromotion, localizations",
                example="brandingSettings"
            )
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="playlists.list",
        name="List Playlists",
        description="Returns a collection of playlists that match the request criteria.",
        http_method="GET",
        url_pattern="/youtube/v3/playlists",
        requires_auth=False,
        category="playlists",
        documentation_url="https://developers.google.com/youtube/v3/docs/playlists/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to include: snippet, status, contentDetails, player, localizations",
                example="snippet,contentDetails"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated list of playlist IDs"
            ),
            Parameter(
                name="channelId",
                type="string",
                required=False,
                description="Return playlists for this channel"
            ),
            Parameter(
                name="mine",
                type="boolean",
                required=False,
                description="Return authenticated user's playlists (requires OAuth)"
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum results (0-50)",
                default="5"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            ),
            Parameter(
                name="hl",
                type="string",
                required=False,
                description="Language for localized properties"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with playlists",
                example={
                    "kind": "youtube#playlistListResponse",
                    "items": [
                        {
                            "kind": "youtube#playlist",
                            "id": "PLxxx",
                            "snippet": {
                                "title": "My Playlist",
                                "description": "Playlist description",
                                "channelId": "UCxxx",
                                "channelTitle": "Channel Name"
                            },
                            "contentDetails": {
                                "itemCount": 25
                            }
                        }
                    ]
                }
            )
        ],
        examples=[
            {
                "description": "Get playlists for a channel",
                "url": "/youtube/v3/playlists?part=snippet,contentDetails&channelId=UC_x5XG1OV2P6uZZ5FSM9Ttw&maxResults=25&key=YOUR_API_KEY"
            }
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="playlists.insert",
        name="Create Playlist",
        description="Creates a playlist.",
        http_method="POST",
        url_pattern="/youtube/v3/playlists",
        requires_auth=True,
        category="playlists",
        documentation_url="https://developers.google.com/youtube/v3/docs/playlists/insert",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to set: snippet, status",
                example="snippet,status"
            )
        ],
        notes="Request body must include snippet.title. Optional: snippet.description, status.privacyStatus"
    ))
    
    db.endpoints.append(APIEndpoint(
        action="playlists.update",
        name="Update Playlist",
        description="Updates a playlist's metadata.",
        http_method="PUT",
        url_pattern="/youtube/v3/playlists",
        requires_auth=True,
        category="playlists",
        documentation_url="https://developers.google.com/youtube/v3/docs/playlists/update",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to update",
                example="snippet,status"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="playlists.delete",
        name="Delete Playlist",
        description="Deletes a playlist.",
        http_method="DELETE",
        url_pattern="/youtube/v3/playlists",
        requires_auth=True,
        category="playlists",
        documentation_url="https://developers.google.com/youtube/v3/docs/playlists/delete",
        parameters=[
            Parameter(
                name="id",
                type="string",
                required=True,
                description="Playlist ID to delete"
            )
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="playlistItems.list",
        name="List Playlist Items",
        description="Returns a collection of playlist items that match the request criteria.",
        http_method="GET",
        url_pattern="/youtube/v3/playlistItems",
        requires_auth=False,
        category="playlists",
        documentation_url="https://developers.google.com/youtube/v3/docs/playlistItems/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to include: snippet, contentDetails, status",
                example="snippet,contentDetails"
            ),
            Parameter(
                name="playlistId",
                type="string",
                required=False,
                description="Return items in this playlist"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated list of playlist item IDs"
            ),
            Parameter(
                name="videoId",
                type="string",
                required=False,
                description="Return item for this video in the playlist"
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum results (0-50)",
                default="5"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with playlist items",
                example={
                    "kind": "youtube#playlistItemListResponse",
                    "items": [
                        {
                            "kind": "youtube#playlistItem",
                            "id": "UExxx",
                            "snippet": {
                                "title": "Video Title",
                                "description": "Video description",
                                "position": 0,
                                "resourceId": {
                                    "kind": "youtube#video",
                                    "videoId": "dQw4w9WgXcQ"
                                }
                            },
                            "contentDetails": {
                                "videoId": "dQw4w9WgXcQ",
                                "videoPublishedAt": "2009-10-25T06:57:33Z"
                            }
                        }
                    ]
                }
            )
        ],
        examples=[
            {
                "description": "Get videos in a playlist",
                "url": "/youtube/v3/playlistItems?part=snippet,contentDetails&playlistId=PLxxx&maxResults=50&key=YOUR_API_KEY"
            }
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="playlistItems.insert",
        name="Add to Playlist",
        description="Adds a resource to a playlist.",
        http_method="POST",
        url_pattern="/youtube/v3/playlistItems",
        requires_auth=True,
        category="playlists",
        documentation_url="https://developers.google.com/youtube/v3/docs/playlistItems/insert",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Must include 'snippet'",
                default="snippet"
            )
        ],
        notes="Request body must include snippet.playlistId and snippet.resourceId"
    ))
    
    db.endpoints.append(APIEndpoint(
        action="playlistItems.update",
        name="Update Playlist Item",
        description="Updates a playlist item.",
        http_method="PUT",
        url_pattern="/youtube/v3/playlistItems",
        requires_auth=True,
        category="playlists",
        documentation_url="https://developers.google.com/youtube/v3/docs/playlistItems/update",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to update",
                example="snippet"
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="playlistItems.delete",
        name="Remove from Playlist",
        description="Deletes a playlist item.",
        http_method="DELETE",
        url_pattern="/youtube/v3/playlistItems",
        requires_auth=True,
        category="playlists",
        documentation_url="https://developers.google.com/youtube/v3/docs/playlistItems/delete",
        parameters=[
            Parameter(
                name="id",
                type="string",
                required=True,
                description="Playlist item ID to delete"
            )
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="subscriptions.list",
        name="List Subscriptions",
        description="Returns subscription resources that match the request criteria.",
        http_method="GET",
        url_pattern="/youtube/v3/subscriptions",
        requires_auth=False,
        category="subscriptions",
        documentation_url="https://developers.google.com/youtube/v3/docs/subscriptions/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to include: snippet, contentDetails, subscriberSnippet",
                example="snippet,contentDetails"
            ),
            Parameter(
                name="channelId",
                type="string",
                required=False,
                description="Return subscriptions for this channel"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated subscription IDs"
            ),
            Parameter(
                name="mine",
                type="boolean",
                required=False,
                description="Return authenticated user's subscriptions (requires OAuth)"
            ),
            Parameter(
                name="myRecentSubscribers",
                type="boolean",
                required=False,
                description="Return recent subscribers to authenticated user's channel"
            ),
            Parameter(
                name="mySubscribers",
                type="boolean",
                required=False,
                description="Return subscribers to authenticated user's channel"
            ),
            Parameter(
                name="forChannelId",
                type="string",
                required=False,
                description="Check if user is subscribed to these channel IDs"
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum results (0-50)",
                default="5"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            ),
            Parameter(
                name="order",
                type="enum",
                required=False,
                description="Sort order",
                default="relevance",
                enum_values=["alphabetical", "relevance", "unread"]
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with subscriptions",
                example={
                    "kind": "youtube#subscriptionListResponse",
                    "items": [
                        {
                            "kind": "youtube#subscription",
                            "id": "xxx",
                            "snippet": {
                                "title": "Channel Name",
                                "description": "Channel description",
                                "resourceId": {
                                    "kind": "youtube#channel",
                                    "channelId": "UCxxx"
                                },
                                "thumbnails": {}
                            }
                        }
                    ]
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="subscriptions.insert",
        name="Subscribe to Channel",
        description="Adds a subscription for the authenticated user's channel.",
        http_method="POST",
        url_pattern="/youtube/v3/subscriptions",
        requires_auth=True,
        category="subscriptions",
        documentation_url="https://developers.google.com/youtube/v3/docs/subscriptions/insert",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Must include 'snippet'",
                default="snippet"
            )
        ],
        notes="Request body must include snippet.resourceId.channelId"
    ))
    
    db.endpoints.append(APIEndpoint(
        action="subscriptions.delete",
        name="Unsubscribe from Channel",
        description="Deletes a subscription.",
        http_method="DELETE",
        url_pattern="/youtube/v3/subscriptions",
        requires_auth=True,
        category="subscriptions",
        documentation_url="https://developers.google.com/youtube/v3/docs/subscriptions/delete",
        parameters=[
            Parameter(
                name="id",
                type="string",
                required=True,
                description="Subscription ID to delete"
            )
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="captions.list",
        name="List Captions",
        description="Returns a list of caption tracks for a video.",
        http_method="GET",
        url_pattern="/youtube/v3/captions",
        requires_auth=True,
        category="captions",
        documentation_url="https://developers.google.com/youtube/v3/docs/captions/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to include: id, snippet",
                example="snippet"
            ),
            Parameter(
                name="videoId",
                type="string",
                required=True,
                description="Video ID to get captions for"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated caption track IDs"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with caption tracks",
                example={
                    "kind": "youtube#captionListResponse",
                    "items": [
                        {
                            "kind": "youtube#caption",
                            "id": "xxx",
                            "snippet": {
                                "videoId": "VIDEO_ID",
                                "lastUpdated": "2024-01-15T10:00:00Z",
                                "trackKind": "standard",
                                "language": "en",
                                "name": "English",
                                "audioTrackType": "unknown",
                                "isCC": False,
                                "isLarge": False,
                                "isEasyReader": False,
                                "isDraft": False,
                                "isAutoSynced": False,
                                "status": "serving"
                            }
                        }
                    ]
                }
            )
        ]
    ))
    
    db.endpoints.append(APIEndpoint(
        action="captions.download",
        name="Download Captions",
        description="Downloads a caption track.",
        http_method="GET",
        url_pattern="/youtube/v3/captions/{id}",
        requires_auth=True,
        category="captions",
        documentation_url="https://developers.google.com/youtube/v3/docs/captions/download",
        parameters=[
            Parameter(
                name="id",
                type="string",
                required=True,
                description="Caption track ID",
                location="path"
            ),
            Parameter(
                name="tfmt",
                type="enum",
                required=False,
                description="Output format",
                enum_values=["sbv", "scc", "srt", "ttml", "vtt"]
            ),
            Parameter(
                name="tlang",
                type="string",
                required=False,
                description="Translate to this language code"
            )
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="videoCategories.list",
        name="List Video Categories",
        description="Returns a list of video categories.",
        http_method="GET",
        url_pattern="/youtube/v3/videoCategories",
        requires_auth=False,
        category="utility",
        documentation_url="https://developers.google.com/youtube/v3/docs/videoCategories/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Must be 'snippet'",
                default="snippet"
            ),
            Parameter(
                name="regionCode",
                type="string",
                required=False,
                description="ISO 3166-1 alpha-2 country code",
                example="US"
            ),
            Parameter(
                name="id",
                type="string",
                required=False,
                description="Comma-separated category IDs"
            ),
            Parameter(
                name="hl",
                type="string",
                required=False,
                description="Language for category names",
                example="en"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with categories",
                example={
                    "kind": "youtube#videoCategoryListResponse",
                    "items": [
                        {
                            "kind": "youtube#videoCategory",
                            "id": "10",
                            "snippet": {
                                "title": "Music",
                                "assignable": True,
                                "channelId": "UCBR8-60-B28hp2BmDPdntcQ"
                            }
                        }
                    ]
                }
            )
        ],
        examples=[
            {
                "description": "Get video categories for US",
                "url": "/youtube/v3/videoCategories?part=snippet&regionCode=US&key=YOUR_API_KEY"
            }
        ]
    ))
    
    
    db.endpoints.append(APIEndpoint(
        action="activities.list",
        name="List Activities",
        description="Returns a list of channel activity events that match the request criteria.",
        http_method="GET",
        url_pattern="/youtube/v3/activities",
        requires_auth=False,
        category="activities",
        documentation_url="https://developers.google.com/youtube/v3/docs/activities/list",
        parameters=[
            Parameter(
                name="part",
                type="string",
                required=True,
                description="Parts to include: snippet, contentDetails",
                example="snippet,contentDetails"
            ),
            Parameter(
                name="channelId",
                type="string",
                required=False,
                description="Return activities for this channel"
            ),
            Parameter(
                name="mine",
                type="boolean",
                required=False,
                description="Return authenticated user's activities (requires OAuth)"
            ),
            Parameter(
                name="maxResults",
                type="integer",
                required=False,
                description="Maximum results (0-50)",
                default="5"
            ),
            Parameter(
                name="pageToken",
                type="string",
                required=False,
                description="Token for pagination"
            ),
            Parameter(
                name="publishedAfter",
                type="timestamp",
                required=False,
                description="Filter by date (RFC 3339)"
            ),
            Parameter(
                name="publishedBefore",
                type="timestamp",
                required=False,
                description="Filter by date (RFC 3339)"
            ),
            Parameter(
                name="regionCode",
                type="string",
                required=False,
                description="ISO 3166-1 alpha-2 country code"
            )
        ],
        responses=[
            Response(
                status_code=200,
                description="Successful response with activities",
                example={
                    "kind": "youtube#activityListResponse",
                    "items": [
                        {
                            "kind": "youtube#activity",
                            "id": "xxx",
                            "snippet": {
                                "type": "upload",
                                "publishedAt": "2024-01-15T10:00:00Z",
                                "channelId": "UCxxx",
                                "title": "New Video Title",
                                "description": "Video description"
                            },
                            "contentDetails": {
                                "upload": {
                                    "videoId": "VIDEO_ID"
                                }
                            }
                        }
                    ]
                }
            )
        ]
    ))
    
    return db


def main():
    """主函数：生成数据库文件"""
    print("Building YouTube Data API Database...")
    
    db = build_youtube_api_database()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    json_path = os.path.join(output_dir, "youtube_api.json")
    db.save(json_path)
    print(f"✓ Saved JSON database to: {json_path}")
    
    md_path = os.path.join(output_dir, "youtube_api_llm.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(db.generate_llm_context())
    print(f"✓ Saved LLM documentation to: {md_path}")
    
    summary_path = os.path.join(output_dir, "youtube_api_summary.md")
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
