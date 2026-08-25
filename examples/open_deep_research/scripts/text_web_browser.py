# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import json
import mimetypes
import os
import pathlib
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse
from .mdconvert import PdfConverter

import pathvalidate
import requests
from trafilatura import extract
import fitz
import base64
from pathlib import Path
from .visual_qa import compress_image_to_target_size

from smolagents import Tool

from .cookies import COOKIES
from .mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException


class SimpleTextBrowser:
    """(In preview) An extremely simple text-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
        self,
        start_page: Optional[str] = None,
        viewport_size: Optional[int] = 1024 * 8,
        downloads_folder: Optional[Union[str, None]] = None,
        request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        self.history: List[Tuple[str, float]] = list()
        self.page_title: Optional[str] = None
        self.set_address(self.start_page)
        self.request_kwargs = request_kwargs
        self.request_kwargs["cookies"] = COOKIES
        self._mdconvert = MarkdownConverter()
        self._page_content: str = ""
        
    def get_content_type(self, url: str) -> str:
        """Fetch the content type of the given URL using browser headers.
        
        Handles redirects automatically to ensure correct content type is returned.
        """
        try:
            request_kwargs = self.request_kwargs.copy() if self.request_kwargs else {}
            request_kwargs['allow_redirects'] = True
            response = requests.head(url, **request_kwargs)
            return response.headers.get("content-type", "")
        except:
            return ""



    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str, filter_year: Optional[int] = None) -> None:
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")

        else:
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)


    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content



    def visit_page(self, path_or_uri: str, filter_year: Optional[int] = None) -> str:
        """Update the address, visit the page, and return the page content."""
        self.set_address(path_or_uri, filter_year=filter_year)
        return self.page_content





    def _fetch_page(self, url: str) -> None:
        download_path = ""
        try:
            if url.startswith("file://"):
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path)
                self.page_title = res.title
                self._set_page_content(res.text_content)
            else:
                # Prepare the request parameters
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                request_kwargs["stream"] = True

                # Send a HTTP request to the URL
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                # If the HTTP request was successful
                content_type = response.headers.get("content-type", "")

                # Text or HTML
                if "text/" in content_type.lower():
                    res = self._mdconvert.convert_response(response)
                    self.page_title = res.title
                    self._set_page_content(res.text_content)
                # A download
                else:
                    # Try producing a safe filename
                    fname = None
                    download_path = None
                    try:
                        disposition_filename = None
                        content_disposition = response.headers.get('content-disposition', '')
                        if content_disposition:
                            filename_match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\r\n]+)["\']?', content_disposition, re.IGNORECASE)
                            if filename_match:
                                disposition_filename = filename_match.group(1)
                                disposition_filename = unquote(disposition_filename).strip()
                        
                        if disposition_filename:
                            base_fname = pathvalidate.sanitize_filename(disposition_filename)
                        else:
                            base_fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        if base_fname:
                            _, ext = os.path.splitext(base_fname)
                            if not ext:
                                guessed_ext = mimetypes.guess_extension(content_type)
                                if guessed_ext:
                                    fname = base_fname + guessed_ext
                                else:
                                    fname = base_fname 
                            else:
                                fname = base_fname
                        
                        if fname:
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                            suffix = 0
                            while os.path.exists(download_path) and suffix < 1000:
                                suffix += 1
                                base, ext = os.path.splitext(fname)
                                new_fname = f"{base}__{suffix}{ext}"
                                download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))

                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # Open a file for writing
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Render it
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)

        except UnsupportedFormatException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
        except requests.exceptions.RequestException as request_exception:
            try:
                self.page_title = f"Error {response.status_code}"

                # If the error was rendered in HTML we might as well render it
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content}")
                else:
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text}")
            except NameError:
                self.page_title = "Error"
                self._set_page_content(f"## Error\n\n{str(request_exception)}")

    def _state(self) -> Tuple[str, str]:
        header = f"Address: {self.address}\n"
        if self.page_title is not None:
            header += f"Title: {self.page_title}\n"

        address = self.address
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == address:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                break

        return (header, self.page_content)


class VisitTool(Tool):
    name = "visit_page"
    description = """
Visit a resource at a given URL and return relevant information.
• YouTube videos → get transcript
• Audio files (.wav, .mp3, .m4a) → download the file
• Office docs (.docx, .xlsx, .pptx) → download the file
• Images (.png, .jpg) → download the file
• Plain text files (.txt) → download the file
• ZIP files → download the file

NOT for HTML pages → use fetch_html
NOT for PDF files → use fetch_pdf
    """
    inputs = {"url": {"type": "string", "description": "URL of special files: YouTube videos, audio files (.wav/.mp3/.m4a), office docs (.docx/.xlsx/.pptx), images, or text files."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        content_type = self.browser.get_content_type(url)
        print("content_type",content_type)
        
        if url.startswith("https://www.youtube.com/watch?"):
            self.browser.visit_page(url)
        elif "text/html" in content_type:
            return ("This tool is not suitable for HTML pages. Use fetch_html for HTML pages.")
        elif "application/pdf" in content_type or url.lower().endswith('.pdf'):
            return ("This tool is not suitable for PDF files. Use fetch_pdf for PDF content extraction with query-based relevance.")
        else:
            print(f"visit_page: {url}")
            self.browser.visit_page(url)

        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content

    



import torch
import pynvml
from transformers import AutoTokenizer
from bs4 import BeautifulSoup, Comment

class FetchHtmlTool(Tool):
    name = "fetch_html"
    description = """
• ONLY for HTML webpages and websites
• Reture relevant content based on your query

NOT for PDF files → use fetch_pdf
NOT for other file types → use visit_page
    """
    inputs = {
        "url": {"type": "string", "description": "URL of an HTML webpage or website (must be text/html content)."},
        "query": {"type": "string", "description": "Your search query - what information are you looking for on this webpage?"}
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str, query: str) -> str:
        content_type = self.browser.get_content_type(url)
        print(f"content_type: {content_type}")
        if "text/html" not in content_type:
            return "This tool only supports HTML pages. Please use visit_page for PDF or TXT files."
        if url.startswith("https://www.youtube.com/watch?"):
            return "This appears to be a YouTube video page.\nPlease use visit_page to retrieve the transcript instead of fetch_html."
        
        result_html = self.get_html(url)
        chunks = self.split_content(result_html)
        
        all_responses = []
        for i, chunk in enumerate(chunks):
            response = self.analyze_content_with_llm(query, chunk)
            all_responses.append(f"Chunk {i+1} analysis: {response}")
        
        final_result = self.summarize_responses(query, all_responses)
        
        if self.is_full_js_page(url):
            final_result += "\n\nThis page is protected by JavaScript. Please use other webpages. If the website is critical, suggest to your manager that he instruct coder_agent to use the site’s API (if available) or web scraping to retrieve the content."
        
        return final_result

    def split_content(self, content: str) -> list:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        chunk_size_tokens = 40000
        overlap_tokens = 200
        max_chunks = 4
        
        tokens = tokenizer.encode(content)
        
        print(f"[DEBUG] 总token数: {len(tokens)}")
        
        if len(tokens) <= chunk_size_tokens:
            return [content]
        
        chunks = []
        start_token = 0
        
        while start_token < len(tokens) and len(chunks) < max_chunks:
            end_token = start_token + chunk_size_tokens
            
            if end_token >= len(tokens):
                chunk_tokens = tokens[start_token:]
            else:
                chunk_tokens = tokens[start_token:end_token]
            
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            print(f"[DEBUG] 创建块 {len(chunks)}，token数: {len(chunk_tokens)}")
            
            start_token = end_token - overlap_tokens
            
            if start_token >= len(tokens):
                break
        
        return chunks

    def analyze_content_with_llm(self, query: str, content: str) -> str:
        """使用大模型分析内容"""
        
        base_url = os.getenv("HTML_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("HTML_API_KEY", "")
        model_name = os.getenv("HTML_MODEL", "qwen-turbo-2025-02-11")  
        
        prompt = f"""You are an expert in analyzing content and extracting useful information from it.

## Your task
I will give you a question and some content. The content maybe include the answer of the question(maybe not include). You should thoroughly analyze the following question and provide a correct answer. If there is no direct answer but some keywords that maybe contribute to the answer, output them.

## Question
{query}

## Content
{content}

## Tips
If you made any assumptions before answering a question, clarify them in your answer.
If the answer is in the content, answer the relevate information.
if the answer is not in the content, answer "I don't know". **Do not make up any information.**
"""
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.0,
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120
            )

            # print(f"payload: {json.dumps(payload, indent=4)}")
            # print(f"response: {json.dumps(response.json(), indent=4)}")
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error calling AI model: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Failed to call AI model: {str(e)}"

    def summarize_responses(self, query: str, responses: list) -> str:
        """汇总所有模型响应"""
        useful_responses = []
        for response in responses:
            if "I don't know" not in response and "Failed to call" not in response and "Error calling" not in response:
                useful_responses.append(response)
        
        if not useful_responses:
            return f"No relevant information found in the webpage content. Plaease try other webpages or change your query."
        
        summary = f"Analysis Results:\n\n"
        for i, response in enumerate(useful_responses, 1):
            summary += f"{response}\n\n"
        
        if len(useful_responses) > 1:
            summary += "Summary: Multiple relevant pieces of information were found across different sections of the webpage."
        
        return summary.strip()

    def get_html(self, url: str) -> str:
        """Fetch the HTML content of the given URL using browser headers."""
        try:
            request_kwargs = self.browser.request_kwargs.copy() if self.browser.request_kwargs else {}
            response = requests.get(url, **request_kwargs)
            raw_html = response.text
            soup = BeautifulSoup(raw_html, 'html.parser')
            simplified_html = self.simplify_html(soup)
            simplified_html = self.clean_xml(simplified_html)
            return simplified_html
        except Exception as e:
            return f"Error: Failed to fetch content from {url} - {str(e)}"

    def simplify_html(self, soup, keep_attr: bool = False) -> str:
        for script in soup(["script", "style"]):
            script.decompose()
        if not keep_attr:
            for tag in soup.find_all(True):
                tag.attrs = {}

        # remove empty tags recursively
        format_tags = {"br", "hr", "input", "iframe"}

        while True:
            removed = False
            for tag in soup.find_all():
                if (
                    not tag.text.strip()
                    and not any(child.name in format_tags for child in tag.find_all())
                    and tag.name not in format_tags
                ):
                    tag.decompose()
                    removed = True
            if not removed:
                break

        for tag in soup.find_all("a"):
            del tag["href"]
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

        def concat_text(text):
            text = "".join(text.split("\n"))
            text = "".join(text.split("\t"))
            text = "".join(text.split(" "))
            return text

        for tag in soup.find_all():
            children = [child for child in tag.contents if not isinstance(child, str)]
            if len(children) == 1:
                tag_text = tag.get_text()
                child_text = "".join([child.get_text() for child in tag.contents if not isinstance(child, str)])
                if concat_text(child_text) == concat_text(tag_text):
                    tag.replace_with_children()
        res = str(soup)
        lines = [line for line in res.split("\n") if line.strip()]
        res = "\n".join(lines)
        return res

    def clean_xml(self, html)->str:
        # remove tags starts with <?xml
        html = re.sub(r"<\?xml.*?>", "", html)
        # remove tags starts with <!DOCTYPE
        html = re.sub(r"<!DOCTYPE.*?>", "", html)
        # remove tags starts with <!DOCTYPE
        html = re.sub(r"<!doctype.*?>", "", html)
        return html


    def is_full_js_page(self, url):
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')

        text = soup.body.get_text(separator=' ', strip=True).lower()

        #     return True

        placeholders = ['enable javascript','requires javascript']
        if any(placeholder in text for placeholder in placeholders):
            return True

        return False

def remove_css_only(html_content: str) -> str:
    """
    Remove only CSS-related content from HTML while preserving links and other elements.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove <style> tags and their content
        for style in soup.find_all('style'):
            style.decompose()
        
        # Remove <link> tags that reference CSS files
        for link in soup.find_all('link'):
            if link.get('rel') and ('stylesheet' in link.get('rel', [])):
                link.decompose()
        
        # Remove style attributes from all elements
        for element in soup.find_all(attrs={'style': True}):
            del element['style']
        
        # Remove CSS classes (optional - you can keep them if needed for structure)
        # for element in soup.find_all(attrs={'class': True}):
        #     del element['class']
        
        return str(soup)
    except Exception as e:
        print(f"[WARNING] CSS removal failed: {e}, returning original content")
        return html_content

class FetchRawHtmlTool(Tool):
    name = "fetch_raw_html"
    description = """
• Access RAW HTML content while preserving links and structure  
• Use ONLY when you need link information from the webpage(e.g. image links;web links;etc)
• If you need text content, try fetch_html (which is optimized for text content)
• ONLY use fetch_raw_html when you specifically need:
  - Link URLs (href attributes)
  - HTML structure with preserved attributes
  - Any specific HTML elements that fetch_html might have removed"""
    inputs = {
        "url": {"type": "string", "description": "URL of an HTML webpage or website (must be text/html content)."},
        "query": {"type": "string", "description": "Your search query - what information are you looking for on this webpage?"}
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str, query: str) -> str:
        content_type = self.browser.get_content_type(url)
        print(f"content_type: {content_type}")
        if "text/html" not in content_type:
            return "This tool only supports HTML pages. Please use visit_page for PDF or TXT files."
        if url.startswith("https://www.youtube.com/watch?"):
            return "This appears to be a YouTube video page.\nPlease use visit_page to retrieve the transcript instead of fetch_raw_html."
        

        result_html = self.get_raw_html_content(url)
       
        chunks = self.split_content(result_html)
        
        all_responses = []
        for i, chunk in enumerate(chunks):
            response = self.analyze_content_with_llm(query, chunk)
            all_responses.append(f"Chunk {i+1} analysis: {response}")
        
        final_result = self.summarize_responses(query, all_responses, url)
        
        if self.is_full_js_page(url):
            final_result += "\n\nThis page is protected by JavaScript. Please use other webpages. If the website is critical, suggest to your manager that he instruct coder_agent to use the site’s API (if available) or web scraping to retrieve the content."
        
        return final_result

    def split_content(self, content: str) -> list:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        chunk_size_tokens = 40000
        overlap_tokens = 200
        max_chunks = 4
        
        tokens = tokenizer.encode(content)
        
        print(f"[DEBUG] 总token数: {len(tokens)}")
        
        if len(tokens) <= chunk_size_tokens:
            return [content]
        
        chunks = []
        start_token = 0
        
        while start_token < len(tokens) and len(chunks) < max_chunks:
            end_token = start_token + chunk_size_tokens
            
            if end_token >= len(tokens):
                chunk_tokens = tokens[start_token:]
            else:
                chunk_tokens = tokens[start_token:end_token]
            
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            print(f"[DEBUG] 创建块 {len(chunks)}，token数: {len(chunk_tokens)}")
            
            start_token = end_token - overlap_tokens
            
            if start_token >= len(tokens):
                break
        
        return chunks

    def analyze_content_with_llm(self, query: str, content: str) -> str:
        """使用大模型分析内容"""
        
        base_url = os.getenv("HTML_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("HTML_API_KEY", "")
        model_name = os.getenv("HTML_MODEL", "qwen-turbo-2025-02-11")  
        
        prompt = f"""You are an expert in analyzing HTML content and extracting useful information from it.

## Your task
I will give you a question and some HTML content. The content maybe include the answer of the question(maybe not include). You should thoroughly analyze the following question and provide a correct answer. If there is no direct answer but some keywords that maybe contribute to the answer, output them.

NOTE: This is RAW HTML content with links preserved. Pay special attention to:
- Links (href attributes) if relevant to the query
- Form elements and their attributes
- Any HTML structure that might contain the requested information

## Question
{query}

## HTML Content
{content}

## Tips
If you made any assumptions before answering a question, clarify them in your answer.
If the answer is in the content, answer the relevant information including any important links.
If the answer is not in the content, answer "I don't know". **Do not make up any information.**
If you find relevant links, include them in your response with their full URLs.
"""
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.0,
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120
            )

            # print(f"payload: {json.dumps(payload, indent=4)}")
            # print(f"response: {json.dumps(response.json(), indent=4)}")
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error calling AI model: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Failed to call AI model: {str(e)}"

    def summarize_responses(self, query: str, responses: list, url: str = None) -> str:
        """汇总所有模型响应"""
        useful_responses = []
        for response in responses:
            if "I don't know" not in response and "Failed to call" not in response and "Error calling" not in response:
                useful_responses.append(response)
        
        if not useful_responses:
            return f"No relevant information found in the webpage content. Please try other webpages or change your query."
        
        url_info = f"Accessed URL: {url}\n\n" if url else ""
        summary = f"{url_info}Raw HTML Analysis Results:\n\n"
        for i, response in enumerate(useful_responses, 1):
            summary += f"{response}\n\n"
        
        if len(useful_responses) > 1:
            summary += "Summary: Multiple relevant pieces of information were found across different sections of the webpage, including preserved HTML structure and links."
        
        return summary.strip()

    def get_raw_html_content(self, url: str) -> str:
        """Fetch the raw HTML content with CSS removed but links preserved."""
        try:
            request_kwargs = self.browser.request_kwargs.copy() if self.browser.request_kwargs else {}
            response = requests.get(url, **request_kwargs)
            raw_html = response.text
            print(f"[DEBUG] raw_html: {raw_html[:100]}")
            
            css_free_html = remove_css_only(raw_html)
            print(f"[DEBUG] css_free_html: {css_free_html[:100]}")
            return css_free_html
        except Exception as e:
            return f"Error: Failed to fetch content from {url} - {str(e)}"


    def is_full_js_page(self, url):
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')

        text = soup.body.get_text(separator=' ', strip=True).lower()

        placeholders = ['enable javascript','requires javascript']
        if any(placeholder in text for placeholder in placeholders):
            return True

        return False

class FetchPdfTool(Tool):
    name = "fetch_pdf"
    description = """
• ONLY for PDF documents (.pdf files)
• Return relevant content based on the query

NOT for HTML pages → use fetch_html
NOT for other file types → use visit_page
"""
    inputs = {
        "url": {"type": "string", "description": "URL of a PDF file (must end with .pdf or be application/pdf content-type)."},
        "query": {"type": "string", "description": "Your specific question about the PDF content - what information do you need from this document?"}
    }
    output_type = "string"

    def __init__(self, browser, downloads_folder: str = "/backup/zhangxiangxin/data/ds/download"):
        super().__init__()
        self.browser = browser
        self.downloads_folder = downloads_folder
        
        self.api_base_url = os.getenv("HTML_BASE_URL")
        self.api_key = os.getenv("HTML_API_KEY")
        self.model_name = os.getenv("HTML_MODEL")
        
        self.pdf_converter = PdfConverter()
        
        self._pdf_processor = None
        
    @property
    def pdf_processor(self):
        """懒加载PDF处理器"""
        if self._pdf_processor is None:
            from .pdf_processor import PDFProcessor
            self._pdf_processor = PDFProcessor(
                embed_model_name="BAAI/bge-large-en",
                chunk_size=1000,
                chunk_overlap=100,
                top_k=10,
                token_threshold=80000
            )
        return self._pdf_processor
        
    def forward(self, url: str, query: str) -> str:
        is_file_url = url.startswith("file://")
        is_local_path = not (url.startswith("http://") or url.startswith("https://") or url.startswith("file://"))
        
        try:
            if not is_file_url and not is_local_path:
                content_type = self.browser.get_content_type(url)
                if "application/pdf" not in content_type and not url.lower().endswith('.pdf'):
                    return "This tool only supports PDF files. Please use fetch_html for HTML pages or visit_page for other content types."
                
                try:
                    relevant_chunks, full_text = self.pdf_processor.process_pdf_with_query(
                        url, query, self.downloads_folder
                    )
                    
                    if relevant_chunks:
                        print(f"[INFO] Using RAG retrieval for PDF processing")
                        
                        from .pdf_processor import format_relevant_chunks_for_llm
                        formatted_prompt = format_relevant_chunks_for_llm(relevant_chunks, query)
                        
                        llm_response = self.call_llm(formatted_prompt)
                        
                        token_count = self.pdf_processor.count_tokens(full_text)
                        return f"""**PDF Analysis Result for Query: "{query}"**

**Analysis:**

{llm_response}

---
- Source: {url}"""
                    
                    elif full_text:
                        print(f"[INFO] Token count below threshold, using traditional processing with images")
                        pdf_path = self.pdf_processor.download_pdf(url, self.downloads_folder)
                        pdf_result = self.pdf_processor.extract_pdf_with_images(pdf_path, self.downloads_folder)
                        
                        if not pdf_result or not pdf_result.text_content.strip():
                            return f"""**PDF Analysis Result for Query: "{query}"**

**No Text Content Found**

The PDF was processed but no readable text content was extracted."""
                        
                        return self._process_pdf(pdf_result, query, url)
                        
                except Exception as e:
                    print(f"[WARNING] PDF processor failed: {e}, falling back to traditional method")
                    pass
            
            if is_file_url:
                pdf_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                pdf_result = self.pdf_processor.extract_pdf_with_images(pdf_path, self.downloads_folder)
            elif is_local_path:
                pdf_path = url
                pdf_result = self.pdf_processor.extract_pdf_with_images(pdf_path, self.downloads_folder)
            else:
                pdf_path = self.download_pdf(url)
                pdf_result = self.pdf_processor.extract_pdf_with_images(pdf_path, self.downloads_folder)
            
            if not pdf_result or not pdf_result.text_content.strip():
                return f"""**PDF Analysis Result for Query: "{query}"**

**No Text Content Found**

The PDF was processed but no readable text content was extracted. This could mean:
- The document is primarily images/scanned pages without OCR
- The PDF is corrupted or password-protected
- The document contains only non-text elements

**PDF Info:**
- URL: {url}

Please try a different document or check if this is the correct file."""
            
            return self._process_pdf(pdf_result, query, url)
            
        except Exception as e:
            return f"""**PDF Processing Error**

Error: {str(e)}

Query: {query}
URL: {url}

"""
    
    def _process_pdf(self, pdf_result, query: str, url: str) -> str:
        """处理PDF（包含图片处理）"""
        image_paths = re.findall(r'\[IMAGE EXTRACTED\] (.+)', pdf_result.text_content)
        
        image_descriptions = {}
        max_images_for_description = 5
        
        if len(image_paths) <= max_images_for_description and len(image_paths) > 0:
            print(f"[INFO] Generating descriptions for {len(image_paths)} images...")
            image_descriptions = self.generate_image_descriptions(image_paths, max_images_for_description)
            print(f"[DEBUG] image_descriptions: {image_descriptions}")
        
        enhanced_pdf_content = pdf_result.text_content
        if image_descriptions:
            for img_path, description in image_descriptions.items():
                old_marker = f"[IMAGE EXTRACTED] {img_path}"
                new_marker = f"[IMAGE EXTRACTED] {img_path}\n[IMAGE DESCRIPTION] {description}"
                enhanced_pdf_content = enhanced_pdf_content.replace(old_marker, new_marker)
        
        additional_instruction = ""
        if image_descriptions:
            additional_instruction = "\n\nThis document contains images with descriptions. When you see '[IMAGE EXTRACTED]' followed by '[IMAGE DESCRIPTION]', consider both the image location and its content description in your analysis. If you think a image is helpful to answer the question, please output the full path of the image."
        elif len(image_paths) > 0:
            additional_instruction = "\n\nThis document contains images. When you see '[IMAGE EXTRACTED]' markers, these indicate the location of images in the document. If you think a image is helpful to answer the question, please output the full path of the image."
        
        full_prompt = f"""You are an expert document analyzer. Based on the following PDF document content, please answer the user's query comprehensively.

## User Query: 
{query}

## PDF Document Content:
{enhanced_pdf_content[:80000]}{additional_instruction}


Please provide a detailed and relevant response to the user's query based on the document content above.
If the document content is not relevant to the user's query, please return "No relevant content found".
"""
        
        llm_response = self.call_llm(full_prompt)
        
        image_count = len(image_paths)
        
        image_info = ""
        if 0 < image_count <= 5:
            image_info = "\n\n- The tool has downloaded the images in the PDF to a local folder. The path is:\n"
            for path in image_paths:
                image_info += f"  {path}\n"
        
        return f"""**PDF Analysis Result for Query: "{query}"**

**Analysis:**

{llm_response}

---
- Source: {url}{image_info}"""

    def download_pdf(self, url: str) -> str:
        """下载PDF文件到本地，使用浏览器方式"""
        try:
            if "arxiv" in url:
                url = url.replace("abs", "pdf")
            
            request_kwargs = self.browser.request_kwargs.copy() if self.browser.request_kwargs else {}
            response = requests.get(url, **request_kwargs)
            response.raise_for_status()
            
            os.makedirs(self.downloads_folder, exist_ok=True)
            
            fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
            if not fname or not fname.endswith('.pdf'):
                fname = f"temp_pdf_{uuid.uuid4().hex[:8]}.pdf"
            
            download_path = os.path.join(self.downloads_folder, fname)
            
            with open(download_path, "wb") as f:
                f.write(response.content)
            
            return download_path
            
        except Exception as e:
            raise Exception(f"Failed to download PDF: {str(e)}")

    def generate_image_descriptions(self, image_paths: list, max_images: int = 5) -> dict:
        """
        使用GPT-4o-mini为PDF中的图像生成描述
        
        Args:
            image_paths: 图像路径列表
            max_images: 最大处理图像数量
            
        Returns:
            dict: {image_path: description} 的映射
        """
        if len(image_paths) > max_images:
            return {}
            
        descriptions = {}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('PLUGIN_API_KEY')}"
        }
        
        api_base_url = os.getenv('PLUGIN_BASE_URL', 'https://api.openai.com/v1')
        api_url = f"{api_base_url.rstrip('/')}/chat/completions"
        
        for img_path in image_paths:
            try:
                if not os.path.exists(img_path):
                    descriptions[img_path] = "Image file not found"
                    continue

                try:
                    compressed_img_path = compress_image_to_target_size(img_path, max_size_kb=500)
                    print(f"[INFO] Compressed image for description: {os.path.basename(img_path)}")
                except Exception as compress_error:
                    print(f"[WARNING] Image compression failed for {img_path}: {compress_error}")
                    compressed_img_path = img_path
                
                with open(compressed_img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                
                mime_type, _ = mimetypes.guess_type(compressed_img_path)
                if not mime_type:
                    mime_type = "image/jpeg"
                
                payload = {
                    "model": "gemini-3.5-flash",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": "describe this image"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 400
                }
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                description = result["choices"][0]["message"]["content"].strip()
                descriptions[img_path] = description
                
            except Exception as e:
                print(f"[WARNING] Failed to generate description for {img_path}: {e}")
                descriptions[img_path] = "Description generation failed"
                
        return descriptions

    def call_llm(self, prompt: str) -> str:
        """调用配置的AI模型（OpenAI格式API）"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.0,
            }
            
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error calling AI model: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Failed to call AI model: {str(e)}"



class DownloadTool(Tool):
    name = "download_file"
    description = """
Download a file at a given URL. The file should be of this format: [".pdf", ".txt", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".png", ".docx"]
After using this tool, for further inspection of this page you should return the download path to your manager via final_answer, and they will be able to inspect it."""
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the file to be downloaded."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        try:
            if "arxiv" in url:
                url = url.replace("abs", "pdf")
            
            request_kwargs = self.browser.request_kwargs.copy() if self.browser.request_kwargs else {}
            request_kwargs['timeout'] = 30
            
            response = requests.get(url, **request_kwargs)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "")
            extension = mimetypes.guess_extension(content_type)
            
            filename = self._extract_filename_from_url(url, extension)
            
            new_path = self._get_unique_filepath(filename)
            
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            with open(new_path, "wb") as f:
                f.write(response.content)

            return f"File was downloaded and saved under path {new_path}."
            
        except requests.exceptions.Timeout:
            return f"Error: Download timeout after 30 seconds for URL: {url}"
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to download file from {url} - {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error during download - {str(e)}"
    
    def _extract_filename_from_url(self, url: str, extension: str = None) -> str:
        """从URL中智能提取文件名"""
        if not extension:
            if url.endswith('.pdf'):
                extension = '.pdf'
            elif url.endswith('.zip'):
                extension = '.zip'
            elif url.endswith('.jpg') or url.endswith('.jpeg'):
                extension = '.jpg'
            elif url.endswith('.png'):
                extension = '.png'
            elif url.endswith('.xlsx'):
                extension = '.xlsx'
            elif url.endswith('.docx'):
                extension = '.docx'
            elif url.endswith('.mp3'):
                extension = '.mp3'
            elif url.endswith('.wav'):
                extension = '.wav'
            else:
                extension = '.file'
        
        if 'muse.jhu.edu' in url:
            match = re.search(r'/article/(\d+)/pdf', url)
            if match:
                return f"{match.group(1)}{extension}"
        
        elif 'arxiv.org' in url:
            match = re.search(r'/pdf/([^/]+?)(?:\.pdf)?$', url)
            if match:
                return f"{match.group(1)}.pdf"
        
        elif 'sample-videos.com' in url:
            match = re.search(r'/([^/]+?\.(zip|mp3|wav|mp4))$', url)
            if match:
                return match.group(1)
        
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)
        
        if '?' in path:
            path = path.split('?')[0]
        if '#' in path:
            path = path.split('#')[0]
        
        filename = os.path.basename(path)
        
        if not filename or filename in ['pdf', 'download', 'file']:
            path_parts = [p for p in path.split('/') if p and p not in ['', 'pdf', 'download', 'file']]
            if path_parts:
                filename = path_parts[-1]
            else:
                domain = parsed_url.netloc.replace('www.', '').replace('.', '_')
                filename = f"{domain}_file"
        
        if '.' not in filename:
            filename += extension
        elif not filename.endswith(extension):
            name_without_ext = os.path.splitext(filename)[0]
            filename = name_without_ext + extension
        
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        return filename
    
    def _get_unique_filepath(self, filename: str) -> str:
        """生成唯一的文件路径，避免覆盖现有文件"""
        base_path = "/backup/zhangxiangxin/data/ds/download"
        file_path = os.path.join(base_path, filename)
        
        if not os.path.exists(file_path):
            return file_path
        
        name, ext = os.path.splitext(filename)
        counter = 1
        
        while True:
            new_filename = f"{name}_{counter}{ext}"
            new_file_path = os.path.join(base_path, new_filename)
            
            if not os.path.exists(new_file_path):
                return new_file_path
            
            counter += 1
            
            if counter > 100:
                unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
                return os.path.join(base_path, unique_filename)

class ArchiveSearchTool(Tool):
    name = "find_archived_url"
    description = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    inputs = {
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {
            "type": "string",
            "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
        },
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url, date) -> str:
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"
        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()
        if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
            closest = response["archived_snapshots"]["closest"]
            print("Archive found!", closest)

        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
            print("Archive found!", closest)
        else:
            raise Exception(f"Your {url=} was not archived on Wayback Machine, try a different url.")
        target_url = closest["url"]
        self.browser.visit_page(target_url)
        header, content = self.browser._state()
        return (
            f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
            + header.strip()
        )

class WikipediaHistoryTool(Tool):
    name = "fetch_wikipedia_history"
    description = """
Get Wikipedia page revision history information.
• Specifically for Wikipedia page revision history queries
• Input Wikipedia URL and related query content
• Return revision history information analysis related to the query

NOT for non-Wikipedia pages → Only for Wikipedia pages
"""
    inputs = {
        "url": {"type": "string", "description": "Wikipedia page URL (must be a Wikipedia page)"},
        "query": {"type": "string", "description": "Query content related to revision history"}
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str, query: str) -> str:
        # Check if it's a Wikipedia page
        if not self.is_wikipedia_page(url):
            return f"This tool can only be used for wikipedia pages, the url {url} is not a wikipedia page."
        
        try:
            # Extract title from URL
            title = self.extract_title_from_url(url)
            print(f"[INFO] Extracted title from URL: {title}")
            
            # Get revision history (up to 3000 records)
            history = self.get_revision_history(title, limit=3000)
            print(f"[INFO] Retrieved {len(history)} revision records")
            
            if not history:
                return f"Unable to retrieve revision history for page '{title}'."
            
            # Process history in batches and get analysis results
            analysis_result = self.analyze_history_in_batches(query, history, title)
            
            # Save all information to file
            file_path = self.save_to_file(url, query, title, history, analysis_result)
            
            final_result = f"""**Wikipedia Revision History Analysis Result**

**Page:** {title}
**Query:** {query}
**Total Revision Records:** {len(history)} records

**Analysis Result:**

{analysis_result}

---
- Source: {url}
- All revision history records saved to: {file_path}
"""
            
            return final_result
            
        except Exception as e:
            return f"Error occurred while retrieving Wikipedia revision history: {str(e)}"

    def analyze_history_in_batches(self, query: str, history: list, title: str) -> str:
        """Analyze revision history in batches of 1000 records and summarize results"""
        batch_size = 1000
        batch_results = []
        
        print(f"[INFO] Processing {len(history)} records in batches of {batch_size}")
        
        for i in range(0, len(history), batch_size):
            batch = history[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(history) + batch_size - 1) // batch_size
            
            print(f"[INFO] Processing batch {batch_num}/{total_batches} ({len(batch)} records)")
            
            # Format this batch for LLM analysis
            formatted_batch = self.format_history_for_llm(batch)
            
            # Analyze this batch
            batch_analysis = self.analyze_history_with_llm(
                f"{query} (Batch {batch_num}/{total_batches})", 
                formatted_batch, 
                title
            )
            
            batch_results.append({
                'batch_num': batch_num,
                'record_count': len(batch),
                'analysis': batch_analysis
            })
        
        # Summarize all batch results
        return self.summarize_batch_results(query, batch_results, title, len(history))

    def summarize_batch_results(self, query: str, batch_results: list, title: str, total_records: int) -> str:
        """Summarize all batch analysis results into a final comprehensive analysis"""
        
        # Prepare summary content
        batch_summaries = []
        for result in batch_results:
            batch_summaries.append(f"**Batch {result['batch_num']} ({result['record_count']} records):**\n{result['analysis']}")
        
        combined_summaries = "\n\n---\n\n".join(batch_summaries)
        
        # Get configuration from environment variables
        base_url = os.getenv("HTML_BASE_URL")
        api_key = os.getenv("HTML_API_KEY")
        model_name = os.getenv("HTML_MODEL")
        
        # Build summary prompt
        summary_prompt = f"""You are a professional analysis expert. I have analyzed Wikipedia revision history for the page "{title}" in multiple batches(arranged chronologically), each corresponding to a different time period. Now I need you to provide a comprehensive summary.

## Original Query
{query}

## Total Records Analyzed
{total_records} revision records across {len(batch_results)} batches

## Batch Analysis Results
{combined_summaries}

## Your Task
Please provide a comprehensive summary that:
1. Synthesizes the key findings from all batches
2. Identifies the most important patterns, trends, or information related to the original query
3. Provides a coherent narrative that addresses the original query

If no relevant information was found across all batches, please state that clearly. Do not make up any information.
"""
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key to headers if available
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user", 
                        "content": summary_prompt
                    }
                ],
                "temperature": 0.0,
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=180  # Longer timeout for summary
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result["choices"][0]["message"]["content"]
                
                # Combine batch info with summary
                final_result = f"""## Comprehensive Analysis Summary

{summary}

## Processing Details
- Total revision records analyzed: {total_records}
- Processed in {len(batch_results)} batches of up to 1000 records each
- Each batch was individually analyzed and then synthesized into this comprehensive summary"""
                
                return final_result
            else:
                # Fallback to simple concatenation if API fails
                return f"API Error (HTTP {response.status_code}). Batch results:\n\n{combined_summaries}"
                
        except Exception as e:
            # Fallback to simple concatenation if summary fails
            print(f"[ERROR] Failed to generate summary: {e}")
            return f"Failed to generate comprehensive summary. Individual batch results:\n\n{combined_summaries}"

    def save_to_file(self, url: str, query: str, title: str, history: list, analysis_result: str) -> str:
        """Save all information to a file and return the file path"""
        import datetime
        
        # Create output directory
        output_dir = "/backup/zhangxiangxin/data/ds/download/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
        filename = f"wikipedia_history_{safe_title}_{timestamp}.txt"
        file_path = os.path.join(output_dir, filename)
        
        # Save raw revision history to text file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Wikipedia Revision History\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"URL: {url}\n")
                f.write(f"Title: {title}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Total Revisions: {len(history)}\n\n")
                
                f.write(f"Revision History:\n")
                f.write(f"{'-'*50}\n")
                for i, revision in enumerate(history, 1):
                    f.write(f"Revision {i}:\n")
                    f.write(f"  Time: {revision.get('Time', 'N/A')}\n")
                    f.write(f"  User: {revision.get('User', 'N/A')}\n")
                    f.write(f"  Comment: {revision.get('Comment', 'No comment')}\n")
                    f.write(f"  URL: {revision.get('Revision URL', 'N/A')}\n\n")
            
            # Return absolute path
            return os.path.abspath(file_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to save file: {e}")
            return f"Failed to save file: {str(e)}"

    def is_wikipedia_page(self, url: str) -> bool:
        """Check if URL is a Wikipedia page"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check if it's a Wikipedia domain
        wikipedia_domains = [
            'wikipedia.org',
            'en.wikipedia.org',
            'zh.wikipedia.org',
            'ja.wikipedia.org',
            'de.wikipedia.org',
            'fr.wikipedia.org',
            'es.wikipedia.org',
            'ru.wikipedia.org',
            'it.wikipedia.org',
            'pt.wikipedia.org'
        ]
        
        return any(domain.endswith(wiki_domain) for wiki_domain in wikipedia_domains)

    def extract_title_from_url(self, url: str) -> str:
        """Extract article title from Wikipedia URL"""
        path = urlparse(url).path
        if path.startswith("/wiki/"):
            return unquote(path.split("/wiki/")[1])
        raise ValueError("URL format is incorrect, unable to extract title")

    def get_revision_history(self, title: str, limit: int = 3000) -> list:
        """Get revision records (limited quantity to improve performance)"""
        
        S = requests.Session()
        
        # Determine API endpoint based on URL
        endpoint = "https://en.wikipedia.org/w/api.php"
        
        revisions = []
        cont = {}
        fetched_count = 0

        while fetched_count < limit:
            params = {
                "action": "query",
                "format": "json",
                "prop": "revisions",
                "titles": title,
                "rvlimit": min(500, limit - fetched_count),
                "rvprop": "timestamp|user|comment|ids|size",
                "rvdir": "newer",
                **cont
            }

            try:
                response = S.get(url=endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    print(f"[ERROR] API error: {data['error']}")
                    break
                
                pages = data["query"]["pages"]
                for page_id in pages:
                    if page_id == "-1":  # Page doesn't exist
                        print(f"[WARNING] Page '{title}' does not exist")
                        return []
                    
                    for rev in pages[page_id].get("revisions", []):
                        revision_data = {
                            "Time": rev.get("timestamp"),
                            "User": rev.get("user", "Anonymous user"),
                            "Comment": rev.get("comment", ""),
                            "Revision URL": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}"
                        }
                        revisions.append(revision_data)
                        fetched_count += 1

                if "continue" in data and fetched_count < limit:
                    cont = data["continue"]
                    time.sleep(0.3)
                else:
                    break
                    
            except Exception as e:
                print(f"[ERROR] Error retrieving revision records: {e}")
                break

        return revisions

    def format_history_for_llm(self, history: list) -> str:
        """
        Formats history records.
        """
        if not history:
            return "No revision records"

        # Use all provided history records (batch processing handles the limitation)
        limited_history = history

        headers = ["#", "Time", "User", "Edit summary", "Revision URL"]
        header_row = f"| {' | '.join(headers)} |"
        separator_row = f"|{'|'.join(['---'] * len(headers))}|"

        table_rows = []
        for i, revision in enumerate(limited_history, 1):
            time = str(revision.get('Time', 'N/A')).replace('|', r'\|').replace('\n', ' ')
            user = str(revision.get('User', 'N/A')).replace('|', r'\|').replace('\n', ' ')
            comment = str(revision.get('Comment', 'No summary')).replace('|', r'\|').replace('\n', ' ')
            url = str(revision.get('Revision URL', 'N/A')).replace('|', r'\|').replace('\n', ' ')
            
            row_data = [str(i), time, user, comment, url]
            table_rows.append(f"| {' | '.join(row_data)} |")

        return "\n".join([header_row, separator_row] + table_rows)

    def analyze_history_with_llm(self, query: str, history_data: str, title: str) -> str:
        """Analyze revision history using LLM"""
        
        # Get configuration from environment variables
        base_url = os.getenv("HTML_BASE_URL")
        api_key = os.getenv("HTML_API_KEY")
        model_name = os.getenv("HTML_MODEL")
        
        # Build prompt
        prompt = f"""You are a professional analysis expert. I will give you a query and related Wikipedia page revision history records. Please analyze these revision history information based on the query content.

## Your task
I will give you a query and related Wikipedia page revision history records. Please analyze these revision history information based on the query content.

## Query Content
{query}

## Revision History Data
{history_data}

If you find relevant links, include them in your response with their full URLs.
If there is no information directly related to the query in the history records, please state "No information directly related to the query was found in the provided revision history records". Do not make up any information.
"""
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key to headers if available
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error calling AI model: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Failed to call AI model: {str(e)}"
