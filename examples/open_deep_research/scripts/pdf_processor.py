"""
PDF处理模块，包含基于向量检索的智能内容提取功能
"""
import os
import re
import uuid
import tempfile
import base64
import mimetypes
from typing import List, Tuple, Optional
from urllib.parse import urlparse
from pathlib import Path

import requests
import pathvalidate
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfminer.high_level
import fitz
from transformers import AutoTokenizer


class PDFProcessor:
    """PDF文档处理器，支持向量检索和智能内容提取"""
    
    def __init__(self, 
                 embed_model_name: str = "BAAI/bge-large-en",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 top_k: int = 5,
                 token_threshold: int = 80000):
        """
        初始化PDF处理器
        
        Args:
            embed_model_name: 嵌入模型名称
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            top_k: 返回最相关的段落数量
            token_threshold: 使用RAG的token数阈值
        """
        self.embed_model_name = embed_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.token_threshold = token_threshold
        
        self._embed_model = None
        self._tokenizer = None
        
    @property
    def embed_model(self):
        """懒加载嵌入模型"""
        if self._embed_model is None:
            print(f"Loading embedding model: {self.embed_model_name}")
            self._embed_model = SentenceTransformer(self.embed_model_name)
        return self._embed_model
    
    @property
    def tokenizer(self):
        """懒加载tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._tokenizer
    
    def download_pdf(self, url: str, downloads_folder: str = "/backup/zhangxiangxin/data/ds/download") -> str:
        """
        下载PDF文件到本地
        
        Args:
            url: PDF文件URL
            downloads_folder: 下载目录
            
        Returns:
            本地文件路径
        """
        try:
            if "arxiv" in url:
                url = url.replace("abs", "pdf")
                if not url.endswith('.pdf'):
                    url += '.pdf'
            
            _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}
            response = requests.get(url, headers=_headers, timeout=30)
            response.raise_for_status()
            
            os.makedirs(downloads_folder, exist_ok=True)
            
            fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
            if not fname or not fname.endswith('.pdf'):
                fname = f"temp_pdf_{uuid.uuid4().hex[:8]}.pdf"
            
            download_path = os.path.join(downloads_folder, fname)
            
            print(f"[INFO] Saving PDF to {download_path}")
            with open(download_path, "wb") as f:
                f.write(response.content)
            
            return download_path
            
        except Exception as e:
            raise Exception(f"Failed to download PDF: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF文件提取文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        try:
            text_content = pdfminer.high_level.extract_text(pdf_path)
            return text_content
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 文本内容
            
        Returns:
            token数量
        """
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"[WARNING] Token counting failed: {e}")
            return len(text) // 4
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        将文本分割成重叠的块
        
        Args:
            text: 原始文本
            
        Returns:
            文本块列表
        """
        if not text.strip():
            return []
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                if len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:] + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def build_vector_index(self, chunks: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        为文本块构建向量索引
        
        Args:
            chunks: 文本块列表
            
        Returns:
            (向量矩阵, 文本块列表)
        """
        if not chunks:
            return np.array([]), []
        
        print(f"Building vector index for {len(chunks)} chunks...")
        
        embeddings = self.embed_model.encode(chunks, convert_to_numpy=True)
        
        return embeddings, chunks
    
    def retrieve_relevant_chunks(self, 
                                query: str, 
                                embeddings: np.ndarray, 
                                chunks: List[str]) -> List[Tuple[str, float]]:
        """
        基于查询检索最相关的文本块
        
        Args:
            query: 查询文本
            embeddings: 文本块的嵌入向量矩阵
            chunks: 文本块列表
            
        Returns:
            (文本块, 相似度分数)的列表，按相似度降序排列
        """
        if len(chunks) == 0:
            return []
        
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            relevant_chunks.append((chunks[idx], float(similarities[idx])))
        
        return relevant_chunks
    
    def calculate_image_quality_score(self, pix):
        """
        计算图像的质量评分，用于决定是否值得保存
        
        Returns:
            float: 质量评分 (0-100)，分数越高质量越好
        """
        try:
            score = 0
            
            area = pix.width * pix.height
            if area >= 10000:
                score += 30
            elif area >= 2500:
                score += 20
            elif area >= 1000:
                score += 10
            
            aspect_ratio = max(pix.width, pix.height) / min(pix.width, pix.height)
            if aspect_ratio <= 3:
                score += 20
            elif aspect_ratio <= 10:
                score += 10
            
            img_data = pix.samples
            if img_data:
                sample_step = max(1, len(img_data) // 500)
                sampled_data = img_data[::sample_step]
                
                unique_values = set()
                for i in range(0, min(len(sampled_data), 1000), pix.n):
                    if i + pix.n <= len(sampled_data):
                        pixel_bytes = tuple(sampled_data[i:i+pix.n])
                        unique_values.add(pixel_bytes)
                
                color_count = len(unique_values)
                if color_count >= 50:
                    score += 30
                elif color_count >= 20:
                    score += 20
                elif color_count >= 10:
                    score += 10
            
            if pix.alpha:
                alpha_channel = []
                sample_step = max(1, len(img_data) // 200)
                for i in range(pix.n-1, len(img_data), pix.n * sample_step):
                    if i < len(img_data):
                        alpha_channel.append(img_data[i])
                
                if alpha_channel:
                    opaque_ratio = sum(1 for a in alpha_channel if a > 200) / len(alpha_channel)
                    if opaque_ratio > 0.8:
                        score += 20
                    elif opaque_ratio > 0.5:
                        score += 10
            else:
                score += 20 
                
            return score
            
        except Exception as e:
            print(f"[WARNING] Image quality calculation error: {e}")
            return 50

    def is_valid_image(self, pix, min_width=50, min_height=50, min_unique_colors=10):
        """
        检查图像是否为有效的内容图像，过滤掉遮罩、填充层等异常图像
        """
        try:
            if pix.width < min_width or pix.height < min_height:
                return False
            
            aspect_ratio = max(pix.width, pix.height) / min(pix.width, pix.height)
            if aspect_ratio > 20:
                return False
                
            img_data = pix.samples
            if not img_data:
                return False
                
            sample_step = max(1, len(img_data) // 1000)
            sampled_data = img_data[::sample_step]
            
            unique_values = set()
            total_samples = 0
            
            for i in range(0, len(sampled_data), pix.n):
                if i + pix.n <= len(sampled_data):
                    pixel_bytes = tuple(sampled_data[i:i+pix.n])
                    unique_values.add(pixel_bytes)
                    total_samples += 1
                    
                    if len(unique_values) >= min_unique_colors:
                        break
                        
            if len(unique_values) < min_unique_colors:
                return False
            
            if total_samples > 0:
                color_diversity_ratio = len(unique_values) / total_samples
                if color_diversity_ratio < 0.01:
                    return False
                
            if pix.alpha:
                alpha_channel = []
                for i in range(pix.n-1, len(sampled_data), pix.n):
                    if i < len(sampled_data):
                        alpha_channel.append(sampled_data[i])
                        
                if alpha_channel:
                    transparent_ratio = sum(1 for a in alpha_channel if a < 50) / len(alpha_channel)
                    if transparent_ratio > 0.8:
                        return False
                        
            if len(unique_values) == 1:
                return False
            elif len(unique_values) <= 3 and total_samples > 100:
                return False
                        
            return True
            
        except Exception as e:
            print(f"[WARNING] Image validation error: {e}")
            return True

    def extract_pdf_with_images(self, pdf_path: str, downloads_folder: str = "/backup/zhangxiangxin/data/ds/download"):
        """
        提取PDF内容和图像，使用PyMuPDF
        
        Args:
            pdf_path: PDF文件路径
            downloads_folder: 下载目录
            
        Returns:
            包含文本和图像标记的完整内容
        """
        pdf_pics_dir = os.path.join(downloads_folder, "pdf_pics")
        os.makedirs(pdf_pics_dir, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        combined_content = []
        pdf_name = Path(pdf_path).stem
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            combined_content.append(f"\n--- Page {page_num + 1} ---\n")
            
            text_blocks = page.get_text("dict")
            
            image_list = page.get_images()
            
            page_elements = []
            
            for block in text_blocks["blocks"]:
                if "lines" in block:
                    y_position = block["bbox"][1]
                    text_content = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_content += span["text"]
                        text_content += "\n"
                    if text_content.strip():
                        page_elements.append({
                            "type": "text",
                            "y_pos": y_position,
                            "content": text_content.strip()
                        })
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 1:
                    continue
                
                quality_score = self.calculate_image_quality_score(pix)
                
                if not self.is_valid_image(pix):
                    print(f"[INFO] Skipping suspicious image on page {page_num + 1}, image {img_index + 1}")
                    continue
                
                if quality_score < 40:
                    print(f"[INFO] Skipping low-quality image on page {page_num + 1}, image {img_index + 1}")
                    continue
                
                image_rects = page.get_image_rects(img)
                if image_rects:
                    y_position = image_rects[0].y0
                else:
                    y_position = 0
                
                try:
                    test_pix = pix
                    if pix.alpha:
                        test_pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    estimated_size = test_pix.width * test_pix.height * test_pix.n * 0.1
                    
                    if estimated_size < 500:
                        print(f"[INFO] Skipping tiny image on page {page_num + 1}, image {img_index + 1}")
                        if test_pix != pix:
                            test_pix = None
                        continue
                    
                    img_filename = f"{pdf_name}_page{page_num + 1}_img{img_index + 1}.png"
                    img_path = os.path.join(pdf_pics_dir, img_filename)
                    
                    if pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    pix.save(img_path)
                    
                    file_size = os.path.getsize(img_path)
                    if file_size < 1024:
                        print(f"[INFO] Removing tiny saved image {img_filename}")
                        os.remove(img_path)
                        pix = None
                        continue
                    
                    print(f"[INFO] Saved image: {img_filename} ({file_size} bytes)")
                    pix = None
                    
                    page_elements.append({
                        "type": "image",
                        "y_pos": y_position,
                        "content": f"[IMAGE EXTRACTED] {img_path}"
                    })
                    
                except Exception as e:
                    print(f"[WARNING] Failed to process image: {e}")
                    continue
            
            page_elements.sort(key=lambda x: x["y_pos"])
            
            for element in page_elements:
                combined_content.append(element["content"])
                combined_content.append("\n")
        
        doc.close()
        
        text_content = "\n".join(combined_content)
        
        class PDFResult:
            def __init__(self, text_content):
                self.text_content = text_content
                self.title = None
        
        return PDFResult(text_content)
    
    def process_pdf_with_query(self, 
                              url: str, 
                              query: str, 
                              downloads_folder: str = "/backup/zhangxiangxin/data/ds/download",
                              cleanup: bool = False) -> Tuple[List[Tuple[str, float]], str]:
        """
        完整的PDF处理流程：下载、提取、分块、向量检索
        
        Args:
            url: PDF文件URL
            query: 查询问题
            downloads_folder: 下载目录
            cleanup: 是否清理临时文件（默认False，保留文件）
            
        Returns:
            (相关文本块列表, 原始文本)
        """
        pdf_path = None
        try:
            pdf_path = self.download_pdf(url, downloads_folder)
            
            full_text = self.extract_text_from_pdf(pdf_path)
            
            if not full_text.strip():
                raise Exception("PDF appears to be empty or text extraction failed")
            
            token_count = self.count_tokens(full_text)
            print(f"[INFO] PDF token count: {token_count}")
            
            if token_count > self.token_threshold:
                print(f"[INFO] Token count ({token_count}) > threshold ({self.token_threshold}), using RAG retrieval")
                chunks = self.split_text_into_chunks(full_text)
                
                if not chunks:
                    raise Exception("No text chunks created from PDF")
                
                embeddings, chunks = self.build_vector_index(chunks)
                
                relevant_chunks = self.retrieve_relevant_chunks(query, embeddings, chunks)
                
                return relevant_chunks, full_text
            else:
                print(f"[INFO] Token count ({token_count}) <= threshold ({self.token_threshold}), using traditional processing")
                return [], full_text
            
        except Exception as e:
            print(f"[ERROR] PDF processing failed: {e}")
            raise
            
        finally:
            # if cleanup and pdf_path and os.path.exists(pdf_path):
            #     try:
            #         os.remove(pdf_path)
            #         print(f"[INFO] Cleaned up temporary file: {pdf_path}")
            #     except Exception as e:
            #         print(f"[WARNING] Failed to clean up file {pdf_path}: {e}")
            pass



def format_relevant_chunks_for_llm(relevant_chunks: List[Tuple[str, float]], 
                                   query: str, 
                                   max_context_length: int = 12000) -> str:
    """
    格式化相关文本块供LLM处理
    
    Args:
        relevant_chunks: 相关文本块列表
        query: 查询问题
        max_context_length: 最大上下文长度
        
    Returns:
        格式化的提示文本
    """
    if not relevant_chunks:
        return f"Query: {query}\n\nNo relevant content found in the PDF."
    
    context_parts = []
    current_length = 0
    
    for i, (chunk, score) in enumerate(relevant_chunks):
        chunk_text = f"**Relevant Section {i+1}** (Similarity: {score:.3f}):\n{chunk}\n"
        
        if current_length + len(chunk_text) > max_context_length:
            break
            
        context_parts.append(chunk_text)
        current_length += len(chunk_text)
    
    context = "\n".join(context_parts)
    
    prompt = f"""You are an expert document analyzer. Based on the following PDF document content, please answer the user's query comprehensively.

## User Query: 
{query}

## PDF Document Content:
{context}


Please provide a detailed and relevant response to the user's query based on the document content above.
If the document content is not relevant to the user's query, please return "No relevant content found".
"""

    return prompt 
