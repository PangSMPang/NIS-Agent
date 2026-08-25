import base64
import json
import mimetypes
import os
import uuid
from io import BytesIO
from typing import Optional

import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

from smolagents import Tool, tool


load_dotenv(override=True)


def process_images_and_text(image_path, query, client):
    from transformers import AutoProcessor

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        },
    ]
    idefics_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-chatty")
    prompt_with_template = idefics_processor.apply_chat_template(messages, add_generation_prompt=True)

    # load images from local directory

    # encode images to strings which can be sent to the endpoint
    def encode_local_image(image_path):
        # load image
        image = Image.open(image_path).convert("RGB")

        # Convert the image to a base64 string
        buffer = BytesIO()
        image.save(buffer, format="JPEG")  # Use the appropriate format (e.g., JPEG, PNG)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # add string formatting required by the endpoint
        image_string = f"data:image/jpeg;base64,{base64_image}"

        return image_string

    image_string = encode_local_image(image_path)
    prompt_with_images = prompt_with_template.replace("<image>", "![]({}) ").format(image_string)

    payload = {
        "inputs": prompt_with_images,
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 200,
        },
    }

    return json.loads(client.post(json=payload).decode())[0]


# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("/backup/zhangxiangxin/data/ds/download", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def compress_image_to_target_size(image_path, max_size_kb=300):
    """
    Compress image to target size using Pillow.
    Priority: Quality reduction first, then size reduction.
    No cropping, only compression and resizing.
    """
    target_size_bytes = max_size_kb * 1024
    file_size_bytes = os.path.getsize(image_path)
    
    # If already small enough, return original
    if file_size_bytes <= target_size_bytes:
        print(f"Image is already small enough: {file_size_bytes/1024:.1f}KB")
        return image_path
    
    # Create compressed pics directory if it doesn't exist
    compressed_dir = "/backup/zhangxiangxin/data/ds/download/compressed_pics"
    os.makedirs(compressed_dir, exist_ok=True)
    
    # Open and prepare image
    img = Image.open(image_path)
    original_format = img.format
    
    # Convert to RGB if necessary (for JPEG compression)
    if img.mode in ('RGBA', 'LA', 'P'):
        if original_format == 'PNG' and img.mode == 'RGBA':
            # Preserve transparency by converting to RGB with white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        else:
            img = img.convert('RGB')
    
    # Save to compressed_pics directory
    compressed_path = os.path.join(compressed_dir, f"compressed_{os.path.basename(image_path)}")
    
    # Determine output format (prefer JPEG for better compression)
    if image_path.lower().endswith('.png'):
        output_format = 'JPEG'
        compressed_path = compressed_path.replace('.png', '.jpg')
    else:
        output_format = 'JPEG'
        if not compressed_path.lower().endswith('.jpg'):
            compressed_path += '.jpg'
    
    original_width, original_height = img.size
    current_img = img.copy()
    
    # Phase 1: Quality reduction (95 -> 85 -> 75 -> 65 -> 55 -> 45 -> 35)
    qualities = [95, 85, 75, 65, 55, 45, 35]
    
    for quality in qualities:
        # Save with current quality
        buffer = BytesIO()
        current_img.save(buffer, format=output_format, quality=quality, optimize=True)
        size_bytes = len(buffer.getvalue())
        
        print(f"Testing quality {quality}: {size_bytes/1024:.1f}KB")
        
        if size_bytes <= target_size_bytes:
            # Save final image
            current_img.save(compressed_path, format=output_format, quality=quality, optimize=True)
            final_size = os.path.getsize(compressed_path)
            print(f"Compression successful with quality {quality}: {file_size_bytes/1024:.1f}KB -> {final_size/1024:.1f}KB")
            return compressed_path
    
    # Phase 2: Size reduction with minimum quality (35)
    min_quality = 35
    scale = 1.0
    
    while scale > 0.1:
        scale *= 0.9
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save with minimum quality
        buffer = BytesIO()
        resized_img.save(buffer, format=output_format, quality=min_quality, optimize=True)
        size_bytes = len(buffer.getvalue())
        
        print(f"Testing resize {scale:.2f} ({new_width}x{new_height}): {size_bytes/1024:.1f}KB")
        
        if size_bytes <= target_size_bytes:
            # Save final image
            resized_img.save(compressed_path, format=output_format, quality=min_quality, optimize=True)
            final_size = os.path.getsize(compressed_path)
            print(f"Compression successful with resize {scale:.2f}: {file_size_bytes/1024:.1f}KB -> {final_size/1024:.1f}KB")
            return compressed_path
    
    final_width = int(original_width * scale)
    final_height = int(original_height * scale)
    final_img = img.resize((final_width, final_height), Image.Resampling.LANCZOS)
    final_img.save(compressed_path, format=output_format, quality=min_quality, optimize=True)
    
    final_size = os.path.getsize(compressed_path)
    print(f"Maximum compression applied with scale {scale:.2f}: {file_size_bytes/1024:.1f}KB -> {final_size/1024:.1f}KB")
    return compressed_path


headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('PLUGIN_API_KEY')}"}


def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path


class VisualQATool(Tool):
    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "string",
        },
        "question": {"description": "the question to answer", "type": "string", "nullable": True},
    }
    output_type = "string"

    client = InferenceClient("HuggingFaceM4/idefics2-8b-chatty")

    def forward(self, image_path: str, question: Optional[str] = None) -> str:
        output = ""
        add_note = False
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        try:
            output = process_images_and_text(image_path, question, self.client)
        except Exception as e:
            print(e)
            if "Payload Too Large" in str(e):
                new_image_path = resize_image(image_path)
                output = process_images_and_text(new_image_path, question, self.client)

        if add_note:
            output = (
                f"You did not provide a particular question, so here is a detailed caption for the image: {output}"
            )

        return output


@tool
def visualizer(image_path: str, question: Optional[str] = None) -> str:
    """A tool that can answer questions about attached images.

    Args:
        image_path: The path to the image on which to answer the question. This should be a local path to downloaded image.
        question: The question to answer.
    """

    add_note = False
    if not question:
        add_note = True
        question = "Please write a detailed caption for this image."
    if not isinstance(image_path, str):
        raise Exception("You should provide at least `image_path` string argument to this tool!")
    
    # Check if image_path is a local path
    if image_path.startswith(('http://', 'https://', 'ftp://')):
        return "Error: This tool only supports local image files. Please download the image first and provide a local file path."

    # Compress image to target size if necessary
    processed_image_path = compress_image_to_target_size(image_path, max_size_kb=300)
    
    mime_type, _ = mimetypes.guess_type(processed_image_path)
    # Handle JPEG mime type for compressed images
    if processed_image_path.endswith('.jpg') and not mime_type:
        mime_type = 'image/jpeg'
    
    try:
        base64_image = encode_image(processed_image_path)
    except Exception as e:
        raise Exception(f"Failed to encode image: {e}")

    payload = {
        "model": "gemini-3.5-flash",
        # "model": "gpt-4o",
        # "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        "max_tokens": 1000,
    }
    # Construct the API URL using environment variable
    api_base_url = os.getenv('PLUGIN_BASE_URL', 'https://api.openai.com/v1')
    api_url = f"{api_base_url.rstrip('/')}/chat/completions"
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        if "413" in str(e) or "Payload Too Large" in str(e):
            # If still too large, try more aggressive compression
            further_compressed = compress_image_to_target_size(image_path, max_size_kb=300)
            base64_image = encode_image(further_compressed)
            payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:{mime_type};base64,{base64_image}"
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API request failed: {e}")
    except Exception as e:
        raise Exception(f"Response format unexpected: {response.json() if 'response' in locals() else str(e)}")

    if add_note:
        output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

    return output
