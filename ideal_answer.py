

import os
import io
import base64
from PIL import Image
from groq import Groq

def call_api_generate_paragraph(image: Image.Image, api_key: str) -> str:
    """
    Generate a detailed description using Groq's Vision model.
    """
    if not api_key:
        return "Please provide a Groq API Key."
    
    if image is None:
        return "No image provided."

    try:
        # 1. Initialize Groq client
        client = Groq(api_key=api_key)
        
        # 2. Convert PIL Image to base64
        # Groq (like OpenAI) expects the image as a base64 encoded string.
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 3. Define the prompt
        prompt = (
            "You are an expert image description writer. "
            "Write a rich, detailed descriptive paragraph (about 150-300 words) "
            "that vividly describes this image. Focus on objects, colors, people, "
            "lighting, mood, and any activities visible. "
            "Provide ONLY the description text."
        )
        
        # 4. Call Groq Completion (using a vision-capable model)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        if completion.choices[0].message.content:
            return completion.choices[0].message.content.strip()
        else:
            return "Groq failed to generate a response. Please check your image or API key."

    except Exception as e:
        return f"Groq API Error: {str(e)}"
