from KG_builder.llm.base.base_model import BaseLLM
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
# from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class CostModelAPIError(Exception):
    pass
    
class CostModel(BaseLLM):
    """Paid API models (GPT, Gemini)"""
    def __init__(self, **args):
        super().__init__(**args)
    

def _reformat_messages(messages: list):
    """
    Reformat messages for Gemini.

    Args:
        messages: The list of messages provided in the request.

    Returns:
        tuple: (system_instruction, contents_list)
    """
    system_instruction = None
    contexts = []

    for message in messages:
        if message["role"] == "system":
            system_instruction = message["content"]
        else:
            context = types.Content(
                parts=[types.Part(text=message["content"])],
                role=message["role"],
            )
            contexts.append(context)

    return system_instruction, contexts


def _message_type(messages: list):
    if any(isinstance(item, types.Part) for item in messages):
        return "file"
    
    # chat message
    if isinstance(messages, list) and len(messages) > 0:
        if isinstance(messages[0], dict) and "role" in messages[0]:
            return "chat"
    
    return "unknown"


class GeminiModel(CostModel):
    def __init__(self, **args):
        super().__init__(**args)
        self.api_key = os.environ["GEMINI_API_KEY"]
        
        if not self.api_key:
            raise ValueError("Set api key in .env file of pass an valid api key")
        
        try: 
            self.instance = genai.Client(api_key=self.api_key)
        
        except Exception as e:
            raise CostModelAPIError(f"Failed to connect to Gemini: {str(e)}")


    def generate_response(self, messages: list, **args):
        response_format = args.get("response_format")
        
        config_params: dict[str, any] = {}
        system_instruction = None
        
        if _message_type(messages=messages) == "chat":
            system_instruction, context = _reformat_messages(messages)
            
        elif _message_type(messages=messages) == "file":
            context = messages
            
        if system_instruction:
            config_params["system_instruction"] = system_instruction
            
        if response_format is not None and response_format["type"] == "json_object":
            config_params["response_mime_type"] = "application/json"
            if "response_schema" in response_format:
                config_params["response_schema"] = response_format["response_schema"]
                
        config = GenerateContentConfig(**config_params)
        
        try:
            response = self.instance.models.generate_content(
                model=self.name,
                contents=context, 
                config=config
            )
            # check if text in response format
            if hasattr(response, "text"):
                return response.text
            
            # check if candidate in response format
            elif hasattr(response, "candidate") and response.candidates():
                return response.candidates[0].content.parts[0].text
            
            else:
                raise CostModelAPIError("No valid response from Gemini")
        
        except Exception as e:
            raise CostModelAPIError(f"Error: {str(e)}")
    
    
class GPTModel(CostModel):
    def __init__(self, **args):
        super().__init__(**args)
        # self.client = OpenAI(api_key=os.environ["OPENAI"])
        
    def generate_response(self, context: str, **args):
        response = self.client.responses.create(
            model=self.name,
            messages=[
                {"role": "system", "content" : args.get("system")},
                {"role": "user", "content" : context}
            ]
        )
        response = response.output_text
        
    
    