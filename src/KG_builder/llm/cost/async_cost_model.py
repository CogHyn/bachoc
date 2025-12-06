
from KG_builder.llm.base.async_base_mode import AsyncBaseLLM
from google import genai
from google.genai.types import GenerateContentConfig
# from openai import OpenAI
import asyncio
import os


from dotenv import load_dotenv

load_dotenv()

class AsyncCostModel(AsyncBaseLLM):
    """Paid API models (GPT, Gemini)"""
    def __init__(self, **args):
        super().__init__(**args)

class AsyncGeminiModel(AsyncCostModel):
    def __init__(self, **args):
        super().__init__(**args)
        self.instance = genai.Client(api_key=os.environ["GEMINI_API_KEY_3"])
        
        
    async def generate_response(self, context: str, **args):
        config = GenerateContentConfig(
            system_instruction=args.get("system"),
        )
        
        event_loop = asyncio.get_event_loop()
        
        response = await event_loop.run_in_executor(
            None,
            lambda: self.instance.models.generate_content(
                    model=self.name,
                    contents=context, 
                    config=config
                )
        )
        
        return response.text
    
    
class AsyncGPTModel(AsyncBaseLLM):
    def __init__(self, **args):
        super().__init__(**args)
        self.client = OpenAI(api_key=os.environ["OPENAI"])
        
    async def generate_response(self, context: str, **args):
        event_loop = asyncio.get_event_loop()
        response = await event_loop.run_in_executor(
            None, 
            lambda: self.client.responses.create(
                model=self.name,
                messages=[
                    {"role": "system", "content" : args.get("system")},
                    {"role": "user", "content" : context}
                ]
            )
        )
        response = response.output_text
        
        
    if __name__ == "__main__":
        
        from dotenv import load_dotenv
        
        load_dotenv()
        
        from KG_builder.llm.cost.cost_model import GeminiModel
        from KG_builder.llm.cost.async_cost_model import AsyncGeminiModel
        from KG_builder.utils.utils import perf
        
        inst: GeminiModel = GeminiModel(model_name="gemini-2.0-flash")
        async_inst: AsyncGeminiModel  = AsyncGeminiModel(model_name="gemini-2.0-flash")
        
        
        chats = [
            "Hello, my name is Huynh",
            "Show me how to code OS",
            "Hello, my name is Huynh",
            "Show me how to code OS",
            "Hello, my name is Huynh",
            "Show me how to code OS",
            "Hello, my name is Huynh",
            "Show me how to code OS",
        ]
        
        @perf
        def query(llm: GeminiModel, chats):
            for c in chats:
                llm.chat(
                    c
                )
                
                
        import asyncio
        @perf
        async def async_query(llm: AsyncGeminiModel, chats):
            resp = await asyncio.gather(*[llm.chat(c) for c in chats])
            for r in resp:
                print(r)
            
        
        # query(inst, chats)
        
        asyncio.run(async_query(async_inst, chats))
        