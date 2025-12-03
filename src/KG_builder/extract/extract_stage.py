import json
import time
import os
from pathlib import Path
from KG_builder.llm.base.base_model import BaseLLM
from KG_builder.triple_models import TripleList
from KG_builder.utils.llm_utils import load_model
from KG_builder.utils.clean_data import clean_vn_text
from KG_builder.config import SECTIONS_DEFINITION
from KG_builder.utils.chunking import extract_specific_sections
from KG_builder.convert_pdf_to_text.extract_table import extract_triples_from_table, extract_table_from_pdf
from KG_builder.convert_pdf_to_text.core import extract_context_from_pdf

class Stage:
    def __init__(
        self,
        text: str,
        llm: BaseLLM,
        predicates: dict[str, list[str]],
        response_format: dict[str, any],
        context: str, 
        system_instruction: str,
        main_subject: str | None = None,
    ):
        self.text = text
        self.llm = llm
        self.predicates = predicates
        self.context = context
        self.system_instruction = system_instruction
        self.main_subject = main_subject
        self.response_format = response_format
        
        
    def build_message(self):
        if not self.main_subject:
            self.main_subject = ""
            
        messages = [
            {"role": "user", "content": self.context.format(
                main_subject=self.main_subject,
                predicates=self.predicates,
                text=self.text
            )},
            {"role": "system", "content": self.system_instruction}
        ]
        return messages
    
    
    def extract_triples(self):
        messages = self.build_message()
        response = self.llm.generate_response(messages, response_format=self.response_format)
        return json.loads(response)
    

class TripleExtraction:
    def __init__(self):
        self.stages: list[Stage] = []
        

    def add_stage(self, stage: Stage):
        self.stages.append(stage)
        
    
    def run(self, llm: BaseLLM, pdf_path: str):
        # load text -> clean text
        self.stages: list[Stage] = []
        try:
            text = extract_context_from_pdf(
                pdf_path=pdf_path
            )
        except RuntimeError as e:
            print(f"[WARN] Cannot extract text from {pdf_path}: {e}")
            text = ""

        if not text:
            print(f"No text extracted from file.")
            return []
        
        cleaned_text = clean_vn_text(text)
        
        for i, section in enumerate(SECTIONS_DEFINITION):
            section_text = extract_specific_sections(cleaned_text, section["start_word"], section["end_word"])
            
            stage=Stage(
                text=section_text,
                llm=llm,
                predicates=section["predicates"],
                response_format=response_format,
                context=section["context"],
                system_instruction=section["system_instruction"]
            )
            self.add_stage(stage=stage)
            
        results = []
        current_main_subject = None
        
        for i, stage in enumerate(self.stages):
            if current_main_subject:
                stage.main_subject = current_main_subject
            result = stage.extract_triples()
            
            if result.get("main_subject"):
                current_main_subject = result.get("main_subject")
            results.append(result)

        table_data = extract_table_from_pdf(
            pdf_path=pdf_path,
            genai=llm
        )
        
        parsed_data = json.loads(table_data)
        table_data_path = f"../table_data/{current_main_subject}.json"
        
        os.makedirs(os.path.dirname(table_data_path), exist_ok=True)
        with open(table_data_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        
        table_triples = extract_triples_from_table(table_data_path=table_data_path, main_subject=current_main_subject)
        
        results.extend(table_triples)
        
        output_path = f"../output/triples_{current_main_subject}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        return results
    

if __name__ == "__main__":
    llm = load_model("gemini-2.5-flash")
    
    response_format = {
        "type": "json_object",
        "response_mime_type": "application/json",
        "response_schema": TripleList
    }
    
    builder = TripleExtraction()
    
    start = time.perf_counter()
    
    for path in os.listdir("../pdf_data/")[:10]:
        pdf_path = os.path.join("../pdf_data/", path)
        start_file = time.perf_counter()
        
        results = builder.run(llm=llm, pdf_path=pdf_path)
        
        end_file = time.perf_counter()
        run_time_file = end_file - start_file
        print(f"File {pdf_path} extracted triples completes in {run_time_file}")
        
    end = time.perf_counter()
    run_time = end - start
    print(f"Triples extraction completes in {run_time}")
