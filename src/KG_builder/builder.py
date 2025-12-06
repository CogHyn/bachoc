from __future__ import annotations

import asyncio
import json
import os
import glob
import time
from pathlib import Path
from typing import Dict, Set, Any, List

from KG_builder.extract.extract_stage import TripleExtraction
from KG_builder.embedding.load.free import QwenEmbedding
from KG_builder.utils.llm_utils import load_async_model, load_model
from KG_builder.prompts.prompts import DEFINITION_PROMPT
from KG_builder.extract.definition import async_collect_definition
from KG_builder.models.ops import RelationTypeService, EntityService
from KG_builder.triple_models import TripleList

class KnowledgeGraphBuilder:
    def __init__(self, *,
                triple_extraction: TripleExtraction,
                response_format: Dict[str, Any],
                threshold: float = 0.2,
                definition_model: str = "gemini-2.0-flash",
                llm_model: str = "gemini-2.5-flash",
                embedding_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        
        self.threshold = threshold
        self.response_format = response_format
        self.builder = triple_extraction
        self.llm = load_model(llm_model)
        self.definition_model = load_async_model(definition_model)
        
        self.embedding_name = embedding_model
        if "qwen" in self.embedding_name.lower():
            self.embed_model = QwenEmbedding(model_name=self.embedding_name)
            
        self.db_lock = asyncio.Lock()

    async def process_single_file(self, input_path: str, output_path: str, semaphore: asyncio.Semaphore):
        async with semaphore:
            print(f"Processing: {input_path}")
            start_time = time.perf_counter()
            
            # 1. Extraction (Async/Threaded)
            try:
                if hasattr(self.builder, 'run_async'):
                    results = await self.builder.run_async(llm=self.llm, pdf_path=input_path, response_format=self.response_format)
                else:
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(None, self.builder.run, self.llm, input_path, self.response_format)
            except Exception as e:
                print(f"Extraction error in {input_path}: {e}")
                return

            if not results:
                print(f"No triples found in {input_path}")
                return

            # 2. Collect Entities & Predicates
            entities: Set[str] = set()
            predicates: Set[str] = set()
            
            for item in results:
                triples_to_process = []
                
                if "triples" in item and isinstance(item["triples"], list):
                    triples_to_process = item["triples"]
                elif "subject" in item:
                    triples_to_process = [item]
                
                for triple in triples_to_process:
                    s = triple.get("subject")
                    p = triple.get("predicate")
                    o = triple.get("object")
                    if s: entities.add(s)
                    if p: predicates.add(p)
                    if o: entities.add(o)

            # 3. Generate Definitions (Async IO)
            predicates_list = list(predicates)
            definitions = []
            if predicates_list:
                try:
                    def_response = await async_collect_definition(
                        predicates_list,
                        self.definition_model,
                        **DEFINITION_PROMPT
                    )
                    def_map = {d.get("type", ""): d.get("definition", "") for d in def_response}
                    definitions = [def_map.get(p, "") for p in predicates_list]
                except Exception as e:
                    print(f"Definition generation error: {e}")
                    definitions = [""] * len(predicates_list)

            entities_list = [e for e in entities if e.strip()] # Loại bỏ entity rỗng
            
            if not entities_list:
                entities_embed = []
            else:
                entities_embed = self.embed_model.encode_sync(entities_list)
            
            if definitions:
                definition_embed = self.embed_model.encode_sync(definitions)
            else:
                definition_embed = []

            # 5. Resolution & Save (Async Lock)
            map_entities: Dict[str, str] = {}
            map_predicates: Dict[str, str] = {}

            async with self.db_lock:
                for entity, embed in zip(entities_list, entities_embed):
                    ans = EntityService.query(embed=embed, top_k=1)
                    
                    if len(ans) == 0 or ans[0][1] > self.threshold:
                        EntityService.add(name=entity, embedding=embed)
                        map_entities[entity] = entity
                    else:
                        map_entities[entity] = ans[0][0].name

                for (relation, definition), embed in zip(zip(predicates_list, definitions), definition_embed):
                    ans = RelationTypeService.query(embed=embed, top_k=1)
                    
                    if len(ans) == 0 or ans[0][1] > self.threshold:
                        RelationTypeService.add(type=relation, definition=definition, embedding=embed)
                        map_predicates[relation] = relation
                    else:
                        map_predicates[relation] = ans[0][0].type

            # 6. Remap & Export
            final_stages = []
            
            for item in results:
                new_item = item.copy()
                triples_list = []
                
                source_triples = []
                if "triples" in item and isinstance(item["triples"], list):
                    source_triples = item["triples"]
                elif "subject" in item:
                    source_triples = [item]
                
                for trip in source_triples:
                    new_trip = trip.copy()
                    if new_trip.get("subject") in map_entities:
                        new_trip["subject"] = map_entities[new_trip["subject"]]
                    if new_trip.get("predicate") in map_predicates:
                        new_trip["predicate"] = map_predicates[new_trip["predicate"]]
                    if new_trip.get("object") in map_entities:
                        new_trip["object"] = map_entities[new_trip["object"]]
                    triples_list.append(new_trip)
                
                if "triples" in item:
                    new_item["triples"] = triples_list
                    final_stages.append(new_item)
                else:
                    final_stages.append({"stage_name": "Table Extraction", "triples": triples_list})

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_file": str(input_path),
                    "stages": final_stages
                }, f, ensure_ascii=False, indent=2)
            
            print(f"Finished: {output_path} in {time.perf_counter() - start_time:.2f}s")

    async def run_batch(self, input_dir: str, output_dir: str, max_concurrency: int = 5):
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return

        print(f"Found {len(pdf_files)} files. Starting batch process...")
        
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = []

        for pdf_path in pdf_files:
            filename = Path(pdf_path).stem
            out_path = os.path.join(output_dir, f"{filename}.json")
            tasks.append(self.process_single_file(pdf_path, out_path, semaphore))

        await asyncio.gather(*tasks)
        print("Batch processing completed")

if __name__ == "__main__":
    # Cấu hình
    INPUT_DIR = "./raw_data"
    OUTPUT_DIR = "./knowledge_graph_output"
    MAX_WORKERS = 3 # Giảm worker xuống một chút vì Embedding chạy sync sẽ ăn CPU/GPU của main thread

    response_format = {
        "type": "json_object",
        "response_mime_type": "application/json",
        "response_schema": TripleList
    }
    
    extractor = TripleExtraction()
    
    kg_builder = KnowledgeGraphBuilder(
        triple_extraction=extractor,
        response_format=response_format,
        threshold=0.25
    )
    
    start_total = time.perf_counter()
    asyncio.run(kg_builder.run_batch(INPUT_DIR, OUTPUT_DIR, MAX_WORKERS))
    print(f"Total time: {time.perf_counter() - start_total:.2f}s")