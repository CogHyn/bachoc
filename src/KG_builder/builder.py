from __future__ import annotations

import asyncio
import json

from pathlib import Path
from typing import Dict, Set, Any
from KG_builder.extract.extract_stage import TripleExtraction, Stage
from KG_builder.embedding.load.free import QwenEmbedding
from KG_builder.utils.llm_utils import load_async_model
from KG_builder.config import SECTIONS_DEFINITION
from KG_builder.prompts.prompts import DEFINITION_PROMPT
from KG_builder.utils.chunking import extract_specific_sections
from KG_builder.utils.llm_utils import load_model
from KG_builder.utils.clean_data import clean_vn_text
from KG_builder.extract.definition import async_collect_definition
from KG_builder.models.ops import RelationTypeService, EntityService
from KG_builder.triple_models import TripleList


class KnowledgeGraphBuilder:
    def __init__(self, *,
                triple_extraction: TripleExtraction,
                response_format: Dict[str, Any],
                threhold: float = 0.2,
                definition_model: str = "gemini-2.0-flash",
                llm_model: str = "gemini-2.0-flash",
                embedding_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.threshold = threhold
        self.response_format = response_format
        self.builder = triple_extraction
        self.llm = load_model(llm_model)
        self.definition_model = load_async_model(definition_model)
        self.embedding_name = embedding_model
        if "qwen" in self.embedding_name.lower():
            self.embed_model = QwenEmbedding(model_name=self.embedding_name)            

    def run(self, *,
            input_path: str,
            output_path: str):
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        # text = clean_vn_text(text)
        for section in SECTIONS_DEFINITION:
            # print(section)
            section_text = extract_specific_sections(text, section["start_word"], section["end_word"])
            print(section_text)
            print("-" * 100)
            self.builder.add_stage(
                stage=Stage(
                    text=section_text,
                    llm=self.llm,
                    predicates=section["predicates"],
                    response_format=self.response_format,
                    context=section["context"],
                    system_instruction=section["system_instruction"]
                )
            )
        
        result = self.builder.run()
        
        entities: Set[str] = set()
        predicates: Set[str] = set()
        
        # print(result)
        
        for stage in result:
            if not isinstance(stage, dict):
                continue
            for triple in stage["triples"]:
                if not isinstance(triple, dict):
                    continue
                _subject = triple.get("subject", None)
                _predicate = triple.get("predicate", None)
                _object = triple.get("object", None)
                entities.add(_subject)
                predicates.add(_predicate)
                entities.add(_object)
        
        
        print(predicates)
        response = asyncio.run(async_collect_definition(
            predicates,
            self.definition_model,
            **DEFINITION_PROMPT
        ))
        
        print(response)

        definitions = [item["definition"] for item in response]
        
        predicates = list(predicates)
        entities = list(entities)
        
        definition_embed = self.embed_model.encode_sync(definitions)
        
        # for definition in definitions:
        #     # print(definition)
        #     definition_embed.append(self.embed_model.encode_sync([definition]))
        
        entities_embed = self.embed_model.encode_sync(entities)
        
        map_predicates: Dict[str, str] = {}
        map_entities: Dict[str, str] = {}
        
        for i, (entity, embed) in enumerate(zip(entities, entities_embed)):
            ans = EntityService.query(embed=embed, top_k=1)
            if len(ans) == 0 or ans[0][1] > self.threshold:
               EntityService.add(
                   name = entity,
                   embedding=embed
               ) 
               map_entities[entity] = entity
               continue
           
            map_entities[entity] = ans[0][0].name
            
        for i, ((relation, definition), embed) in enumerate(zip(zip(predicates, definitions), definition_embed)):
            ans = RelationTypeService.query(embed=embed, top_k=1)
            if len(ans) == 0 or ans[0][1] > self.threshold:
                RelationTypeService.add(
                    type=relation,
                    definition=definition,
                    embedding=embed
                )
                map_predicates[relation] = relation
                continue
            
            map_predicates[relation] = ans[0][0].type

        
        for stage in result:
            if not isinstance(stage, list):
                continue
            for triple in stage["striple"]:
                if not isinstance(triple, dict):
                    continue
                
                triple["subject"] = map_entities[triple["subject"]]
                triple["predicate"] = map_predicates[triple["predicate"]]
                triple["object"] = map_entities[triple["object"]]
                
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "stages": result                
            }, f, ensure_ascii=False, indent=2)
                
                
        



# Backwards compatibility alias
KG_builder = KnowledgeGraphBuilder


if __name__ == "__main__":

    response_format = {
        "type": "json_object",
        "response_mime_type": "application/json",
        "response_schema": TripleList
    }
    
    extractor = TripleExtraction()
    
    builder = KnowledgeGraphBuilder(
        triple_extraction=extractor,
        response_format=response_format,
    )
    
    builder.run(
        input_path="./data/(17191314786197_30_06_2024_20_04)nguyen-van-tuan-1975-11-18-1719752669.txt",
        output_path="./output/nguyen-van-tuan.json"
    )
    