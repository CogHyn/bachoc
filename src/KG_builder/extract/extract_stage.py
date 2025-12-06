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
import asyncio


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

    def _process_text_pipeline(self, llm, pdf_path: str, response_format):
        """
        Task 1: Xử lý toàn bộ phần Text -> Trả về triples và main_subject tìm được.
        Hàm này chạy tuần tự các stage text.
        """
        # 1. Extract & Clean Text
        self.stages = [] # Reset stages
        try:
            text = extract_context_from_pdf(pdf_path=pdf_path)
        except RuntimeError as e:
            print(f"[WARN] Cannot extract text from {pdf_path}: {e}")
            text = ""

        if not text:
            print(f"No text extracted from file.")
            return [], None

        cleaned_text = clean_vn_text(text)

        # 2. Build Stages
        for i, section in enumerate(SECTIONS_DEFINITION):
            section_text = extract_specific_sections(
                cleaned_text, section["start_word"], section["end_word"]
            )
            
            stage = Stage(
                text=section_text,
                llm=llm,
                predicates=section["predicates"],
                response_format=response_format,
                context=section["context"],
                system_instruction=section["system_instruction"]
            )
            self.add_stage(stage=stage)

        # 3. Execute Stages (Sequential dependency on main_subject)
        results = []
        current_main_subject = None

        for i, stage in enumerate(self.stages):
            if current_main_subject:
                stage.main_subject = current_main_subject
            
            # Blocking Call here
            result = stage.extract_triples()

            if result.get("main_subject"):
                current_main_subject = result.get("main_subject")
            results.append(result)
            
        return results, current_main_subject

    def _process_table_extraction(self, llm, pdf_path: str):
        """
        Task 2: Chỉ thực hiện việc trích xuất bảng (Nặng nhất).
        Chưa lưu file hay parse vội vì chưa có main_subject.
        """
        try:
            # Blocking Call here (OCR/Vision LLM)
            table_data_str = extract_table_from_pdf(
                pdf_path=pdf_path,
                genai=llm
            )
            return table_data_str
        except Exception as e:
            print(f"[WARN] Table extraction failed: {e}")
            return "[]"

    async def run_async(self, llm: BaseLLM, pdf_path: str, response_format):
        """
        Hàm chính chạy Async.
        """
        loop = asyncio.get_running_loop()

        # --- TẠO 2 TASK SONG SONG ---
        
        # Task 1: Chạy luồng Text trong Thread riêng (để không block event loop)
        text_task = loop.run_in_executor(
            None, 
            self._process_text_pipeline, 
            llm, pdf_path, response_format
        )

        # Task 2: Chạy luồng Table Extraction trong Thread riêng
        # Lưu ý: Task này không cần main_subject để chạy, chỉ cần pdf_path
        table_task = loop.run_in_executor(
            None, 
            self._process_table_extraction, 
            llm, pdf_path
        )

        # --- ĐỢI CẢ 2 TASK HOÀN THÀNH ---
        # Thời gian chạy = Max(thời gian text, thời gian table) thay vì Sum()
        (text_results, current_main_subject), table_data_str = await asyncio.gather(text_task, table_task)

        # --- HỢP NHẤT KẾT QUẢ (Post-processing) ---
        
        # Xử lý dữ liệu bảng sau khi đã có main_subject từ luồng Text
        if table_data_str and current_main_subject:
            try:
                parsed_data = json.loads(table_data_str)
                table_data_path = f"../table_data/{current_main_subject}.json"

                os.makedirs(os.path.dirname(table_data_path), exist_ok=True)
                with open(table_data_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, ensure_ascii=False, indent=4)

                # Extract triples từ file json vừa lưu
                table_triples = extract_triples_from_table(
                    table_data_path=table_data_path, 
                    main_subject=current_main_subject
                )
                text_results.extend(table_triples)
            except Exception as e:
                print(f"[ERR] Error processing table data for {current_main_subject}: {e}")

        return text_results

# --- PHẦN MAIN ĐỂ CHẠY ---
async def main():
    llm = load_model("gemini-2.5-flash") # Đảm bảo load_model thread-safe hoặc init trong mỗi thread nếu cần
    
    response_format = {
        "type": "json_object",
        "response_mime_type": "application/json",
        "response_schema": TripleList
    }
    
    builder = TripleExtraction()
    
    RAW_PATH = "/Users/huynhnguyen/WorkDir/bachoc_1/raw_data"
    
    start = time.perf_counter()
    
    # Lấy danh sách file
    files = [os.path.join(RAW_PATH, p) for p in os.listdir(RAW_PATH)[:100]]
    
    # Cách 1: Chạy tuần tự từng file (nhưng trong mỗi file thì Text/Table chạy song song)
    for pdf_path in files:
        start_file = time.perf_counter()
        results = await builder.run_async(llm=llm, pdf_path=pdf_path, response_format=response_format)
        print(f"File {pdf_path} done in {time.perf_counter() - start_file:.2f}s")

    # Cách 2 (Optional): Nếu muốn chạy song song NHIỀU FILE cùng lúc
    # tasks = [builder.run_async(llm, f, response_format) for f in files]
    # await asyncio.gather(*tasks) 
    
    print(f"Total time: {time.perf_counter() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())