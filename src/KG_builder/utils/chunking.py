"""Chunking strategy to divide text file into main topics."""

def extract_specific_sections(
    text: str,
    start_keyword: str,
    end_keyword: str
) -> dict[str, any]:
    """Extract paragraph start with start_keyword and end before end_keyword"""
    section: dict[str, str] = {}
    lines = text.split("\n")
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        has_start = start_keyword.lower() in line.lower()
        
        if has_start:
            # Start to extract paragraph
            section_lines = [line]
            section_start_idx = i
            i += 1
            
            found_end = False
            while i < len(lines):
                current_line = lines[i]
                
                # Check end_keyword
                has_end = end_keyword.lower() in current_line.lower()
                
                if has_end:
                    found_end = True
                    break
                
                section_lines.append(current_line)
                i += 1
            
            # Lưu section nếu tìm thấy end hoặc hết file
            section = {
                'content': '\n'.join(section_lines),
                'start_line': section_start_idx,
                'end_line': i - 1,
                'has_end_marker': found_end,
                'start_keyword': start_keyword,
                'end_keyword': end_keyword if found_end else None
            }
        else:
            i += 1
    return section