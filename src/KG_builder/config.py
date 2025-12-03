from KG_builder.prompts.prompts import (
    EXTRACT_TRIPLE_WORKING_INFO_PROMPT,
    EXTRACT_TRIPLE_PERSONAL_INFO_PROMPT,
    EXTRACT_TRIPLE_USER_PROMPT,
)

# Qwen parameters
DEVICE_MAP = "auto"
MAX_NEW_TOKENS = 3000
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.2

# Relation with definition
SECTION_PREDICATES_1 = {
    "Types": [
        "sinh ngày", "giới tính", "quốc tịch", "dân tộc", "tôn giáo", "là Đảng viên Đảng Cộng sản Việt Nam",
        "quê quán", "hộ khẩu thường trú", "địa chỉ liên hệ", "điện thoại nhà riêng", "điện thoại di động", 
        "email" 
    ],
    "Definitions": [
        "Specifies the exact date a person was born (Date of Birth).",
        "Specifies the person's gender.",
        "Specifies the country of which the person is legally a citizen (Nationality).",
        "Specifies the person's ethnic group.",
        "Specifies the person's religious affiliation or belief (Religion).",
        "Indicates the person's political affiliation status as a member of the Communist Party of Vietnam.",
        "Specifies the person's ancestral hometown or place of origin (Place of Origin).",
        "Specifies the official, registered permanent address (Permanent Residence Address).",
        "Specifies the current physical address used for correspondence or contact (Contact Address).",
        "Provides the fixed-line telephone number at the person's residence (Home Phone Number).",
        "Provides the person's personal cellular/mobile phone number (Mobile Phone Number).",
        "Provides the person's electronic mail address for contact (Email Address)."
    ]
}

SECTION_PREDICATES_2 = {
    "Types": [
        "học đại học tại", "học ngành", "công tác tại", "thực tập vị trí",
        "chuyên ngành", "thực tập tại", "nghiên cứu tại", "giữ chức vụ",
        "thỉnh giảng tại", "được cấp", "nghiên cứu vị trí", "công tác vị trí", "được cấp tại",
        "được bổ nhiệm chức danh PGS vào", "đăng ký xét đạt tiêu chuẩn chức danh Phó Giáo Sư tại HĐGS cơ sở",
        "Đăng ký xét đạt tiêu chuẩn chức danh Phó Giáo Sư tại HĐGS ngành, liên ngành",
        "chủ yếu nghiên cứu", "đã hướng dẫn bảo vệ thành công luận văn", "đã công bố", "đã hoàn thành đề tài NCKH từ cấp cơ sở trở lên", "là tác giả chính của",
        "đã xuất bản", "đã được cấp", "đạt giải thưởng quốc gia quốc tế", "được khen thưởng",
        "đã nghỉ hưu", "cơ quan công tác hiện nay", "có địa chỉ", "có số điện thoại", "tọa lạc ở",
        "chức vụ cao nhất"
    ],
    "Definitions": [
        "Indicates the university or institution where a person completed their undergraduate education.",
        "Specifies the academic discipline or field of study a person pursued during their education.",
        "Denotes the organization or institution where a person is or was employed.",
        "Specifies the area of expertise or specialization within a broader field of study or work.",
        "Specifies the formal title or role a person held while completing an internship at an organization.",
        "Indicates the organization or institution where a person completed a formal internship.",
        "Denotes the institution or facility where a person conducted their research activities.",
        "Specifies the formal title or role a person occupied within an organization.",
        "Indicates the institution where a person gave lectures on a non-permanent or temporary basis.",
        "Specifies the degree or certification received by a person.",
        "Specifies the formal title or role a person held while conducting research at an institution.",
        "Specifies the formal title or capacity a person occupied during their employment at an organization.",
        "Specifies the institution, council where the degree or certification received by a person",
        "Indicates the year or date when a person was formally given the academic title of Associate Professor.",
        "Indicates the local/institutional council where the person submitted their application for the Associate Professor title review.",
        "Indicates the national/specialized council where the person submitted their application for the Associate Professor title review.",
        "Specifies the main focus area or subject of a person's research activities.",
        "Indicates the quantity and level of students (Master's, Specialist II, PhD candidates) whose final thesis, dissertation, or specialized paper the subject has mentored and successfully defended",
        "Indicates the number of scientific articles, papers, or works a person has released.",
        "Indicates the quantity and level of formal Scientific Research Projects (NCKH - Nghiên cứu Khoa học) that the subject has successfully completed, starting from the institutional (grassroots) level up to national/international levels",
        "Specifies the number of publications where the person holds the primary authorship role.",
        "Indicates the number of books or specialized texts a person has released.",
        "Indicates the number of patents, useful solutions, or intellectual property rights a person has obtained.",
        "Denotes the number of national or international accolades or prizes received for artistic or athletic achievements.",
        "Indicates commendations, certificates of merit, or official recognition received.",
        "Indicates whether the subject has ceased working permanently, typically due to age or service length, and is receiving a pension",
        "Specifies the organization or institution where the person is currently employed or working.",
        "Provides the physical mailing or street address of a person, organization, company or entity.",
        "Provides the contact telephone number of a person, organization, company or entity.",
        "Specifies the broader geographical area (city, district, province, country) where an entity (typically a static institution or landmark) is situated",
        "Specifies the most senior or highest-ranking official title or administrative role a person has ever held during their professional career."
    ]
}

TABLE_PREDICATES = {
    "Types" : [
        "đã công bố bài báo", "có số tác giả", "là tác giả chính của", "là đồng tác giả của", "được đăng trên tạp chí", "xếp hạng tạp chí", 
        "số lần trích dẫn", "chi tiết xuất bản", "xuất bảo vào ngày", "giữ vai trò", "mã số", "cấp quản lý",
        "thực hiện trong giai đoạn", "nghiệm thu vào", "đạt xếp loại", "loại hình", "nhà xuất bản", "xuất bản năm",
        "là chủ biên", "đóng góp trang", "văn bản xác nhận", "được cấp bởi", "được cấp ngày", "văn bản giao nhiệm vụ",
        "được cấp phép bởi", "văn bản ứng dụng", "ghi chú", "đã sáng chế", "đã nghiệm thu nhiệm vụ khoa học", "đã đạt thành tích, có tác phẩm",
        "phát triển chương trình", "đã xuất bản"
    ]
}

PAPERS_PREDICATES_MAPPER = {
    "title": "đã công bố bài báo",
    "num_authors": "có số tác giả",
    "is_main_author": {"false": "đồng tác giả của", "true": "là tác giả chính của"},
    "journal_name_ISSN": "được đăng trên tạp chí",
    "journal_ranking": "xếp hạng tạp chí",
    "citation_count": "số lần trích dẫn",
    "volume_issue_pages": "chi tiết xuất bản",
    "published_date": "xuất bảo vào ngày"
}

PROJECTS_PREDICATES_MAPPER = {
    "title": "đã nghiệm thu nhiệm vụ khoa học",
    "role": "giữ vai trò",
    "project_code": "mã số",
    "management_level": "cấp quản lý",
    "period": "thực hiện trong giai đoạn",
    "acceptance_date": "nghiệm thu vào",
    "rating": "đạt xếp loại"
}

BOOKS_PREDICATES_MAPPER = {
    "title": "đã xuất bản",
    "type": "loại hình",
    "publisher": "nhà xuất bản",
    "publish_year": "xuất bản năm",
    "num_authors": "có số tác giả",
    "is_editor_in_chief": {"false": "thành viên biên soạn của", "true": "chủ biên"},
    "compiled_pages": "biên soạn trang",
    "verification_document_id": "văn bản xác nhận"
}

PATENTS_PREDICATES_MAPPER = {
    "title": "đã sáng chế",
    "issuing_authority": "được cấp bởi",
    "issue_date": "được cấp ngày",
    "is_main_inventor": {"false": "đồng tác giả của", "true": "là tác giả chính của"},
    "num_inventors": "có số tác giả"
}

ACHIEVEMENTS_PREDICATES_MAPPER = {
    "title": "đã đạt thành tích, có tác phẩm",
    "certifying_organization": "được công nhận bởi",
    "certificate_document_id": "có văn bản công nhận",
    "award": "đạt giải thưởng",
    "num_contributors": "có tổng số thành viên"
}

TRAINING_PROGRAMS_PREDICATES_MAPPER = {
    "title": "phát triển chương trình",
    "applicant_role": "giữ vai trò",
    "assignment_document_id": "văn bản giao nhiệm vụ",
    "certifying_authority": "được cấp phép bởi",
    "implementation_document_id": "văn bản ứng dụng",
    "notes": "ghi chú"
}

SECTIONS_DEFINITION = [
    {
        "context": EXTRACT_TRIPLE_USER_PROMPT,
        "system_instruction": EXTRACT_TRIPLE_PERSONAL_INFO_PROMPT,
        "start_word": "THÔNG TIN CÁ NHÂN",
        "end_word": "Quá trình công tác",
        "predicates": SECTION_PREDICATES_1
    },
    {
        "context": EXTRACT_TRIPLE_USER_PROMPT,
        "system_instruction": EXTRACT_TRIPLE_WORKING_INFO_PROMPT,
        "start_word": "Quá trình công tác",
        "end_word": "Trình độ đào tạo",
        "predicates": SECTION_PREDICATES_2
    },
    {
        "context": EXTRACT_TRIPLE_USER_PROMPT,
        "system_instruction": EXTRACT_TRIPLE_WORKING_INFO_PROMPT,
        "start_word": "Trình độ đào tạo",
        "end_word": "TỰ KHAI THEO",
        "predicates": SECTION_PREDICATES_2
    }
]
