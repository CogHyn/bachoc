# Kiến trúc pipeline

**Vấn đề**: Việc cố định pipeline bằng các method trong class -> Việc thực hiện thay đổi và thêm stage là rất khó khăn.

Vậy thay đổi design pattern như nào?

## Pipeline pattern

**Note**: Vẫn cố định pipeline ở bước đọc text và ghi kết quả.

**Note**: Tăng thông lượng xử lý.

**Problem**: Làm thế nào?

Vấn đề là chia pipeline đó liệu cách thức để tăng thông lượng là làm như thế nào? Nên thiết kế pipeline như nào? Làm sao để đảm bảo rằng việc để các stage hoạt động đúng đắn mà không bị rằng buộc lẫn nhau. Liệu pipeline phải có những điều gì để đảm bảo nó hoạt động đúng.

Những thành phần hiện tại của dự án đang khá rối.
Nó thực hiện không đồng nhất.


Có tài liệu nào về vấn đề này không?

- Mỗi stage là một thread riêng mà trong 1 stage có khả năng xử lý batch
- Làm sao để đảm bảo mỗi stage không bị lẫn.
- Thật ra nó cũng chẳng quan tâm rằng đoạn nội dung đó thuộc file nào?

**Pipeline**

**Model**

**Embedding Model**

**Storage**

Phải tuân thủ policy nào?

Hay chia mẹ thành 2 phần riêng biệt :)

Vậy việc tạo 1 cục text trước là nước đi cần thiết?

Đơn giản hơn cho quá trình tạo triple.

Tạo 1 tập triple đầu tiên từ bảng.

Nếu đã là bảng rồi thì consolid là không cần thiết :v 

Note: rõ ràng đang thiếu tài liệu để đọc.

Rõ ràng mình đang theo hướng edc?

Input(Text) -> Pipeline (LLM) -> Output (Triples) -> Consolidation -> Storage.

Nên cài đặt như nào? 

