# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Với mức 0.0, câu trả lời qua mỗi phiên được giữ nguyên, chứng tỏ cách mô hình lựa chọn token kế tiếp là deterministic tức cố định. Với mức 0.7, mô hình bắt đầu trở nên sáng tạo hơn, có vẻ xác suất của token tiếp theo không bị cố định, mà có vẻ công bằng hơn với các token có xác suất thấp. Với mức 1.2 và 1.8, mô hình trở nên sáng tạo hơn khá nhiều, câu trả lời vô cùng đa dạng, nhưng tính nhất quán không cao và mô hình có dấu hiệu halluciante.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với hợp đồng pháp lý, cần một mô hình có tính nhất quán, mạch lạc, tôi sẽ đặt từ 0.0 -> 0.7, mức vừa đủ để mô hình vẫn có thể linh hoạt nhưng câu trả lời vẫn ở mức độ kiểm soát được, không bịa quá nhiều. Với slogan quảng cáo, mức 0.8->1.2 là ổn, mang tính sáng tạo cao, có thể tăng lên 1.8 hoặc cao hơn nhưng nên cân nhắc đến các yếu tố khác như chi phí.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Với ~500 token đầu ra, sẽ có ~50 token đầu vào, dựa trên thống kê qua 20 lần chạy model gemini2.5-flash. Vậy chi phí cho workload này sẽ là ~50.6$ mỗi ngày. với model nhỏ, ta có tương đương số token như trên, ước tính rơi vào 8$ - 10$ cho workload này mỗi ngày. Model lớn sẽ phù hợp trong trường hợp xây dựng một chatbot AI cho website giả sử giáo dục, hoặc y tế cần đưa ra câu trả lời ở mức có thể chấp nhận được cho đến hoàn hảo. Model nhỏ sẽ phù hợp cho những việc như ngăn chặn prompt injection, đặt là một bước nhỏ trong workflow để phê duyệt, ngăn chặn input/output nếu đi lệch voiwsm mục đích của mô hình chính. 

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Hai trường hợp cho hai cách giải thích hoàn toàn khác nhau, chứng tỏ mô hình LLM có cá tính và có thể được lập trình. Một nhà thơ sẽ cho câu trả lời bay bổng thơ văn, không quá trừu tượng và sử dụng khá nhiều biện pháp nghệ thuật. Kỹ sư phần mềm thì giải thích ở một khía cạnh đầy tính kỹ thuật, giọng văn khá cứng nhắc và đôi lúc sẽ giải thích hơi khó hiểu, phù hợp cho người học kỹ thuật. System prompt định hình cá tính và cách trả lời của mô hình, ngoài ra còn có thể điều khiển độ dài của câu trả lời, tùy thuộc vào cách chúng ta cài cá tính của mô hình cụ thể tới đâu và mô hình được tiền huấn luyện để "nhập vai" tới mức nào.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> So sánh giữa 3 đoạn văn có độ dài khác như từ 100 từ đến 200 từ đến ~1000 từ, có thể thấy số từ càng lớn thì sự chênh lệch token giữa hai cách đếm từ càng lớn, với tiếng Việt, từ dấu câu đến ô trống và ký tự là rất nhiều token được thêm vào, hệ thống ngôn ngữ Tiếng Việt vô cùng đa dạng vậy cho nên số token thường tăng theo cấp số nhân khi số từ tăng đối với mỗi output. Vậy nên ta cần dự toán ngân sách dư ra đôi chút.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> chatbot văn bản sẽ hưởng lợi hơn hai trường hợp kia, đối với c không cần thiết cho llm output token và streaming ra do tại thời điểm real time ấy thì không có người dùng nào thật sự đang dùng, b cũng không cần do việc cho một speech - llm streaming sẽ khiến câu trả lời không liền mạch, như khi nghe một người đọc từng từ một thay vì đọc cả câu vậy.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Với delay cố định, việc đặt một request sử dụng một thời gian cố định gần như không giải quyết được vấn đề quá tải, giống như lấy 1000 viên bi đút vào một chai nước, nhưng không thể vào thì chờ 5s sau lại đút tiếp 1000 viên bi ấy vào cùng lúc. Exponential backoff mỗi khi rq thất bại sẽ tăng thời gian chờ theo cấp số nhân, giúp các rq có thể được xử lý một cách tuần tự, vừa phải với kích cỡ của cổ chai nước ấy. Mặc dù exp backoff có thể giảm tải số lượng rq bằng cách tăng thời gian chờ của các rq ấy, thế nhưng ở trên trục thời gian tại các mốc thời gian vẫn sẽ có những cột request khổng lồ, jitter thay vì tất cả đều vào lúc giây thứ 4 sẽ đặt các rq ở một cách ngẫu nhiên, giả sử 3.8s, 4s... giúp giải quyết vấn đề quá tải mà exp backoff chưa giải quyết triệt để. 

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> Bạn là ai và câu trả lời của bạn có tính chất thế nào. Nếu không trả lời được hai câu hỏi này thì hành vi của trợ lý sẽ thay đổi rõ rệt. 

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Giả sử câu hỏi về một chuyến du lịch, bạn cung cấp thông tin cho chatbot ở câu đầu tiên rằng bạn bị dị ứng hải sản, và ở 3 câu tiếp theo bạn không hề đề cập hay nhắc lại về vấn đề ấy với chatbot, ở câu thứ 5 khi bạn xin đề xuất nhà hàng, chatbot sẽ đề xuất cho bạn những nhà hàng sushi chẳng hạn hoặc nhà hàng chuyên món mà bạn bị dị ứng.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
