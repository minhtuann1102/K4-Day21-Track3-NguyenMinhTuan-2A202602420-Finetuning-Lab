# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Minh Tuấn  **MSSV**: 2A202602420  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (split cố định seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json: mean 93.1, p50 93, p95 98, p99 100, max 101)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 max_steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: verdict `reasoning preserved — safe to train on traces`)*. Cả `open_tag_present` và `body_present` đều là true. Chuỗi render mẫu vẫn giữ nguyên khối `<think>` và thẻ đóng `</think>`, đảm bảo các vết suy luận (reasoning traces) không bị template âm thầm nuốt mất trong quá trình áp dụng template.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số token nằm trong vùng tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn prompt câu hỏi của user và system (`<|im_start|>system...`, `<|im_start|>user...`) được gán nhãn `-100` hoàn toàn chính xác, chứng minh loss mask chỉ giám sát phần sinh đầu ra của assistant.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3762.9 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1115.1 |
| (c) LoRA fine-tune | 0.9700 | 0.5444 | 1.0000 | 1585.7 |

**(b) có thật sự mạnh hơn (a) không?** Có — baseline (b) vượt trội hoàn toàn so với (a): độ chính xác target tăng từ 0.0 lên 0.765 (76.5%), định dạng JSON hợp lệ đạt 100% so với 0% của (a), đồng thời độ trễ giảm hơn 3 lần từ 3762.9 ms xuống 1115.1 ms do mô hình không còn sinh văn bản lan man ngoài cấu trúc.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa — giữ nguyên mã băm SHA `719e74d3b6232053` gốc của repo để đảm bảo tính liêm chính và sự công bằng tuyệt đối cho phép đối chứng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 (1e-4) | 0.6259 | **0.9700** | 457.6 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 (1e-4) | 0.5377 | **0.9700** | 286.2 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 (1e-5) | 1.5702 | **0.0000** | 459.3 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 (1e-4) | 0.7058 | **0.9400** | 565.2 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` được nâng rank lên $r=283$ nhằm khớp sát số lượng tham số huấn luyện với `correct` ($32,456,704$ so với $32,464,896$, độ lệch $< 0.03\%$). Trên tập đánh giá target ở NB5, `attn_only` đạt độ chính xác $0.9700$, ngang bằng (hoà) với `correct` ($0.9700$). Tuy nhiên, thứ tự này hoàn toàn trái ngược với thứ tự theo train loss: ở NB4, `attn_only` có train loss thấp hơn hẳn ($0.5377$ so với $0.6259$ của `correct`). Hiện tượng này cho thấy việc dồn một rank khổng lồ vào một vài module hẹp (chỉ các lớp attention $q, v$) giúp mô hình ép giảm loss nhanh hơn trên tập train nhờ việc ghi nhớ cục bộ, nhưng không mang lại bất kỳ sự vượt trội nào về khả năng thích ứng tác vụ thực tế trên tập kiểm thử so với việc phân bổ rank nhỏ ($r=16$) trải rộng trên toàn bộ các lớp linear (`text-linear`). Do đó, đòn bẩy quyết định nằm ở độ phủ vị trí gắn adapter thay vì việc cố gắng tăng rank tại một vài lớp riêng lẻ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` chỉ thay đổi duy nhất learning rate từ $1\times 10^{-4}$ xuống $1\times 10^{-5}$ (thang LR tiêu chuẩn khi thực hiện full fine-tuning). Đường loss của run này hầu như dậm chân tại chỗ và kết thúc ở mức $1.5702$, cao hơn gấp $2.5$ lần so với mức $0.6259$ của `correct`, dẫn tới việc mô hình hoàn toàn không học được định dạng JSON và nhận điểm target $0.0000$. Nếu một người chỉ quan sát loss phẳng lì và kết quả $0.0$ mà không biết về learning rate, họ sẽ rất dễ đưa ra kết luận sai lầm rằng: "mô hình không đủ năng lực", "bài toán trích xuất JSON tiếng Việt quá phức tạp", hoặc "kỹ thuật LoRA không hiệu quả đối với bài toán này". Trong thực tế, vì LoRA chỉ cập nhật một ma trận bổ sung có số tham số cực kỳ nhỏ so với mô hình gốc, nó đòi hỏi learning rate lớn hơn gấp $5$ đến $10$ lần so với full fine-tuning để gradient step có thể dịch chuyển đủ xa trong không gian biểu diễn.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
QLoRA (lượng tử hóa base model về 4-bit NF4) tiết kiệm lượng VRAM vô cùng ấn tượng: bộ nhớ đỉnh giảm từ $8.78\text{ GB}$ xuống chỉ còn $3.86\text{ GB}$ (giảm hơn $56\%$). Tuy nhiên, sự đánh đổi là thời gian huấn luyện tăng thêm đáng kể từ $457.6\text{ s}$ lên $565.2\text{ s}$ (chậm hơn khoảng $23.5\%$ do chi phí giải nén lượng tử on-the-fly trong từng forward/backward pass), đồng thời độ chính xác target bị tụt giảm từ $0.9700$ xuống $0.9400$. Các số liệu thực nghiệm này hoàn toàn ủng hộ khuyến nghị của đội ngũ phát triển Qwen ("không khuyến khích dùng QLoRA cho dòng Qwen3.5") nếu phần cứng của bạn đã đủ VRAM chứa bản 16-bit (như T4 16GB). Việc ép lượng tử hóa 4-bit gây thất thoát thông tin biểu diễn làm suy giảm độ chính xác trích xuất trường mà không mang lại lợi thế về tốc độ xử lý.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.247` · `valid_trace_rate = 0.00`

Diễn giải:
Mặc dù bản fine-tune thể hiện sự thăng tiến vượt bậc trên nhiệm vụ đích với mức tăng trưởng target $\Delta = +0.205$ (đạt độ chính xác $0.970$ so với $0.765$ của baseline prompt tối ưu), hệ thống đánh giá vẫn đưa ra phán quyết FAILED từ cổng kiểm soát chất lượng hồi quy. Nguyên nhân cốt lõi là chỉ số kiểm tra năng lực tổng quát (regression capability) bị suy giảm nghiêm trọng từ $0.7911$ xuống $0.5444$ ($\Delta = -0.247$, vượt xa ngưỡng dung sai an toàn cho phép là $0.020$).

Hiện tượng này phản ánh trực tiếp vấn đề quên thảm họa (catastrophic forgetting) kinh điển trong quá trình fine-tuning mô hình ngôn ngữ lớn. Khi toàn bộ 225 mẫu huấn luyện chỉ tập trung thuần túy vào cấu trúc JSON của ticket khiếu nại chăm sóc khách hàng mà không có bất kỳ mẫu dữ liệu tổng quát nào được xen kẽ, các trọng số LoRA đã vô tình bóp méo không gian biểu diễn tri thức nền tảng của mô hình. Thất bại này mang ý nghĩa thực tiễn to lớn: nó chứng minh rằng việc chỉ đánh giá đơn chiều trên tập target là một cái bẫy nguy hiểm. Để khắc phục và đưa mô hình qua cổng hồi quy một cách an toàn, quy trình huấn luyện bắt buộc phải áp dụng kỹ thuật replay buffer — tức trộn thêm $1\%\text{–}5\%$ dữ liệu hướng dẫn tổng quát vào tập huấn luyện theo đúng khuyến nghị tại Chương 5 (deck §6.3).

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Nhầm `urgency: trung_binh` do khách có câu "Cảm ơn shop" | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | ✅ FT thắng: FT phân tách tốt ý định trả hàng gấp với thái độ lịch sự. |
| 2 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp. | `san_pham_loi`, `cao`, `đèn bàn LED`, `trung_tinh` | Nhầm sang `intent: doi_tra` do liên quan vỡ hàng | `{"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung_tinh"}` | ✅ FT thắng: Bắt chính xác lỗi sản phẩm bị vỡ và độ khẩn cấp cao. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `{"intent": "hoan_tien", "urgency": "thap", ...}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | ❌ **FT thua**: FT đoán nhầm `urgency: trung_binh` thay vì `thap`. |
| 4 | Chào shop, mình đặt nồi chiên không dầu mã đơn VN949966. Hoàn tiền. Khi nào tiện cũng được. | `hoan_tien`, `thap`, `nồi chiên không dầu`, `tieu_cuc` | `{"intent": "hoan_tien", "urgency": "thap", ...}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "tieu_cuc"}` | ❌ **FT thua**: FT tiếp tục nhầm mức độ khẩn cấp `urgency: trung_binh`. |
| 5 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop. | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | Định dạng dài dòng, nhầm `sentiment: tieu_cuc` do từ "quá hạn" | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | ✅ FT thắng: Trích xuất JSON gọn gàng, nhận diện đúng cả 4 trường. |

Có mẫu chung nào ở các ca FT thua không?
Có một quy luật chung rất rõ ràng: ở tất cả các ca fine-tune thua (điểm đạt 0.75 do sai trường `urgency`), mô hình fine-tune đều có xu hướng quy chụp mức độ khẩn cấp về `trung_binh` bất chấp việc khách hàng đã thể hiện rõ ràng sắc thái thong thả bằng các cụm từ như *"Khi nào tiện cũng được"*, *"Không vội nhé"*. Nguyên nhân là do trong tập huấn luyện nhỏ (225 mẫu), nhãn `urgency: trung_binh` chiếm tỷ trọng áp đảo, khiến mô hình LoRA bị thiên kiến xác suất (prior bias). Trong khi đó, baseline (b) có các câu mô tả định nghĩa chi tiết trong system prompt kèm in-context examples nên nắm bắt tốt hơn các trường hợp phân loại ở biên.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ):**
Dựa trên kết quả thực nghiệm toàn diện, câu trả lời dứt khoát là: **Chưa nên đưa bản fine-tune này vào môi trường production ngay lập tức.** Mặc dù mô hình mang lại sự cải thiện ấn tượng trên bài toán nghiệp vụ đích (độ chính xác target đạt $97.0\%$, vượt trội so với mức $76.5\%$ của baseline prompt tối ưu và $0\%$ của prompt ngây thơ), nó lại thất bại ở cổng kiểm định hồi quy khi đánh mất $24.7\%$ năng lực tổng quát (regression tụt từ $0.7911$ xuống $0.5444$). Nếu deploy bản checkpoint này cho một hệ thống chatbot đa năng, mô hình sẽ gặp lỗi nghiêm trọng khi người dùng đặt các câu hỏi kiến thức hoặc chỉ dẫn ngoài miền triage.

Bài học cốt lõi từ lab là đòn bẩy thực sự của fine-tuning không nằm ở việc cố gắng nâng rank adapter lên cực đại hay chọn thuật toán phức tạp, mà nằm ở **tính toàn vẹn của Loss Mask**, **Learning Rate phù hợp với quy mô LoRA**, và **chiến lược phân bổ dữ liệu**. Thí nghiệm ở NB4 đã chứng minh một rank khổng lồ ($r=283$) tại attention không hề thắng được rank nhỏ ($r=16$) trải rộng toàn bộ lớp tuyến tính, và learning rate sai lệch một bậc độ lớn sẽ phá hỏng hoàn toàn tiến trình huấn luyện. Để có thể deploy an toàn, bước đi kế tiếp bắt buộc phải là bổ sung tập replay buffer để khôi phục chỉ số regression trước khi phát hành.

**Ba điều tôi học được (cụ thể, không generic):**
1. **Kiểm chứng Loss Mask bằng việc giải mã ngược token:** Không bao giờ được tin tưởng mù quáng vào hàm mask mặc định. Việc giải mã ngược các token được tính loss (và token bị gán nhãn `-100`) là cách duy nhất để chứng minh mô hình chỉ học câu trả lời của trợ lý thay vì học vẹt lại chính câu hỏi của người dùng.
2. **Thứ tự theo Train Loss không đồng nghĩa với Thứ tự theo Target Metric:** Run `attn_only` có train loss thấp hơn `correct` ($0.5377$ vs $0.6259$), nhưng điểm target thực tế lại chỉ ngang bằng ($0.9700$). Đánh giá mô hình bằng chỉ số thay thế (loss) là sai lầm nguy hiểm nhất; chỉ có phép đo trên tác vụ đích mới phản ánh năng lực thực.
3. **Phán quyết FAILED vẫn là một kết quả nghiên cứu khoa học giá trị:** Việc mô hình trượt cổng hồi quy (quên kiến thức chung) giúp phát hiện rủi ro catastrophic forgetting sớm trước khi ra production. Một kết luận FAILED được phân tích nguyên nhân nhân quả sâu sắc có giá trị thực tiễn cao hơn nhiều so với việc cố tình làm yếu baseline hoặc nới lỏng cổng để tạo ra một chiến thắng ảo.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ thử nghiệm trộn $3\%\text{–}5\%$ dữ liệu từ bộ Alpaca hoặc ShareGPT tiếng Việt vào tập huấn luyện 225 mẫu để tạo replay buffer, sau đó đánh giá lại xem liệu mô hình có vừa giữ được $97\%$ target accuracy vừa khôi phục điểm regression $\ge 0.79$ để chính thức nhận phán quyết `PASSED` hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
