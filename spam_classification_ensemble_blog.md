# 🚀 Xây dựng Hệ thống Phân loại Email Spam Thông minh với Ensemble k-NN và Embedding

> *Hướng dẫn chi tiết cách kết hợp Transformer Embeddings, FAISS Vector Search và Ensemble Methods để tạo ra một hệ thống lọc spam hiệu quả và thông minh*

---

## 📖 Mục lục

1. [Giới thiệu về bài toán Email Spam](#1-giới-thiệu-về-bài-toán-email-spam)
2. [Hiểu về Sentence Embeddings và Transformer](#2-hiểu-về-sentence-embeddings-và-transformer)
3. [FAISS: Công nghệ tìm kiếm Vector siêu nhanh](#3-faiss-công-nghệ-tìm-kiếm-vector-siêu-nhanh)
4. [Thuật toán k-Nearest Neighbors cho phân loại văn bản](#4-thuật-toán-k-nearest-neighbors-cho-phân-loại-văn-bản)
5. [Ensemble Methods: Nghệ thuật kết hợp nhiều mô hình](#5-ensemble-methods-nghệ-thuật-kết-hợp-nhiều-mô-hình)
6. [Thực nghiệm và phân tích kết quả](#6-thực-nghiệm-và-phân-tích-kết-quả)
7. [Tổng kết và định hướng phát triển](#7-tổng-kết-và-định-hướng-phát-triển)

---

## 1. Giới thiệu về bài toán Email Spam

### 🎯 Tại sao cần lọc Email Spam?

Mỗi ngày có hàng tỷ email được gửi đi trên toàn thế giới, và một tỷ lệ lớn trong số đó là spam (thư rác). Việc tự động phát hiện và lọc spam không chỉ giúp người dùng tiết kiệm thời gian, mà còn bảo vệ họ khỏi các mối đe dọa bảo mật.

### 🔍 Những thách thức chính:

- **Đa dạng về nội dung**: Email spam có thể xuất hiện dưới rất nhiều hình thức khác nhau
- **Hiểu ngôn ngữ tự nhiên**: Cần phải hiểu được ý nghĩa của câu văn, không chỉ đơn thuần tìm từ khóa
- **Xử lý thời gian thực**: Phải phân loại nhanh để không làm chậm hệ thống email
- **Độ chính xác cao**: Tránh nhầm lẫn email quan trọng thành spam (sai lầm nghiêm trọng!)

### 💡 Ý tưởng giải pháp thông minh:

Thay vì sử dụng các phương pháp cũ như đếm từ khóa hay phân tích tần suất từ, chúng ta sẽ:

1. **Sử dụng Transformer để hiểu ngữ nghĩa** - giống như cách con người đọc và hiểu email
2. **Áp dụng FAISS để tìm kiếm nhanh** - tìm những email tương tự trong cơ sở dữ liệu
3. **Kết hợp nhiều thuật toán k-NN** - tăng độ tin cậy của dự đoán
4. **Tạo điểm tin cậy** - biết được mô hình "chắc chắn" đến mức nào

---

## 2. Hiểu về Sentence Embeddings và Transformer

### 🧠 Tại sao cần chuyển đổi văn bản thành số?

Máy tính không thể hiểu được văn bản như con người. Để máy tính có thể "hiểu" được ý nghĩa của câu văn, chúng ta cần chuyển đổi chúng thành các con số:

```python
# Từ văn bản sang ý nghĩa dưới dạng số
"Trúng thưởng miễn phí ngay!" → [0.1, -0.3, 0.8, ..., 0.2]  # Vector 768 chiều
"Chào bạn thân mến"           → [0.4, 0.2, -0.1, ..., 0.5]  # Vector 768 chiều
```

### 🤖 Mô hình Multilingual E5 - Trợ lý thông minh

Chúng ta sử dụng **intfloat/multilingual-e5-base** - một mô hình embedding tiên tiến:

**Điểm mạnh:**
- **Đa ngôn ngữ**: Hiểu được cả tiếng Việt và tiếng Anh
- **Chất lượng cao**: Được huấn luyện trên khối lượng dữ liệu khổng lồ
- **Hiểu ngữ nghĩa sâu**: Nắm bắt được ý nghĩa thật sự của câu văn
- **Xử lý nhanh**: Tốc độ phù hợp cho ứng dụng thực tế

### 🔧 Chi tiết cách triển khai

```python
# Tải mô hình
TEN_MO_HINH = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(TEN_MO_HINH)
model = AutoModel.from_pretrained(TEN_MO_HINH)

# Kỹ thuật quan trọng: Pooling trung bình
def tinh_trung_binh_pooling(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Tạo embedding với tiền tố phù hợp
def tao_embeddings(cac_van_ban, model, tokenizer, device, batch_size=32):
    # Quan trọng: Sử dụng tiền tố "passage:" cho dữ liệu huấn luyện
    van_ban_co_tien_to = [f"passage: {van_ban}" for van_ban in cac_van_ban]
    # Sử dụng tiền tố "query:" cho văn bản cần tìm kiếm
    cau_truy_van = f"query: {van_ban_truy_van}"
```

### 🎯 Chiến lược sử dụng tiền tố

Mô hình E5 yêu cầu sử dụng tiền tố để phân biệt:
- **`"passage:"`** cho các tài liệu trong cơ sở dữ liệu
- **`"query:"`** cho văn bản cần tìm kiếm

Điều này giúp mô hình hiểu được ngữ cảnh và cải thiện chất lượng tìm kiếm.

---

## 3. FAISS: Công nghệ tìm kiếm Vector siêu nhanh

### ⚡ Tại sao cần FAISS?

Khi có hàng triệu email, việc so sánh từng email một sẽ cực kỳ chậm:

```python
# Cách làm thông thường - độ phức tạp O(n) - rất chậm!
for email in co_so_du_lieu:
    do_tuong_tu = tinh_cosine_similarity(vector_truy_van, vector_email)
    
# Cách làm với FAISS - độ phức tạp O(log n) - siêu nhanh!
do_tuong_tu, chi_so = index.search(vector_truy_van, k=5)
```

### 🏗️ Kiến trúc FAISS

**FAISS** (Facebook AI Similarity Search) là thư viện do Meta phát triển để tìm kiếm tương tự trên các vector quy mô lớn:

**Tính năng nổi bật:**
- **Tăng tốc bằng GPU**: Tận dụng sức mạnh GPU để xử lý nhanh hơn
- **Tiết kiệm bộ nhớ**: Tối ưu hóa việc sử dụng bộ nhớ
- **Nhiều thuật toán**: IndexFlat, IndexIVF, IndexHNSW và nhiều hơn nữa
- **Tìm kiếm chính xác và xấp xỉ**: Cân bằng giữa tốc độ và độ chính xác

### 🔧 Cách triển khai

```python
# Tạo chỉ mục FAISS
so_chieu_embedding = X_train_emb.shape[1]  # 768 chiều
index = faiss.IndexFlatIP(so_chieu_embedding)  # Tương tự Inner Product
index.add(X_train_emb.astype("float32"))

# Tìm kiếm các vector tương tự
diem_so, chi_so = index.search(query_embedding, k=5)
```

### 📊 Tại sao chọn IndexFlatIP thay vì IndexFlatL2

Chúng ta chọn **IndexFlatIP** (Inner Product) thay vì **IndexFlatL2** (khoảng cách L2) vì:

- **Inner Product** phù hợp với các vector đã được chuẩn hóa
- **Độ tương tự cosine** = Inner Product khi các vector được chuẩn hóa
- **Tương tự ngữ nghĩa** được nắm bắt tốt hơn

---

## 4. Thuật toán k-Nearest Neighbors cho phân loại văn bản

### 🎯 Ý tưởng cơ bản của k-NN

k-NN dựa trên một giả định đơn giản nhưng mạnh mẽ: **"Những thứ giống nhau thường thuộc cùng một loại"**

```python
# Cho một email mới
cau_truy_van = "Trúng thưởng tiền miễn phí ngay!"

# Tìm k email tương tự nhất trong tập huấn luyện
cac_hang_xom = [
    {"van_ban": "Nhận tiền thưởng miễn phí!", "nhan": "spam", "do_tuong_tu": 0.89},
    {"van_ban": "Click để trúng tiền", "nhan": "spam", "do_tuong_tu": 0.85},
    {"van_ban": "Chúc mừng bạn trúng thưởng", "nhan": "spam", "do_tuong_tu": 0.82}
]

# Bỏ phiếu theo đa số: 3/3 = spam → Dự đoán: SPAM
```

### 🔧 Chi tiết cách triển khai

```python
def phan_loai_voi_knn(van_ban_truy_van, model, tokenizer, device, index, metadata_train, k=1):
    # Bước 1: Chuyển văn bản thành embedding
    truy_van_co_tien_to = f"query: {van_ban_truy_van}"
    embedding_truy_van = tao_embedding(truy_van_co_tien_to)
    
    # Bước 2: Tìm k hàng xóm gần nhất bằng FAISS
    diem_so, chi_so = index.search(embedding_truy_van, k)
    
    # Bước 3: Lấy nhãn từ các hàng xóm
    cac_du_doan = []
    for i in range(k):
        chi_so_hang_xom = chi_so[0][i]
        nhan_hang_xom = metadata_train[chi_so_hang_xom]["nhan"]
        cac_du_doan.append(nhan_hang_xom)
    
    # Bước 4: Bỏ phiếu theo đa số
    du_doan_cuoi_cung = pho_bien_nhat(cac_du_doan)
    return du_doan_cuoi_cung
```

### 📈 Chọn giá trị K phù hợp

**Lựa chọn K** là yếu tố quan trọng:

- **K=1**: Nhạy cảm với nhiễu, có thể quá khớp dữ liệu
- **K=3,5,7**: Lựa chọn cân bằng, tốt cho hầu hết trường hợp  
- **K=lớn**: Quá mượt mà, có thể bỏ sót chi tiết quan trọng

**Thực hành tốt nhất:** Thử nghiệm nhiều giá trị k và chọn giá trị tối ưu trên tập validation.

---

## 5. Ensemble Methods: Nghệ thuật kết hợp nhiều mô hình

### 🌟 Tại sao cần Ensemble?

Một bộ phân loại k-NN đơn lẻ có những hạn chế:

- **K cố định**: Không linh hoạt cho các loại truy vấn khác nhau
- **Không có độ tin cậy**: Không biết mô hình "chắc chắn" đến mức nào
- **Nhạy cảm với lựa chọn k**: Hiệu suất phụ thuộc nhiều vào giá trị k

**Giải pháp Ensemble:** Kết hợp nhiều bộ phân loại k-NN với các giá trị k khác nhau!

### 🏗️ Kiến trúc Ensemble

```python
# Tạo nhiều bộ phân loại
cac_bo_phan_loai = [
    BoPhanLoaiKNN(index, metadata, k=1),   # Chính xác nhưng nhạy cảm
    BoPhanLoaiKNN(index, metadata, k=3),   # Cân bằng
    BoPhanLoaiKNN(index, metadata, k=5),   # Mượt mà nhưng ổn định
]

# Lấy dự đoán từ tất cả bộ phân loại
cac_du_doan = ["spam", "ham", "spam"]     # Dự đoán riêng lẻ
cac_do_tin_cay = [0.9, 0.6, 0.8]         # Độ tin cậy riêng lẻ

# Kết hợp bằng phương pháp ensemble
du_doan_cuoi, do_tin_cay_cuoi = ket_hop_ensemble(cac_du_doan, cac_do_tin_cay)
```

### 🎯 Bộ phân loại KNN với điểm tin cậy

```python
class BoPhanLoaiKNN:
    def du_doan_voi_do_tin_cay(self, embedding_truy_van, k=3):
        # Lấy k hàng xóm gần nhất
        diem_so, chi_so = self.index.search(embedding_truy_van, k)
        
        # Thu thập dự đoán
        cac_du_doan = [metadata_train[idx]["nhan"] for idx in chi_so[0]]
        
        # Bỏ phiếu theo đa số
        nhan_du_doan = pho_bien_nhat(cac_du_doan)
        
        # Tính độ tin cậy
        do_tin_cay_phieu_bau = dem(nhan_du_doan) / k  # Sức mạnh phiếu bầu
        do_tuong_tu_trung_binh = trung_binh(diem_so[0])  # Sức mạnh tương tự
        
        # Độ tin cậy kết hợp (có trọng số)
        do_tin_cay_cuoi = do_tin_cay_phieu_bau * 0.6 + do_tuong_tu_trung_binh * 0.4
        
        return nhan_du_doan, do_tin_cay_cuoi
```

### 🎛️ Ba phương pháp Ensemble

#### 1. **Bỏ phiếu có trọng số** 🏆

Mỗi bộ phân loại bỏ phiếu với trọng số bằng độ tin cậy của nó:

```python
# Ví dụ:
# Bộ phân loại 1 (k=1): "spam" với độ tin cậy 0.9
# Bộ phân loại 2 (k=3): "ham"  với độ tin cậy 0.6
# Bộ phân loại 3 (k=5): "spam" với độ tin cậy 0.8

phieu_bau_theo_nhan = {
    "spam": 0.9 + 0.8 = 1.7,
    "ham":  0.6 = 0.6
}
# Kết quả: "spam" (vì 1.7 > 0.6)
```

**Ưu điểm:** Tận dụng thông tin độ tin cậy, bền vững với dự đoán nhiễu.

#### 2. **Độ tin cậy tối đa** 🎯

Tin theo bộ phân loại tự tin nhất:

```python
cac_do_tin_cay = [0.9, 0.6, 0.8]
chi_so_max = chi_so_lon_nhat(cac_do_tin_cay) = 0
du_doan_cuoi = cac_du_doan[0] = "spam"
```

**Ưu điểm:** Đơn giản, hiệu quả khi có 1 bộ phân loại rất mạnh.

#### 3. **Độ tin cậy trung bình** ⚖️

Bỏ phiếu theo đa số + độ tin cậy trung bình:

```python
# Bước 1: Bỏ phiếu theo đa số → "spam" (2/3 phiếu)
# Bước 2: Độ tin cậy trung bình của các phiếu "spam"
cac_do_tin_cay_khop = [0.9, 0.8]  # Từ các bộ phân loại dự đoán "spam"
do_tin_cay_cuoi = trung_binh(cac_do_tin_cay_khop) = 0.85
```

**Ưu điểm:** Cách tiếp cận thận trọng, ổn định với các dự đoán đa dạng.

### 📊 Khung đánh giá Ensemble

```python
def danh_gia_do_chinh_xac_ensemble(embeddings_test, metadata_test, cac_bo_phan_loai, 
                                  cac_cau_hinh_k, cac_phuong_phap_ensemble):
    # Thử nghiệm nhiều cấu hình
    cac_cau_hinh_k = [
        [1, 3, 5],    # Thận trọng: giá trị k nhỏ
        [3, 5, 7],    # Vừa phải: giá trị k trung bình
        [5, 7, 9],    # Tự do: giá trị k lớn
        [1, 5, 9],    # Hỗn hợp: giá trị k đa dạng
    ]
    
    # Thử tất cả phương pháp ensemble
    for cau_hinh in cac_cau_hinh_k:
        for phuong_phap in ["bo_phieu_co_trong_so", "do_tin_cay_toi_da", "do_tin_cay_trung_binh"]:
            do_chinh_xac, do_tin_cay_tb = kiem_tra_ensemble(cau_hinh, phuong_phap)
            print(f"Cấu hình {cau_hinh} + {phuong_phap}: {do_chinh_xac:.4f} độ chính xác")
```

---

## 6. Thực nghiệm và phân tích kết quả

### 📊 Tổng quan về dữ liệu

- **Kích thước**: 558 mẫu kiểm tra
- **Phân loại**: Ham (email thật) vs Spam (email rác)
- **Chia tách**: 90% huấn luyện, 10% kiểm tra
- **Ngôn ngữ**: Chủ yếu tiếng Anh với một số tiếng Việt

### 🔬 Thiết lập thí nghiệm

```python
# Cấu hình mô hình
TEN_MO_HINH = "intfloat/multilingual-e5-base"
SO_CHIEU_EMBEDDING = 768
TY_LE_KIEM_TRA = 0.1
SEED = 42

# Các chỉ số đánh giá
cac_chi_so = [
    "Độ chính xác",        # Tỷ lệ dự đoán đúng
    "Độ tin cậy trung bình", # Độ tin cậy trung bình
    "Phân tích lỗi",       # Phân tích lỗi chi tiết
]
```

### 📈 Kết quả k-NN đơn lẻ

| Giá trị k | Độ chính xác | Số lỗi | Nhận xét |
|---------|----------|--------|--------|
| k=1     | 98.57%   | 8/558  | Chính xác cao, nhạy cảm với nhiễu |
| k=3     | 99.28%   | 4/558  | **Hiệu suất đơn lẻ tốt nhất** |
| k=5     | 99.10%   | 5/558  | Ổn định, độ chính xác giảm nhẹ |
| k=7     | 98.92%   | 6/558  | Bảo thủ, tốt cho dữ liệu nhiễu |
| k=9     | 98.75%   | 7/558  | Quá mượt, mất độ chính xác |

### 🏆 Kết quả Ensemble

#### Các cấu hình hiệu suất cao nhất:

| Thứ hạng | Cấu hình | Phương pháp | Độ chính xác | Độ tin cậy TB | Cải thiện |
|------|--------------|--------|----------|----------------|-------------|
| 1 | k=[1,3,5] | bỏ_phiếu_có_trọng_số | **99.46%** | 0.995 ± 0.042 | +0.18% |
| 2 | k=[1,3,5] | độ_tin_cậy_trung_bình | **99.46%** | 0.957 ± 0.027 | +0.18% |
| 3 | k=[3,5,7] | độ_tin_cậy_tối_đa | 99.10% | 0.957 ± 0.026 | -0.18% |
| 4 | k=[5,7,9] | độ_tin_cậy_tối_đa | 99.10% | 0.953 ± 0.028 | -0.18% |
| 5 | k=[3,5,7] | bỏ_phiếu_có_trọng_số | 98.92% | 0.998 ± 0.025 | -0.36% |

### 🎯 Những phát hiện quan trọng

#### 1. **Cải thiện từ Ensemble**
- **Ensemble tốt nhất** (99.46%) vs **Đơn lẻ tốt nhất** (99.28%) = **Cải thiện +0.18%**
- **Điểm tin cậy** cung cấp thông tin có giá trị cho việc ra quyết định  
- **Bỏ phiếu có trọng số** và **độ tin cậy trung bình** đều đạt hiệu suất cao nhất

#### 2. **Hiểu biết về cấu hình**
- **k=[1,3,5]** là tối ưu: kết hợp độ chính xác (k=1) + tính ổn định (k=3,5)
- **Cải thiện khiêm tốn**: Với dữ liệu chất lượng cao, ensemble chỉ cải thiện nhẹ
- **Độ tin cậy cao**: Hầu hết cấu hình đều có độ tin cậy > 95%

#### 3. **Phân tích phương pháp**
```python
Phân tích hiệu suất phương pháp:
├── bỏ_phiếu_có_trọng_số    : trung bình=99.05% ± 0.22%, cao nhất=99.46%  # 🏆 Tốt nhất cho k=[1,3,5]
├── độ_tin_cậy_trung_bình   : trung bình=98.96% ± 0.22%, cao nhất=99.46%  # Cạnh tranh mạnh
└── độ_tin_cậy_tối_đa       : trung bình=98.84% ± 0.26%, cao nhất=99.10%  # Ổn định, hiệu suất tốt
```

### 🔍 Phân tích lỗi

#### Các ví dụ bị phân loại sai:

**Dương tính giả** (Ham → Spam):
```
"Miễn phí vận chuyển cho đơn hàng trên 500k"  # Khuyến mãi hợp pháp
→ Tương tự với: "Miễn phí vận chuyển! Mua ngay!"  # Spam trong dữ liệu huấn luyện
```

**Âm tính giả** (Spam → Ham):
```
"Chào bạn, tôi có đề xuất kinh doanh quan trọng"  # Spam tinh vi
→ Tương tự với: "Chào bạn, chúc bạn khỏe mạnh"    # Ham trong dữ liệu huấn luyện
```

### 📊 Phân bố độ tin cậy

Dựa trên kết quả thực nghiệm, chúng ta quan sát thấy:

```python
Phân tích điểm tin cậy:
├── Độ tin cậy rất cao (>0.95): 85% dự đoán ✅ >99% độ chính xác
├── Độ tin cậy cao (0.90-0.95): 12% dự đoán ✅ ~95% độ chính xác  
└── Độ tin cậy trung bình (<0.90): 3% dự đoán ⚠️ cần xem xét thêm
```

**Thông tin quan trọng:** Với độ tin cậy trung bình rất cao (>0.95), hệ thống cho thấy tính ổn định và đáng tin cậy cao cho ứng dụng thực tế!

---

## 7. Tổng kết

### 🏆 Những thành quả đạt được

#### ✅ **Thành tựu kỹ thuật:**
- **99.46% độ chính xác** trên tập kiểm tra (558 mẫu)
- **Cải thiện khiêm tốn +0.18%** so với k-NN đơn lẻ tốt nhất (k=3: 99.28%)
- **Độ tin cậy rất cao** (trung bình >0.95) cho thấy tính ổn định của hệ thống
- **Kiến trúc có thể mở rộng** với FAISS cho triển khai quy mô lớn
- **Khung đánh giá toàn diện** với nhiều cấu hình và phương pháp ensemble

#### ✅ **Lợi ích hệ thống:**
- **Suy luận thời gian thực**: Thời gian phản hồi dưới một giây
- **Kết quả có thể giải thích**: Các hàng xóm gần nhất và điểm tin cậy
- **Dự đoán bền vững**: Ensemble giảm hiện tượng quá khớp
- **Sẵn sàng cho sản xuất**: Pipeline hoàn chỉnh từ đầu vào văn bản đến kết quả
