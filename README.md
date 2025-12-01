# webdev-assignment
# Golden Owl Web Application
[[Demo video]](https://drive.google.com/drive/folders/1PEn-pywJsZgHDjLfDqDk_tOWraOmpBRr?usp=sharing).

Ứng dụng này quản lý dữ liệu điểm thi THPT 2024 với các tính năng:
- Tra cứu điểm thi theo số báo danh (SBD).
- Báo cáo xếp loại thí sinh theo các mức (>=8, 8 > && >=6, 6 > && >=4, <4) bằng biểu đồ (Chart.js).
- Danh sách top 10 thí sinh khối A (Toán, Vật lý, Hóa học).

## Nội dung
- [Yêu Cầu](#yêu-cầu)
- [Cài Đặt và Chạy trên Localhost](#cài-đặt-và-chạy-trên-localhost)
- [Chạy Ứng Dụng với Docker](#chạy-ứng-dụng-với-docker)
- [Ghi Chú](#ghi-chú)

## Yêu Cầu
- Python 3.9+
- PostgreSQL (cho local development)
- Docker & Docker Compose (nếu dùng Docker)
- Git

## Cài Đặt và Chạy trên Localhost
**Giải nén file rar Dataset: diem_thi_thpt_2024.csv**
1. **Clone Repository:**
   ```bash
   git clone https://github.com/your_username/goldenowl.git
   cd goldenowl
2. **Tạo Virtual Environment và Cài Đặt Dependencies:**
   ```
   python -m venv env
   # Trên Windows:
   env\Scripts\activate
   # Trên macOS/Linux:
   source env/bin/activate
   pip install -r requirements.txt
3. **Cấu Hình Cơ Sở Dữ Liệu (goldenowl/settings.py):**
   - Dùng Docker hãy chuyển HOST từ localhost sang db
   ```DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'goldenowl_db',
        'USER': 'goldenowl_user',
        'PASSWORD': '123456',
        'HOST': 'localhost',
        'PORT': '5432',
       }
   }
5. **Tạo Database và User trong PostgreSQL: Sử dụng psql hoặc pgAdmin:**
   ``` sql
   CREATE DATABASE goldenowl_db;
   CREATE USER goldenowl_user WITH PASSWORD '123456';
   GRANT ALL PRIVILEGES ON DATABASE goldenowl_db TO goldenowl_user;
6. **Chạy Migrations và Import Dữ Liệu:**
   ```
   python manage.py makemigrations
   python manage.py migrate
   python manage.py import_csv
7. **Chạy Ứng Dụng:**
   ```
   python manage.py runserver
**Truy cập ứng dụng tại: http://127.0.0.1:8000/dashboard/**

## Chạy Ứng Dụng với Docker
1. **Docker Compose File (docker-compose.yml):**
   ```yaml
      version: '3'
      services:
        postgres_db:
          image: postgres:13
          environment:
            POSTGRES_DB: goldenowl_db
            POSTGRES_USER: goldenowl_user
            POSTGRES_PASSWORD: 123456
          ports:
            - "5432:5432"
          volumes:
            - postgres_data:/var/lib/postgresql/data
      
        django_app:
          build: .
          command: gunicorn goldenowl.wsgi:application --bind 0.0.0.0:8000
          volumes:
            - .:/app
          ports:
            - "8000:8000"
          depends_on:
            - postgres_db
      
      volumes:
        postgres_data:
2. **Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   ENV PYTHONUNBUFFERED 1
   WORKDIR /app
   COPY requirements.txt /app/
   RUN pip install -r requirements.txt
   COPY . /app/
   CMD ["gunicorn", "goldenowl.wsgi:application", "--bind", "0.0.0.0:8000"]
3. **Xây dựng và Chạy Docker Compose:**
   ```docker-compose down
      docker-compose up --build
- Sau khi các container khởi động, truy cập: http://localhost:8000/dashboard/

**Lưu ý:** Khi chạy Docker, trong file settings.py đảm bảo HOST được đặt là tên service PostgreSQL (ví dụ: postgres_db).
## Ghi chú
- Nếu bạn gặp lỗi kết nối giữa Django và PostgreSQL khi dùng Docker, kiểm tra lại biến môi trường trong docker-compose.yml và cấu hình DATABASES trong settings.py.
- Để dừng server hoặc container: Sử dụng Ctrl + C cho local hoặc docker-compose down cho Docker.
# Báo cáo Tổng hợp: Các Kỹ thuật Nền tảng trong Xử lý Ngôn ngữ Tự nhiên với Học sâu

## 1.0 Giới thiệu: Từ Ký hiệu đến Ý nghĩa

### 1.1 Bối cảnh và Thách thức

Ngôn ngữ tự nhiên của con người là một hệ thống ký hiệu rời rạc, nhưng lại có khả năng truyền tải những ý nghĩa vô cùng phức tạp và đa sắc thái. Đối với máy tính, đây là một thách thức cốt lõi: làm thế nào để biểu diễn hệ thống ký hiệu này theo một cách có ý nghĩa về mặt tính toán? Việc phát triển các phương pháp biểu diễn số học—cụ thể là các vector—cho từ ngữ không chỉ là một bài toán kỹ thuật mà còn là một bước đi chiến lược, đặt nền móng cho hầu hết các ứng dụng **Xử lý Ngôn ngữ Tự nhiên (NLP)** hiện đại, từ dịch máy, trả lời câu hỏi cho đến các hệ thống đối thoại.

### 1.2 Các Phương pháp Biểu diễn Truyền thống và Hạn chế của chúng

Trong NLP truyền thống, các từ thường được xem như những ký hiệu rời rạc và độc lập. Phương pháp phổ biến nhất để hiện thực hóa ý tưởng này là sử dụng **vector one-hot**. Trong phương pháp này, mỗi từ trong từ vựng được biểu diễn bằng một vector có chiều dài bằng kích thước của từ vựng, với giá trị 1 tại vị trí tương ứng với từ đó và 0 ở tất cả các vị trí còn lại.

Tuy nhiên, phương pháp này có một nhược điểm cơ bản: nó không thể hiện được sự tương đồng về ngữ nghĩa giữa các từ. Ví dụ, hai từ "hotel" và "motel" có ý nghĩa rất gần nhau, nhưng vector one-hot của chúng lại hoàn toàn trực giao. Điều này có nghĩa là không có một khái niệm tự nhiên nào về sự tương đồng trong không gian vector này.

Để khắc phục điều này, các nhà nghiên cứu đã thử sử dụng các tài nguyên được chú giải thủ công như **WordNet**—một loại từ điển bách khoa chứa các danh sách tập hợp từ đồng nghĩa (*synonym sets*) và các mối quan hệ "is a" (*hypernyms*). Mặc dù hữu ích, WordNet cũng có những hạn chế đáng kể:

- **Thiếu sắc thái**: Nó không thể nắm bắt được sự khác biệt tinh tế giữa các từ đồng nghĩa (ví dụ: "proficient" chỉ là từ đồng nghĩa của "good" trong một số ngữ cảnh nhất định).
- **Không cập nhật**: Nó không thể theo kịp các ý nghĩa mới của từ ngữ (ví dụ: wicked, badass).
- **Tốn nhiều công sức**: Việc tạo và duy trì các tài nguyên này đòi hỏi rất nhiều lao động thủ công của con người.

### 1.3 Giả thuyết Phân bố: Một Bước đột phá

Một ý tưởng đột phá đã thay đổi hoàn toàn cách chúng ta tiếp cận vấn đề này, đó là *ngữ nghĩa học phân bố* (*distributional semantics*). Nguyên tắc cốt lõi của nó được tóm gọn trong câu nói nổi tiếng của nhà ngôn ngữ học J. R. Firth (1957): *"You shall know a word by the company it keeps"* (Bạn sẽ biết một từ qua những từ đồng hành cùng nó).

Nguyên tắc này cho rằng ý nghĩa của một từ có thể được rút ra từ các từ thường xuất hiện bên cạnh nó. Khi một từ $w$ xuất hiện trong một văn bản, ngữ cảnh của nó là tập hợp các từ xuất hiện gần đó (trong một cửa sổ có kích thước cố định). Bằng cách tổng hợp vô số ngữ cảnh của $w$, chúng ta có thể xây dựng một biểu diễn cho $w$.

### 1.4 Chuyển tiếp

Các phương pháp hiện đại đã hiện thực hóa giả thuyết phân bố này để tạo ra các vector từ dày đặc và giàu ý nghĩa, được gọi là **word embeddings**. Các phần tiếp theo của báo cáo này sẽ đi sâu vào hai trong số các thuật toán có ảnh hưởng nhất là **Word2Vec** và **GloVe**, khám phá cách chúng học các biểu diễn này và cách chúng ta có thể đánh giá chất lượng của chúng.

***

## 2.0 Biểu diễn Từ bằng Vector: Word2Vec và GloVe

### 2.1 Không gian Vector Ngữ nghĩa

Phần này khám phá hai phương pháp có ảnh hưởng lớn trong việc học các biểu diễn từ (*word embeddings*) từ các kho dữ liệu văn bản lớn. Các phương pháp này không chỉ giải quyết các vấn đề của vector one-hot mà còn tạo ra một không gian vector nơi các mối quan hệ ngữ nghĩa được mã hóa bằng hình học. Trong không gian này, các từ tương tự sẽ nằm gần nhau và các mối quan hệ tương tự có thể được biểu diễn bằng các phép toán vector đơn giản.

### 2.2 Phân tích Sâu về Word2Vec (Mô hình Skip-gram)

**Word2Vec** là một bộ công cụ do Mikolov và cộng sự phát triển vào năm 2013, trong đó mô hình **Skip-gram** là một trong những kiến trúc nổi bật nhất.

**Khái niệm cốt lõi.** Ý tưởng chính của Skip-gram là dự đoán các từ ngữ cảnh (*context words*) dựa trên một từ trung tâm (*center word*). Mô hình trượt một "cửa sổ ngữ cảnh" (*context window*) qua văn bản. Tại mỗi vị trí, nó lấy từ ở trung tâm và cố gắng dự đoán các từ xuất hiện trong cửa sổ xung quanh nó.

**Hàm mục tiêu và Công thức xác suất.** Mục tiêu của mô hình là điều chỉnh các vector từ để tối đa hóa xác suất dự đoán đúng các từ ngữ cảnh. Điều này tương đương với việc tối thiểu hóa hàm mất mát (*loss function*), được định nghĩa là log likelihood âm trung bình:

$$J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \le j \le m, j \ne 0} \log P(w_{t+j} | w_t; \theta)$$

Trong đó $w_t$ là từ trung tâm, $w_{t+j}$ là một từ ngữ cảnh, và $\theta$ là tất cả các tham số của mô hình. Để tính toán xác suất $P(o|c)$ cho một cặp từ trung tâm $c$ và từ ngữ cảnh $o$, Word2Vec sử dụng hai bộ vector cho mỗi từ: $v_w$ (khi $w$ là từ trung tâm) và $u_w$ (khi $w$ là từ ngữ cảnh). Xác suất được định nghĩa bằng hàm **softmax**:

$$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

Để hiểu sâu hơn về công thức này, ta hãy phân tích từng thành phần. Tích vô hướng $u_o^T v_c$ không chỉ đơn thuần là một con số; nó là thước đo hình học về sự tương đồng. Về mặt toán học, tích vô hướng tỷ lệ với cosine của góc giữa hai vector, do đó nó đo lường mức độ thẳng hàng của chúng trong không gian ngữ nghĩa. Một tích vô hướng cao có nghĩa là hai vector đang chỉ về cùng một hướng, thể hiện một mối quan hệ ngữ nghĩa mạnh mẽ. Hàm mũ $\exp(\cdot)$ sau đó đảm bảo tất cả các giá trị đều dương, và mẫu số thực hiện việc chuẩn hóa trên toàn bộ từ vựng $V$ để biến các điểm số này thành một phân phối xác suất hợp lệ.

**Tối ưu hóa với Negative Sampling.** Vấn đề lớn của hàm softmax là mẫu số cực kỳ tốn kém về mặt tính toán vì nó đòi hỏi phải tính tổng trên toàn bộ từ vựng. Để giải quyết vấn đề này, Word2Vec giới thiệu một kỹ thuật tối ưu hóa hiệu quả hơn gọi là **Negative Sampling**. Ý tưởng chính là thay vì cập nhật trọng số cho tất cả các từ trong từ vựng, mô hình chỉ cập nhật cho từ ngữ cảnh đúng và một vài mẫu "nhiễu" (*negative samples*) được chọn ngẫu nhiên. Mô hình được huấn luyện để phân biệt một cặp từ "thật" với các cặp từ "nhiễu". Hàm mục tiêu được định nghĩa lại như sau:

$$J_{neg-sample}(u_o, v_c, U) = -\log\sigma(u_o^T v_c) - \sum_{k \in \{K \text{ sampled indices}\}} \log\sigma(-u_k^T v_c)$$

Trong đó $\sigma(x) = 1/(1+e^{-x})$ là hàm **sigmoid**. Kỹ thuật này giúp giảm đáng kể chi phí tính toán và vẫn tạo ra các vector từ chất lượng cao.

**Triển khai Code (Python với NumPy):**

```python
import numpy as np
from collections import Counter

class SkipGram:
    def __init__(self, vocab_size, embedding_dim=100):
        """Khởi tạo mô hình Skip-gram với Negative Sampling"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Khởi tạo embeddings: W (center words) và W_out (context words)
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    def sigmoid(self, x):
        """Hàm sigmoid với clipping để tránh overflow"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def forward(self, center_idx, context_idx, negative_indices):
        """
        Forward pass với Negative Sampling
        
        Args:
            center_idx: Index của center word
            context_idx: Index của context word (positive)
            negative_indices: List indices của negative words
        
        Returns:
            loss: Cross-entropy loss
        """
        # Lấy embeddings
        v_c = self.W[center_idx]  # Center word embedding
        u_o = self.W_out[context_idx]  # Positive context embedding
        u_k = self.W_out[negative_indices]  # Negative context embeddings
        
        # Tính positive score
        positive_score = np.dot(v_c, u_o)
        positive_loss = -np.log(self.sigmoid(positive_score) + 1e-8)
        
        # Tính negative scores
        negative_scores = np.dot(u_k, v_c)  # (num_negative,)
        negative_loss = -np.sum(np.log(self.sigmoid(-negative_scores) + 1e-8))
        
        # Total loss
        loss = positive_loss + negative_loss
        
        return loss
    
    def backward(self, center_idx, context_idx, negative_indices, learning_rate=0.01):
        """Backward pass và update weights"""
        v_c = self.W[center_idx]
        u_o = self.W_out[context_idx]
        u_k = self.W_out[negative_indices]
        
        # Tính gradients
        positive_score = np.dot(v_c, u_o)
        positive_grad = self.sigmoid(positive_score) - 1
        
        negative_scores = np.dot(u_k, v_c)
        negative_grads = self.sigmoid(negative_scores)
        
        # Update center embedding
        grad_v_c = positive_grad * u_o + np.sum(negative_grads[:, np.newaxis] * u_k, axis=0)
        self.W[center_idx] -= learning_rate * grad_v_c
        
        # Update positive context embedding
        self.W_out[context_idx] -= learning_rate * positive_grad * v_c
        
        # Update negative context embeddings
        for i, neg_idx in enumerate(negative_indices):
            self.W_out[neg_idx] -= learning_rate * negative_grads[i] * v_c

# Ví dụ sử dụng
def train_skipgram(corpus, vocab_size, embedding_dim=100, window_size=2, 
                   num_negative=5, learning_rate=0.01, epochs=10):
    """Train Skip-gram model"""
    model = SkipGram(vocab_size, embedding_dim)
    
    # Tạo training pairs
    training_pairs = []
    for sentence in corpus:
        for i, center_word in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    training_pairs.append((center_word, sentence[j]))
    
    # Negative sampling distribution (unigram ^ 3/4)
    word_counts = Counter([w for sent in corpus for w in sent])
    word_probs = np.array([word_counts.get(i, 0) ** 0.75 for i in range(vocab_size)])
    word_probs = word_probs / word_probs.sum()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for center_idx, context_idx in training_pairs:
            # Sample negative examples
            negative_indices = np.random.choice(
                vocab_size, size=num_negative, p=word_probs, replace=False
            )
            negative_indices = [idx for idx in negative_indices if idx != context_idx]
            
            # Forward và backward
            loss = model.forward(center_idx, context_idx, negative_indices)
            model.backward(center_idx, context_idx, negative_indices, learning_rate)
            total_loss += loss
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(training_pairs):.4f}")
    
    return model
```

### 2.3 Phân tích về GloVe (Global Vectors)

Trong khi các mô hình dựa trên cửa sổ như Skip-gram rất giỏi trong việc nắm bắt các mẫu ngôn ngữ phức tạp, chúng lại không tận dụng được thống kê đồng xuất hiện (*co-occurrence*) trên toàn bộ kho dữ liệu. Ngược lại, các phương pháp dựa trên đếm (*count-based*) như **LSA** tận dụng tốt thống kê toàn cục nhưng lại kém trong các tác vụ suy luận tương tự. **GloVe** (Global Vectors for Word Representation) được giới thiệu để kết hợp những ưu điểm của cả hai phương pháp.

**Khái niệm cốt lõi của GloVe.** GloVe học trực tiếp từ ma trận đồng xuất hiện (*co-occurrence matrix*) $X$ trên toàn bộ kho dữ liệu, trong đó $X_{ij}$ là số lần từ $j$ xuất hiện trong ngữ cảnh của từ $i$. Thay vì dự đoán các từ riêng lẻ, GloVe tập trung vào việc mô hình hóa tỷ lệ xác suất đồng xuất hiện. Đây chính là cái nhìn sâu sắc cốt lõi của mô hình. Mô hình cố gắng học các vector từ sao cho tích vô hướng của chúng liên quan trực tiếp đến log của xác suất đồng xuất hiện. Hàm mục tiêu của GloVe là một mô hình bình phương tối thiểu có trọng số:

$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j - \log X_{ij})^2$$

Trong đó $f(X_{ij})$ là một hàm trọng số giúp giảm tầm quan trọng của các cặp từ đồng xuất hiện quá thường xuyên hoặc quá hiếm. Việc mô hình học trực tiếp mối quan hệ $\log(X_{ij})$ này chính là lý do tại sao GloVe vượt trội trong các tác vụ suy luận tương tự. Nó khuyến khích các vector học được các mối quan hệ tuyến tính, sao cho hiệu vector như *king - man + woman* có thể xấp xỉ với vector của *queen*, vì mô hình đã học được cách mã hóa các tỷ lệ đồng xuất hiện này vào trong cấu trúc không gian vector.

**Triển khai Code (Python với NumPy):**

```python
import numpy as np
from collections import defaultdict

class GloVe:
    def __init__(self, vocab_size, embedding_dim=100):
        """Khởi tạo mô hình GloVe"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Khởi tạo embeddings và biases
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_tilde = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.b = np.zeros(vocab_size)
        self.b_tilde = np.zeros(vocab_size)
    
    def weight_function(self, x, x_max=100, alpha=0.75):
        """
        Hàm trọng số f(x) cho GloVe
        f(x) = (x/x_max)^alpha nếu x < x_max, else 1
        """
        if x < x_max:
            return (x / x_max) ** alpha
        return 1.0
    
    def build_cooccurrence_matrix(self, corpus, window_size=5):
        """Xây dựng ma trận đồng xuất hiện"""
        cooccurrence = defaultdict(float)
        
        for sentence in corpus:
            for i, word_i in enumerate(sentence):
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                
                for j in range(start, end):
                    if j != i:
                        word_j = sentence[j]
                        # Distance weighting: 1/distance
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        cooccurrence[(word_i, word_j)] += weight
        
        return cooccurrence
    
    def train(self, cooccurrence, learning_rate=0.05, x_max=100, alpha=0.75, epochs=25):
        """Train GloVe model"""
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            for (i, j), x_ij in cooccurrence.items():
                # Tính weight
                weight = self.weight_function(x_ij, x_max, alpha)
                
                # Tính prediction
                prediction = (np.dot(self.W[i], self.W_tilde[j]) + 
                             self.b[i] + self.b_tilde[j])
                
                # Tính loss
                diff = prediction - np.log(x_ij + 1)  # +1 để tránh log(0)
                loss = weight * (diff ** 2)
                total_loss += loss
                count += 1
                
                # Tính gradients
                grad_main = 2 * weight * diff * self.W_tilde[j]
                grad_context = 2 * weight * diff * self.W[i]
                grad_bias_main = 2 * weight * diff
                grad_bias_context = 2 * weight * diff
                
                # Update weights
                self.W[i] -= learning_rate * grad_main
                self.W_tilde[j] -= learning_rate * grad_context
                self.b[i] -= learning_rate * grad_bias_main
                self.b_tilde[j] -= learning_rate * grad_bias_context
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Average Loss: {total_loss/count:.4f}")
        
        # Final embeddings là tổng của W và W_tilde
        self.embeddings = self.W + self.W_tilde
    
    def get_embedding(self, word_idx):
        """Lấy embedding của một từ"""
        return self.embeddings[word_idx]
    
    def find_analogy(self, word_a_idx, word_b_idx, word_c_idx, top_k=5):
        """
        Tìm từ d sao cho: a : b :: c : d
        Sử dụng: d = argmax(cosine(c - a + b, w))
        """
        vec_a = self.embeddings[word_a_idx]
        vec_b = self.embeddings[word_b_idx]
        vec_c = self.embeddings[word_c_idx]
        
        # Tính vector mục tiêu
        target = vec_c - vec_a + vec_b
        
        # Tính cosine similarity với tất cả các từ
        similarities = []
        for i in range(self.vocab_size):
            if i in [word_a_idx, word_b_idx, word_c_idx]:
                continue
            vec = self.embeddings[i]
            cosine = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
            similarities.append((i, cosine))
        
        # Sắp xếp và trả về top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Ví dụ sử dụng
def train_glove(corpus, vocab_size, embedding_dim=100):
    """Train GloVe model"""
    model = GloVe(vocab_size, embedding_dim)
    
    # Xây dựng ma trận đồng xuất hiện
    print("Building co-occurrence matrix...")
    cooccurrence = model.build_cooccurrence_matrix(corpus)
    print(f"Found {len(cooccurrence)} co-occurrence pairs")
    
    # Train model
    print("Training GloVe model...")
    model.train(cooccurrence)
    
    return model
```

### 2.4 Chuyển tiếp

Sau khi đã tạo ra các vector từ bằng các phương pháp như Word2Vec hoặc GloVe, bước tiếp theo là phải xác định chất lượng và sự hữu ích của chúng. Điều này đòi hỏi các phương pháp đánh giá có hệ thống, sẽ được trình bày trong phần tiếp theo.

***

## 3.0 Đánh giá Chất lượng của Vector Từ

### 3.1 Mở đầu

Việc đánh giá chất lượng của các biểu diễn từ đã học là một bước cực kỳ quan trọng. Chúng ta cần các phương pháp định lượng để so sánh các mô hình khác nhau hoặc các siêu tham số khác nhau. Có hai loại hình đánh giá chính: **nội tại** (*intrinsic*) và **ngoại tại** (*extrinsic*), mỗi loại đóng một vai trò riêng trong việc xác định độ tốt của các vector từ.

### 3.2 Đánh giá Nội tại (Intrinsic Evaluation)

Đánh giá nội tại đo lường hiệu suất của các vector từ trên các tác vụ con hoặc tác vụ trung gian. Các tác vụ này thường nhanh chóng và giúp cung cấp cái nhìn sâu sắc về chất lượng ngữ nghĩa và cú pháp của chính các vector. Tuy nhiên, để hữu ích, hiệu suất trên các tác vụ nội tại cần có tương quan dương với hiệu suất trên các tác vụ thực tế.

**Ví dụ 1: Suy luận Tương tự (Word Analogies).** Đây là một trong những phương pháp đánh giá nội tại phổ biến nhất. Tác vụ này kiểm tra xem liệu các mối quan hệ ngữ nghĩa và cú pháp có được mã hóa dưới dạng các phép toán tuyến tính trong không gian vector hay không. Một câu hỏi tương tự có dạng $a:b :: c:?$, ví dụ: *man:woman :: king:?*. Để tìm từ $d$, chúng ta tìm vector $x_d$ tối đa hóa độ tương đồng cosine với vector $x_b - x_a + x_c$:

$$d = \arg\max_i \frac{(x_b - x_a + x_c)^T x_i}{\|x_b - x_a + x_c\| \|x_i\|}$$

**Triển khai Code (Python):**

```python
import numpy as np

class WordEmbeddingEvaluator:
    """Đánh giá chất lượng Word Embeddings"""
    
    def __init__(self, embeddings, word2idx, idx2word):
        """
        Args:
            embeddings: Ma trận embeddings (vocab_size, embedding_dim)
            word2idx: Dictionary mapping word -> index
            idx2word: Dictionary mapping index -> word
        """
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(word2idx)
    
    def cosine_similarity(self, vec1, vec2):
        """Tính cosine similarity giữa 2 vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    
    def find_analogy(self, word_a, word_b, word_c, top_k=5):
        """
        Tìm từ d sao cho: a : b :: c : d
        Sử dụng công thức: d = argmax(cosine(c - a + b, w))
        
        Args:
            word_a, word_b, word_c: Các từ đầu vào
            top_k: Số kết quả trả về
        
        Returns:
            List các (word, similarity_score)
        """
        if word_a not in self.word2idx or word_b not in self.word2idx or word_c not in self.word2idx:
            return []
        
        idx_a = self.word2idx[word_a]
        idx_b = self.word2idx[word_b]
        idx_c = self.word2idx[word_c]
        
        vec_a = self.embeddings[idx_a]
        vec_b = self.embeddings[idx_b]
        vec_c = self.embeddings[idx_c]
        
        # Tính vector mục tiêu: c - a + b
        target_vec = vec_c - vec_a + vec_b
        
        # Tính cosine similarity với tất cả các từ
        similarities = []
        for i in range(self.vocab_size):
            if i in [idx_a, idx_b, idx_c]:
                continue  # Bỏ qua các từ đầu vào
            
            word = self.idx2word[i]
            vec = self.embeddings[i]
            similarity = self.cosine_similarity(target_vec, vec)
            similarities.append((word, similarity))
        
        # Sắp xếp và trả về top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def evaluate_analogies(self, analogy_dataset):
        """
        Đánh giá trên dataset analogies
        
        Args:
            analogy_dataset: List các tuples (word_a, word_b, word_c, word_d)
        
        Returns:
            accuracy: Tỷ lệ đúng
        """
        correct = 0
        total = 0
        
        for word_a, word_b, word_c, word_d in analogy_dataset:
            if word_d not in self.word2idx:
                continue
            
            results = self.find_analogy(word_a, word_b, word_c, top_k=1)
            if results and results[0][0] == word_d:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def find_most_similar(self, word, top_k=10):
        """
        Tìm các từ tương tự nhất với word
        
        Args:
            word: Từ cần tìm similar
            top_k: Số kết quả trả về
        
        Returns:
            List các (word, similarity_score)
        """
        if word not in self.word2idx:
            return []
        
        word_idx = self.word2idx[word]
        word_vec = self.embeddings[word_idx]
        
        similarities = []
        for i in range(self.vocab_size):
            if i == word_idx:
                continue
            
            other_word = self.idx2word[i]
            other_vec = self.embeddings[i]
            similarity = self.cosine_similarity(word_vec, other_vec)
            similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def evaluate_word_similarity(self, similarity_dataset):
        """
        Đánh giá tương quan với human judgments
        
        Args:
            similarity_dataset: List các tuples (word1, word2, human_score)
        
        Returns:
            correlation: Pearson correlation coefficient
        """
        from scipy.stats import pearsonr
        
        predicted_scores = []
        human_scores = []
        
        for word1, word2, human_score in similarity_dataset:
            if word1 not in self.word2idx or word2 not in self.word2idx:
                continue
            
            idx1 = self.word2idx[word1]
            idx2 = self.word2idx[word2]
            
            vec1 = self.embeddings[idx1]
            vec2 = self.embeddings[idx2]
            
            predicted_score = self.cosine_similarity(vec1, vec2)
            predicted_scores.append(predicted_score)
            human_scores.append(human_score)
        
        if len(predicted_scores) < 2:
            return 0
        
        correlation, _ = pearsonr(predicted_scores, human_scores)
        return correlation

# Ví dụ sử dụng
def example_usage():
    """Ví dụ sử dụng Word Embedding Evaluator"""
    
    # Giả sử đã có embeddings được train
    # embeddings = ... (vocab_size, embedding_dim)
    # word2idx = {...}
    # idx2word = {...}
    
    evaluator = WordEmbeddingEvaluator(embeddings, word2idx, idx2word)
    
    # Test word analogy
    print("Word Analogy: man : woman :: king : ?")
    results = evaluator.find_analogy("man", "woman", "king", top_k=5)
    for word, score in results:
        print(f"  {word}: {score:.4f}")
    
    # Test word similarity
    print("\nMost similar words to 'computer':")
    similar = evaluator.find_most_similar("computer", top_k=10)
    for word, score in similar:
        print(f"  {word}: {score:.4f}")
    
    # Evaluate on dataset
    analogy_dataset = [
        ("man", "woman", "king", "queen"),
        ("paris", "france", "tokyo", "japan"),
        # ... more examples
    ]
    accuracy = evaluator.evaluate_analogies(analogy_dataset)
    print(f"\nAnalogy Accuracy: {accuracy:.2%}")
```

**Các ví dụ về suy luận ngữ nghĩa và các vấn đề tiềm ẩn:**

| Input | Kết quả | Ghi chú |
|-------|--------|---------|
| Chicago : Illinois :: Houston | Texas | Đúng |
| Chicago : Illinois :: Philadelphia | Pennsylvania | Đúng |
| Chicago : Illinois :: Phoenix | Arizona | Đúng, nhưng "Phoenix" là một tên địa danh phổ biến ở nhiều tiểu bang. |
| Abuja : Nigeria :: Accra | Ghana | Đúng |
| Abuja : Nigeria :: Ankara | Turkey | Đúng |
| Abuja : Nigeria :: Astana | Kazakhstan | Đúng tại thời điểm dữ liệu được thu thập, nhưng thủ đô đã thay đổi. |

**Các ví dụ về suy luận cú pháp:**

| Input | Kết quả | Ghi chú |
|-------|--------|---------|
| bad : worst :: big | biggest | So sánh nhất (Superlative) |
| good : best :: great | greatest | So sánh nhất (Superlative) |
| dancing : danced :: falling | fell | Thì quá khứ (Past Tense) |
| flying : flew :: hiding | hid | Thì quá khứ (Past Tense) |

**Ví dụ 2: Tương quan với Đánh giá của Con người.** Phương pháp này so sánh độ tương đồng giữa các vector từ (thường được đo bằng cosine) với điểm số tương đồng do con người đánh giá trên một thang điểm cố định. Dưới đây là ví dụ về hiệu suất của GloVe và Skip-gram (SG) trên tập dữ liệu WordSim353 (WS353).

| Mô hình | Kích thước Corpus | Tương quan trên WS353 |
|--------|-------------------|----------------------|
| SG | 6B | 62.8 |
| GloVe | 6B | 65.8 |
| GloVe | 42B | 75.9 |

### 3.3 Đánh giá Ngoại tại (Extrinsic Evaluation)

Đánh giá ngoại tại đo lường hiệu suất của các vector từ trên một tác vụ NLP thực tế cuối cùng, chẳng hạn như phân tích tình cảm (*sentiment analysis*) hoặc nhận dạng thực thể tên (**Named Entity Recognition - NER**). Đây là thước đo cuối cùng về sự hữu ích của các vector, vì nếu việc thay thế một bộ vector này bằng một bộ vector khác cải thiện hiệu suất trên tác vụ cuối cùng, thì đó là một sự cải tiến rõ ràng.

Tuy nhiên, phương pháp này cũng có nhược điểm. Nó thường rất chậm, vì việc huấn luyện lại toàn bộ hệ thống cho một tác vụ cuối cùng có thể mất nhiều thời gian. Hơn nữa, nếu hiệu suất kém, rất khó để chẩn đoán nguyên nhân là do chất lượng của các vector từ hay do các thành phần khác của hệ thống.

### 3.4 Chuyển tiếp

Một khi chúng ta đã có các vector từ được đánh giá tốt, chúng ta có thể sử dụng chúng làm đầu vào cho các mô hình học sâu phức tạp hơn. Các mô hình này, chẳng hạn như mạng nơ-ron, có thể học các ranh giới quyết định phi tuyến và giải quyết các tác vụ NLP cụ thể một cách hiệu quả.

***

## 4.0 Mạng Nơ-ron và Backpropagation cho NLP

### 4.1 Sự cần thiết của các Bộ phân loại Phi tuyến

Dữ liệu ngôn ngữ hiếm khi có thể được phân tách một cách tuyến tính. Do đó, các bộ phân loại tuyến tính đơn giản thường không đủ mạnh. **Mạng nơ-ron** (*Neural Networks*) là một họ các bộ phân loại mạnh mẽ có khả năng học các ranh giới quyết định phức tạp, phi tuyến, khiến chúng trở thành công cụ lý tưởng cho nhiều tác vụ NLP.

### 4.2 Kiến trúc Mạng Nơ-ron Feed-Forward cho NER

Chúng ta sẽ sử dụng tác vụ **Nhận dạng Thực thể Tên** (*Named Entity Recognition - NER*) làm ví dụ để minh họa. Mục tiêu của NER là phân loại một từ là một thực thể tên (ví dụ: PER - Người, LOC - Địa điểm) hay không.

**Thiết lập bài toán: Phân loại dựa trên cửa sổ.** Phương pháp phổ biến là sử dụng một cửa sổ các từ xung quanh từ trung tâm cần phân loại. Các vector từ của các từ trong cửa sổ này được ghép lại với nhau để tạo thành một vector đầu vào duy nhất $x_{window}$. Ví dụ, để phân loại từ "Paris" trong câu "Museums in Paris are amazing", với kích thước cửa sổ là 2, vector đầu vào sẽ là: $x_{window} = [x_{museums}, x_{in}, x_{Paris}, x_{are}, x_{amazing}]$.

**Các thành phần của Mạng.** Một mạng nơ-ron feed-forward đơn giản với một lớp ẩn bao gồm các thành phần sau:

1. **Lớp ẩn** (*Hidden Layer*): Tính toán một giá trị trung gian $z$ bằng một phép biến đổi affine: $z = Wx + b$, sau đó áp dụng một hàm kích hoạt phi tuyến $f$ lên $z$ để tạo ra vector kích hoạt $a$: $a = f(z)$.
2. **Lớp đầu ra** (*Output Layer*): Tính toán một điểm số (*score*) $s$ từ vector kích hoạt $a$: $s = U^T a$.

Hàm kích hoạt phi tuyến đóng vai trò quan trọng trong việc cho phép mô hình học các hàm phức tạp. Một số hàm kích hoạt phổ biến bao gồm:

| Tên hàm | Công thức |
|---------|-----------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ |
| ReLU | $\text{rect}(z) = \max(z, 0)$ |

**Hàm mục tiêu và Huấn luyện.** Để huấn luyện mạng cho các tác vụ phân loại, hàm mất mát **cross-entropy** thường được sử dụng. Đối với một tập dữ liệu gồm $N$ mẫu, hàm mất mát được định nghĩa là:

$$J(\theta) = -\sum_{i=1}^{N} \log\left(\frac{\exp(W_{k(i)} \cdot x^{(i)})}{\sum_{c=1}^{C} \exp(W_c \cdot x^{(i)})}\right)$$

Trong đó $k(i)$ là chỉ số của lớp đúng cho mẫu thứ $i$.

### 4.3 Giải thích về Backpropagation và Tối ưu hóa

**Khái niệm Backpropagation.** **Backpropagation** (Lan truyền ngược) là một thuật toán hiệu quả để tính toán gradient (đạo hàm) của hàm mất mát đối với tất cả các tham số của mạng. Về bản chất, nó là một ứng dụng thông minh của quy tắc chuỗi (*chain rule*) trong giải tích, hoạt động bằng cách lan truyền "tín hiệu lỗi" một cách đệ quy, bắt đầu từ lớp đầu ra và đi ngược lại qua từng lớp của mạng. Điều này cho phép chúng ta biết mỗi tham số cần được điều chỉnh như thế nào để giảm thiểu sai số.

**Gradient Descent.** Một khi đã có các gradient, chúng ta có thể cập nhật các tham số của mô hình bằng thuật toán **Stochastic Gradient Descent (SGD)**. Công thức cập nhật rất đơn giản:

$$\theta_{new} = \theta_{old} - \alpha \nabla_\theta J(\theta)$$

Trong đó $\theta$ là một tham số của mô hình, $\alpha$ là tốc độ học (*learning rate*), và $\nabla_\theta J(\theta)$ là gradient của hàm mất mát.

**Các Kỹ thuật Quan trọng.** Trong thực tế, việc huấn luyện mạng nơ-ron đòi hỏi nhiều kỹ thuật bổ sung để đảm bảo sự ổn định và hiệu quả. Các kỹ thuật quan trọng bao gồm kiểm tra gradient (*gradient checking*) để xác thực việc triển khai, điều chuẩn hóa (*regularization*, ví dụ: L2, Dropout) để chống lại hiện tượng quá khớp (*overfitting*), và các phương pháp khởi tạo tham số (ví dụ: **Xavier initialization**) để giúp quá trình hội tụ ổn định hơn.

**Triển khai Code (Python với NumPy):**

```python
import numpy as np

class FeedForwardNER:
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=100, 
                 num_classes=3, window_size=2):
        """
        Mạng Feed-Forward cho Named Entity Recognition
        
        Args:
            vocab_size: Kích thước từ vựng
            embedding_dim: Chiều của word embeddings
            hidden_dim: Số units trong hidden layer
            num_classes: Số lớp (PER, LOC, O)
            window_size: Kích thước cửa sổ xung quanh từ trung tâm
        """
        self.window_size = window_size
        input_dim = (2 * window_size + 1) * embedding_dim
        
        # Khởi tạo với Xavier initialization
        self.W_emb = np.random.randn(vocab_size, embedding_dim) * np.sqrt(2.0 / vocab_size)
        self.W_hidden = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b_hidden = np.zeros(hidden_dim)
        self.W_output = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
        self.b_output = np.zeros(num_classes)
        
        # Dropout probability
        self.dropout_prob = 0.5
    
    def relu(self, x):
        """Hàm kích hoạt ReLU"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Hàm softmax với numerical stability"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, word_indices, training=True):
        """
        Forward pass
        
        Args:
            word_indices: List các word indices trong cửa sổ
            training: Có apply dropout không
        
        Returns:
            scores: Logits cho các classes
        """
        # 1. Embedding lookup
        embeddings = [self.W_emb[idx] for idx in word_indices]
        x = np.concatenate(embeddings)  # Flatten
        
        # 2. Hidden layer
        z = np.dot(x, self.W_hidden) + self.b_hidden
        a = self.relu(z)
        
        # 3. Dropout (chỉ khi training)
        if training:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_prob, size=a.shape)
            a = a * dropout_mask / (1 - self.dropout_prob)
        
        # 4. Output layer
        scores = np.dot(a, self.W_output) + self.b_output
        
        return scores
    
    def predict(self, word_indices):
        """Dự đoán class cho một cửa sổ từ"""
        scores = self.forward(word_indices, training=False)
        probs = self.softmax(scores.reshape(1, -1))
        return np.argmax(probs)
    
    def compute_loss(self, word_indices, true_class):
        """Tính Cross-Entropy Loss"""
        scores = self.forward(word_indices, training=True)
        probs = self.softmax(scores.reshape(1, -1))
        loss = -np.log(probs[0, true_class] + 1e-8)
        return loss
    
    def backward(self, word_indices, true_class, learning_rate=0.01):
        """Backward pass và update weights"""
        # Forward pass để lấy activations
        embeddings = [self.W_emb[idx] for idx in word_indices]
        x = np.concatenate(embeddings)
        
        z = np.dot(x, self.W_hidden) + self.b_hidden
        a = self.relu(z)
        
        # Apply dropout
        dropout_mask = np.random.binomial(1, 1 - self.dropout_prob, size=a.shape)
        a_drop = a * dropout_mask / (1 - self.dropout_prob)
        
        scores = np.dot(a_drop, self.W_output) + self.b_output
        probs = self.softmax(scores.reshape(1, -1))
        
        # Tính gradients
        # Output layer gradients
        grad_scores = probs.copy()
        grad_scores[0, true_class] -= 1
        
        grad_W_output = np.outer(a_drop, grad_scores[0])
        grad_b_output = grad_scores[0]
        grad_a = np.dot(self.W_output, grad_scores[0])
        
        # Hidden layer gradients (với ReLU derivative)
        grad_z = grad_a * (z > 0) * dropout_mask / (1 - self.dropout_prob)
        grad_W_hidden = np.outer(x, grad_z)
        grad_b_hidden = grad_z
        grad_x = np.dot(self.W_hidden, grad_z)
        
        # Embedding gradients (phân phối về các từ trong cửa sổ)
        embedding_dim = self.W_emb.shape[1]
        for i, word_idx in enumerate(word_indices):
            start = i * embedding_dim
            end = (i + 1) * embedding_dim
            self.W_emb[word_idx] -= learning_rate * grad_x[start:end]
        
        # Update weights
        self.W_hidden -= learning_rate * grad_W_hidden
        self.b_hidden -= learning_rate * grad_b_hidden
        self.W_output -= learning_rate * grad_W_output
        self.b_output -= learning_rate * grad_b_output

# Ví dụ sử dụng
def train_ner_model(sentences, labels, vocab_size, epochs=10):
    """Train NER model"""
    model = FeedForwardNER(vocab_size, embedding_dim=50, hidden_dim=100, 
                          num_classes=3, window_size=2)
    
    for epoch in range(epochs):
        total_loss = 0
        for sentence, sentence_labels in zip(sentences, labels):
            for i in range(len(sentence)):
                # Lấy cửa sổ
                start = max(0, i - model.window_size)
                end = min(len(sentence), i + model.window_size + 1)
                word_indices = sentence[start:end]
                
                # Padding nếu cần
                while len(word_indices) < 2 * model.window_size + 1:
                    word_indices = [0] + word_indices  # Pad với <PAD>
                
                true_class = sentence_labels[i]
                
                # Forward và backward
                loss = model.compute_loss(word_indices, true_class)
                model.backward(word_indices, true_class)
                total_loss += loss
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(sentences):.4f}")
    
    return model
```

### 4.4 Chuyển tiếp

Mặc dù mạng feed-forward rất mạnh mẽ cho các tác vụ với đầu vào có kích thước cố định, nhiều vấn đề trong NLP lại liên quan đến việc xử lý các chuỗi có độ dài thay đổi. Điều này dẫn đến sự cần thiết của một loại kiến trúc mạng khác, có khả năng xử lý dữ liệu tuần tự một cách tự nhiên: mạng nơ-ron hồi quy.

***

## 5.0 Phân tích Cú pháp Phụ thuộc

### 5.1 Tầm quan trọng của Cấu trúc Cú pháp

Để hiểu được ý nghĩa đầy đủ của một câu, chúng ta cần phải phân tích cấu trúc cú pháp của nó—tức là cách các từ kết hợp với nhau. Nếu không có cấu trúc, ý nghĩa có thể trở nên mơ hồ. Chẳng hạn:

- **Sự mơ hồ trong việc gắn kết cụm giới từ** (*PP attachment*): Trong câu "Scientists count whales from space", cụm "from space" có thể bổ nghĩa cho "whales" (những con cá voi từ không gian) hay cho "count" (đếm từ không gian)?
- **Phạm vi phối hợp** (*Coordination scope*): Trong câu "Shuttle veteran and longtime NASA executive Fred Gregory...", cụm "Shuttle veteran" và "longtime NASA executive" có thể cùng mô tả Fred Gregory, hoặc có thể là hai thực thể riêng biệt.

Việc xác định đúng các mối quan hệ giữa các từ là rất quan trọng để giải quyết những sự mơ hồ này.

### 5.2 Ngữ pháp Phụ thuộc (Dependency Grammar)

**Ngữ pháp Phụ thuộc** là một khuôn khổ mô tả cấu trúc cú pháp của một câu thông qua các mối quan hệ nhị phân, bất đối xứng được gọi là phụ thuộc (*dependencies*). Mỗi mối quan hệ này là một "mũi tên" nối một từ *head* (hoặc *governor*) với một từ *dependent* (hoặc *modifier*). Các mũi tên này thường được gán nhãn với tên của quan hệ ngữ pháp (ví dụ: nsubj - chủ ngữ, obj - tân ngữ).

Ví dụ, câu "Bills on ports and immigration were submitted..." có thể được biểu diễn bằng các quan hệ phụ thuộc như:

- submitted -> Bills (nhãn: nsubj:pass)
- Bills -> on (nhãn: nmod)
- on -> ports (nhãn: nmod)
- ports -> immigration (nhãn: conj)

### 5.3 Phân tích cú pháp Phụ thuộc dựa trên Chuyển tiếp (Transition-Based Parsing)

Đây là một phương pháp phổ biến và hiệu quả để xây dựng cây phụ thuộc. Nó hoạt động giống như một máy trạng thái, thực hiện một chuỗi các hành động để dần dần xây dựng cây.

**Thiết lập.** Hệ thống bao gồm ba thành phần chính: một ngăn xếp (*stack* $\sigma$) chứa các từ đã xử lý một phần, một vùng đệm (*buffer* $\beta$) chứa các từ còn lại, và một tập hợp các cung phụ thuộc (*dependency arcs* $A$).

**Các hành động.** Hệ thống **arc-standard**, một biến thể phổ biến, sử dụng ba hành động cơ bản:

1. **SHIFT**: Lấy từ đầu tiên ra khỏi vùng đệm và đẩy nó vào đỉnh ngăn xếp.
2. **LEFT-ARC**: Tạo một cung phụ thuộc từ từ ở đỉnh ngăn xếp ($w_j$) đến từ thứ hai trên ngăn xếp ($w_i$). Sau đó, loại bỏ $w_i$ khỏi ngăn xếp.
3. **RIGHT-ARC**: Tạo một cung phụ thuộc từ từ thứ hai trên ngăn xếp ($w_i$) đến từ ở đỉnh ngăn xếp ($w_j$). Sau đó, loại bỏ $w_j$ khỏi ngăn xếp.

Để làm cho quá trình này trở nên cụ thể hơn, hãy xem xét một ví dụ đơn giản với câu "I ate fish":

- Trạng thái ban đầu: Ngăn xếp: [ROOT], Vùng đệm: [I, ate, fish], Cung: {}
- Hành động: SHIFT -> Ngăn xếp: [ROOT, I], Vùng đệm: [ate, fish], Cung: {}
- Hành động: SHIFT -> Ngăn xếp: [ROOT, I, ate], Vùng đệm: [fish], Cung: {}
- Hành động: LEFT-ARC (nsubj) -> Ngăn xếp: [ROOT, ate], Vùng đệm: [fish], Cung: {ate -> I}
- Hành động: SHIFT -> Ngăn xếp: [ROOT, ate, fish], Vùng đệm: [], Cung: {ate -> I}
- Hành động: RIGHT-ARC (obj) -> Ngăn xếp: [ROOT, ate], Vùng đệm: [], Cung: {ate -> I, ate -> fish}

**Mô hình Nơ-ron để Dự đoán Hành động.** Tại mỗi bước, một mạng nơ-ron feed-forward có thể được huấn luyện để dự đoán hành động tiếp theo. Mô hình trích xuất các đặc trưng từ trạng thái hiện tại (ví dụ: các từ trên đỉnh ngăn xếp và đầu vùng đệm), chuyển chúng thành các vector dày đặc, và đưa vào mạng để nhận được một phân phối xác suất trên các hành động khả thi.

**Triển khai Code (Python):**

```python
class PartialParse:
    """Partial Parse cho Transition-Based Dependency Parsing"""
    
    def __init__(self, sentence):
        self.sentence = sentence
        self.stack = ["ROOT"]
        self.buffer = sentence[:]  # Copy để không modify sentence gốc
        self.dependencies = []
    
    def parse_step(self, transition):
        """Thực hiện một transition"""
        if transition == "SHIFT":
            if len(self.buffer) > 0:
                self.stack.append(self.buffer.pop(0))
        
        elif transition == "LEFT-ARC":
            if len(self.stack) >= 2:
                dependent = self.stack.pop()  # Top
                head = self.stack[-1]  # Second item
                self.dependencies.append((head, dependent))
        
        elif transition == "RIGHT-ARC":
            if len(self.stack) >= 2:
                dependent = self.stack[-2]  # Second item
                head = self.stack[-1]  # Top
                self.stack.pop(-2)  # Remove dependent
                self.dependencies.append((head, dependent))
    
    def is_finished(self):
        """Kiểm tra xem parse đã xong chưa"""
        return len(self.buffer) == 0 and len(self.stack) == 1

class DependencyParser:
    """Neural Dependency Parser"""
    
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=200, num_classes=3):
        """
        Args:
            vocab_size: Kích thước từ vựng
            embedding_dim: Chiều của embeddings
            hidden_dim: Số units trong hidden layer
            num_classes: Số transitions (SHIFT, LEFT-ARC, RIGHT-ARC)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Khởi tạo weights với Xavier initialization
        n_features = 36  # Số features extract từ stack và buffer
        input_dim = n_features * embedding_dim
        
        self.W_emb = np.random.randn(vocab_size, embedding_dim) * np.sqrt(2.0 / vocab_size)
        self.W_hidden = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b_hidden = np.zeros(hidden_dim)
        self.W_output = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
        self.b_output = np.zeros(num_classes)
    
    def extract_features(self, stack, buffer, sentence):
        """
        Trích xuất features từ trạng thái hiện tại
        Features: top 3 từ trong stack, top 3 từ trong buffer, 
                 left/right children của top 2 từ trong stack
        """
        features = []
        
        # Stack features (top 3)
        for i in range(3):
            if i < len(stack):
                features.append(stack[-(i+1)])
            else:
                features.append(0)  # NULL token
        
        # Buffer features (top 3)
        for i in range(3):
            if i < len(buffer):
                features.append(buffer[i])
            else:
                features.append(0)  # NULL token
        
        # Left/right children features (simplified)
        # Trong implementation thực tế, cần track dependencies đã tạo
        for i in range(2):
            if i < len(stack):
                # Left child, right child, left-left, right-right, etc.
                features.extend([0] * 12)  # Placeholder
        
        return features[:36]  # Đảm bảo đúng 36 features
    
    def forward(self, features, training=True):
        """Forward pass"""
        # Embedding lookup
        embeddings = [self.W_emb[f] for f in features]
        x = np.concatenate(embeddings)
        
        # Hidden layer với ReLU
        z = np.dot(x, self.W_hidden) + self.b_hidden
        a = np.maximum(0, z)  # ReLU
        
        # Dropout khi training
        if training:
            dropout_mask = np.random.binomial(1, 0.5, size=a.shape)
            a = a * dropout_mask * 2
        
        # Output layer
        scores = np.dot(a, self.W_output) + self.b_output
        
        return scores
    
    def predict_transition(self, partial_parse, sentence):
        """Dự đoán transition tiếp theo"""
        features = self.extract_features(
            partial_parse.stack, partial_parse.buffer, sentence
        )
        scores = self.forward(features, training=False)
        
        # Chọn transition hợp lệ (legal transitions)
        legal_transitions = self.get_legal_transitions(partial_parse)
        
        # Mask illegal transitions
        for i, legal in enumerate(legal_transitions):
            if not legal:
                scores[i] = -np.inf
        
        return np.argmax(scores)
    
    def get_legal_transitions(self, partial_parse):
        """
        Xác định transitions hợp lệ
        - SHIFT: hợp lệ nếu buffer không rỗng
        - LEFT-ARC: hợp lệ nếu stack có >= 2 items (và top không phải ROOT)
        - RIGHT-ARC: hợp lệ nếu stack có >= 2 items
        """
        legal = [False, False, False]  # [SHIFT, LEFT-ARC, RIGHT-ARC]
        
        if len(partial_parse.buffer) > 0:
            legal[0] = True  # SHIFT
        
        if len(partial_parse.stack) >= 2:
            # LEFT-ARC: top không phải ROOT
            if partial_parse.stack[-1] != "ROOT":
                legal[1] = True
            legal[2] = True  # RIGHT-ARC
        
        return legal

# Ví dụ sử dụng
def parse_sentence(parser, sentence, word2idx):
    """Parse một câu"""
    # Convert sentence to indices
    sentence_idx = [word2idx.get(w, 0) for w in sentence]
    
    # Khởi tạo partial parse
    partial_parse = PartialParse(sentence)
    
    # Parse cho đến khi xong
    max_steps = len(sentence) * 2  # Safety limit
    for step in range(max_steps):
        if partial_parse.is_finished():
            break
        
        # Predict transition
        transition_idx = parser.predict_transition(partial_parse, sentence_idx)
        transitions = ["SHIFT", "LEFT-ARC", "RIGHT-ARC"]
        transition = transitions[transition_idx]
        
        # Apply transition
        partial_parse.parse_step(transition)
    
    return partial_parse.dependencies
```

### 5.4 Chuyển tiếp

Phân tích cú pháp phụ thuộc cho phép chúng ta hiểu được cấu trúc phân cấp của câu. Tuy nhiên, ngôn ngữ về bản chất là một chuỗi tuần tự. Để nắm bắt được bản chất này, chúng ta cần các mô hình được thiết kế đặc biệt để xử lý dữ liệu chuỗi, sẽ được giới thiệu trong phần tiếp theo.

***

## 6.0 Mô hình hóa Chuỗi với Mạng Nơ-ron Hồi quy (RNN)

### 6.1 Nhiệm vụ Mô hình hóa Ngôn ngữ

**Mô hình hóa Ngôn ngữ** (*Language Modeling - LM*) là nhiệm vụ dự đoán từ tiếp theo trong một chuỗi, dựa trên các từ đã có trước đó. Đây là một nhiệm vụ nền tảng trong NLP, không chỉ giúp đo lường khả năng "hiểu" ngôn ngữ của một mô hình mà còn là cốt lõi của nhiều ứng dụng sinh văn bản hiện đại.

### 6.2 Hạn chế của các Mô hình Cũ

**Mô hình n-gram.** Mô hình n-gram dự đoán từ tiếp theo dựa trên $n-1$ từ đứng ngay trước nó bằng cách đếm tần suất xuất hiện. Tuy nhiên, chúng gặp phải hai vấn đề lớn: tính thưa thớt (*sparsity*) (nhiều n-gram hợp lệ có thể không bao giờ xuất hiện trong dữ liệu huấn luyện) và vấn đề lưu trữ (*storage*) (số lượng n-gram tăng theo cấp số nhân).

**Mô hình Cửa sổ Cố định.** Mô hình mạng nơ-ron dựa trên cửa sổ cố định giải quyết được vấn đề thưa thớt nhưng vẫn có hạn chế cố hữu: nó không thể xử lý các ngữ cảnh có độ dài thay đổi và không có sự đối xứng trong cách xử lý các đầu vào.

### 6.3 Kiến trúc Mạng Nơ-ron Hồi quy (RNN)

**Ý tưởng cốt lõi.** **Mạng Nơ-ron Hồi quy** (*Recurrent Neural Network - RNN*) được thiết kế để giải quyết những hạn chế trên. Ý tưởng cốt lõi của RNN là áp dụng cùng một bộ trọng số lặp đi lặp lại qua từng bước thời gian của chuỗi. Tại mỗi bước thời gian $t$, mạng nhận đầu vào $x_t$ và trạng thái ẩn từ bước trước đó $h_{t-1}$ để tính toán trạng thái ẩn mới $h_t$. Trạng thái ẩn này hoạt động như một "bộ nhớ" tóm tắt thông tin từ tất cả các bước trước đó.

**Công thức toán học.** Một RNN đơn giản được định nghĩa bởi các phương trình sau:

- Trạng thái ẩn tại bước thời gian $t$: $h_t = \sigma(W_{hh} h_{t-1} + W_{hx} x_t)$
- Dự đoán đầu ra tại bước thời gian $t$: $\hat{y}_t = \text{softmax}(W_S h_t)$

Trong đó $W_{hh}$, $W_{hx}$, $W_S$ là các ma trận trọng số được chia sẻ qua tất cả các bước thời gian. Kiến trúc này cho phép RNN xử lý các chuỗi có độ dài bất kỳ và, về mặt lý thuyết, lưu giữ thông tin từ các bước rất xa trong quá khứ.

**Triển khai Code (Python với NumPy):**

```python
import numpy as np

class SimpleRNN:
    """Mạng RNN đơn giản cho Language Modeling"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        """
        Args:
            vocab_size: Kích thước từ vựng
            embedding_dim: Chiều của word embeddings
            hidden_dim: Số units trong hidden layer
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Khởi tạo weights với Xavier initialization
        self.W_emb = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)
        self.W_hx = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(1.0 / embedding_dim)
        self.W_S = np.random.randn(hidden_dim, vocab_size) * np.sqrt(1.0 / hidden_dim)
        
        self.b_h = np.zeros(hidden_dim)
        self.b_S = np.zeros(vocab_size)
    
    def sigmoid(self, x):
        """Hàm sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def tanh(self, x):
        """Hàm tanh"""
        return np.tanh(x)
    
    def softmax(self, x):
        """Hàm softmax với numerical stability"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, sequence):
        """
        Forward pass qua toàn bộ sequence
        
        Args:
            sequence: List các word indices
        
        Returns:
            hidden_states: List các hidden states
            outputs: List các output predictions
        """
        hidden_states = []
        outputs = []
        h_prev = np.zeros(self.hidden_dim)  # Khởi tạo hidden state
        
        for word_idx in sequence:
            # Embedding lookup
            x = self.W_emb[word_idx]
            
            # Hidden state: h_t = tanh(W_hh * h_{t-1} + W_hx * x_t + b_h)
            h = self.tanh(np.dot(self.W_hh, h_prev) + np.dot(x, self.W_hx) + self.b_h)
            hidden_states.append(h)
            
            # Output: y_t = softmax(W_S * h_t + b_S)
            scores = np.dot(h, self.W_S) + self.b_S
            output = self.softmax(scores)
            outputs.append(output)
            
            h_prev = h
        
        return hidden_states, outputs
    
    def predict_next_word(self, sequence):
        """Dự đoán từ tiếp theo"""
        hidden_states, outputs = self.forward(sequence)
        last_output = outputs[-1]
        predicted_idx = np.argmax(last_output)
        return predicted_idx
    
    def compute_loss(self, sequence, next_words):
        """
        Tính Cross-Entropy Loss
        
        Args:
            sequence: Input sequence
            next_words: True next words cho mỗi position
        """
        _, outputs = self.forward(sequence)
        total_loss = 0
        
        for output, true_word in zip(outputs, next_words):
            loss = -np.log(output[true_word] + 1e-8)
            total_loss += loss
        
        return total_loss / len(sequence)
    
    def backward(self, sequence, next_words, learning_rate=0.01):
        """Backward pass với BPTT (Backpropagation Through Time)"""
        # Forward pass để lưu activations
        hidden_states, outputs = self.forward(sequence)
        
        # Khởi tạo gradients
        dW_hh = np.zeros_like(self.W_hh)
        dW_hx = np.zeros_like(self.W_hx)
        dW_S = np.zeros_like(self.W_S)
        db_h = np.zeros_like(self.b_h)
        db_S = np.zeros_like(self.b_S)
        dW_emb = np.zeros_like(self.W_emb)
        
        dh_next = np.zeros(self.hidden_dim)  # Gradient từ tương lai
        
        # Backward qua time
        for t in reversed(range(len(sequence))):
            # Output layer gradients
            dy = outputs[t].copy()
            dy[next_words[t]] -= 1  # Cross-entropy với one-hot
            
            dW_S += np.outer(hidden_states[t], dy)
            db_S += dy
            dh = np.dot(self.W_S, dy) + dh_next
            
            # Hidden layer gradients (với tanh derivative)
            dh_raw = dh * (1 - hidden_states[t] ** 2)  # tanh derivative
            
            dW_hh += np.outer(hidden_states[t-1] if t > 0 else np.zeros(self.hidden_dim), dh_raw)
            dW_hx += np.outer(self.W_emb[sequence[t]], dh_raw)
            db_h += dh_raw
            
            dW_emb[sequence[t]] += np.dot(dh_raw, self.W_hx.T)
            
            # Gradient cho hidden state trước đó
            dh_next = np.dot(self.W_hh.T, dh_raw)
        
        # Gradient clipping để tránh exploding gradient
        max_norm = 5.0
        total_norm = np.sqrt(sum([np.sum(g**2) for g in [dW_hh, dW_hx, dW_S]]))
        if total_norm > max_norm:
            clip_coef = max_norm / total_norm
            dW_hh *= clip_coef
            dW_hx *= clip_coef
            dW_S *= clip_coef
        
        # Update weights
        self.W_hh -= learning_rate * dW_hh
        self.W_hx -= learning_rate * dW_hx
        self.W_S -= learning_rate * dW_S
        self.b_h -= learning_rate * db_h
        self.b_S -= learning_rate * db_S
        self.W_emb -= learning_rate * dW_emb

# Ví dụ sử dụng
def train_language_model(corpus, vocab_size, epochs=10):
    """Train RNN Language Model"""
    model = SimpleRNN(vocab_size, embedding_dim=100, hidden_dim=128)
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        
        for sentence in corpus:
            if len(sentence) < 2:
                continue
            
            # Tạo training pairs: sequence và next words
            sequence = sentence[:-1]
            next_words = sentence[1:]
            
            # Forward và backward
            loss = model.compute_loss(sequence, next_words)
            model.backward(sequence, next_words)
            
            total_loss += loss
            count += 1
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/count:.4f}")
    
    return model
```

### 6.4 Những thách thức trong việc Huấn luyện RNN

**Vấn đề Tiêu biến và Bùng nổ Gradient.** Mặc dù có tiềm năng lớn, việc huấn luyện RNN gặp phải một thách thức nghiêm trọng. Khi lan truyền ngược lỗi qua nhiều bước thời gian, gradient được nhân với ma trận trọng số $W_{hh}$ lặp đi lặp lại. Nếu các giá trị trong $W_{hh}$ nhỏ, gradient sẽ co lại theo cấp số nhân và trở nên nhỏ dần đến 0 (**tiêu biến gradient**). Ngược lại, nếu các giá trị lớn, gradient sẽ tăng lên theo cấp số nhân và trở nên cực lớn (**bùng nổ gradient**).

**Hậu quả.** Tiêu biến gradient khiến mô hình gặp khó khăn nghiêm trọng trong việc học các phụ thuộc dài hạn, vì tín hiệu lỗi từ các bước thời gian xa xôi bị mất đi. Bùng nổ gradient làm cho quá trình huấn luyện không ổn định, thường dẫn đến giá trị NaN.

**Giải pháp.** Một giải pháp đơn giản và hiệu quả cho vấn đề bùng nổ gradient là **cắt xén gradient** (*gradient clipping*): nếu norm của vector gradient vượt quá một ngưỡng, nó sẽ được thu nhỏ lại. Để giải quyết vấn đề tiêu biến gradient, các kiến trúc phức tạp hơn như **LSTM** và **GRU** đã được phát triển.

### 6.5 Chuyển tiếp

Để giải quyết hiệu quả vấn đề tiêu biến gradient và cho phép mô hình học các phụ thuộc dài hạn tốt hơn, các nhà nghiên cứu đã thiết kế các kiến trúc RNN nâng cao với các cơ chế cổng (*gating mechanisms*) tinh vi, sẽ được trình bày trong phần tiếp theo.

***

## 7.0 Các Kiến trúc RNN Nâng cao: GRU và LSTM

### 7.1 Mở đầu: Quản lý Bộ nhớ một cách Tường minh

Một RNN đơn giản có một trạng thái ẩn duy nhất liên tục bị ghi đè, giống như một bộ nhớ ngắn hạn bị quá tải. Điều này dẫn đến vấn đề tiêu biến gradient. Bước đột phá của các kiến trúc như **GRU** và **LSTM** là giới thiệu khái niệm quản lý bộ nhớ tường minh thông qua các cơ chế cổng (*gating mechanisms*). Các cổng này hoạt động như những "người điều khiển giao thông" cho luồng thông tin, quyết định một cách linh hoạt thông tin nào cần được giữ lại, thông tin nào cần loại bỏ, và thông tin nào cần được ghi mới.

### 7.2 Đơn vị Hồi quy có Cổng (Gated Recurrent Unit - GRU)

**GRU** là một phiên bản đơn giản hơn của LSTM nhưng vẫn duy trì được hiệu suất mạnh mẽ. Nó sử dụng hai cổng chính để điều khiển luồng thông tin:

- **Cổng cập nhật** (*Update Gate* $z_t$): Cổng này quyết định lượng thông tin từ trạng thái ẩn trước đó ($h_{t-1}$) cần được giữ lại và chuyển sang trạng thái hiện tại. Khi giá trị của cổng này gần bằng 1, phần lớn trạng thái ẩn trước đó được sao chép sang trạng thái mới, cho phép thông tin tồn tại qua nhiều bước thời gian.

- **Cổng đặt lại** (*Reset Gate* $r_t$): Cổng này quyết định lượng thông tin từ trạng thái ẩn trước đó cần được "quên đi" khi tính toán bộ nhớ mới. Nếu cổng này có giá trị gần bằng 0, mô hình sẽ bỏ qua trạng thái ẩn trước đó và tạo ra một trạng thái mới chủ yếu dựa trên đầu vào hiện tại.

### 7.3 Bộ nhớ Dài-Ngắn hạn (Long Short-Term Memory - LSTM)

**LSTM** là một kiến trúc thậm chí còn phức tạp và mạnh mẽ hơn, và là nền tảng cho nhiều thành tựu trong học sâu. Sự đổi mới cốt lõi của LSTM là nó duy trì một thành phần bộ nhớ riêng biệt được gọi là **trạng thái ô** (*cell state* $c_t$). Trạng thái ô này có thể được hình dung như một "xa lộ bộ nhớ" hay một "băng chuyền thông tin", cho phép thông tin đi qua gần như không thay đổi qua nhiều bước thời gian. Các cổng của LSTM đóng vai trò như các lối vào và lối ra của xa lộ này, điều khiển một cách tinh vi những gì được thêm vào, loại bỏ khỏi, hoặc đọc từ bộ nhớ.

**Các cổng chính của LSTM.** LSTM sử dụng ba cổng chính để điều khiển trạng thái ô và trạng thái ẩn:

1. **Cổng quên** (*Forget Gate* $f_t$): Quyết định thông tin nào trong trạng thái ô trước đó ($c_{t-1}$) sẽ bị loại bỏ.
2. **Cổng đầu vào** (*Input Gate* $i_t$): Quyết định thông tin mới nào sẽ được lưu trữ vào trạng thái ô hiện tại.
3. **Cổng đầu ra** (*Output Gate* $o_t$): Quyết định phần nào của trạng thái ô sẽ được xuất ra dưới dạng trạng thái ẩn ($h_t$).

Sự tách biệt giữa trạng thái ô (bộ nhớ dài hạn) và trạng thái ẩn (bộ nhớ làm việc), cùng với các cơ chế cổng tinh vi, cho phép LSTM học và duy trì các phụ thuộc trong những khoảng thời gian rất dài, giải quyết hiệu quả vấn đề tiêu biến gradient.

**Triển khai Code (Python với NumPy):**

```python
import numpy as np

class LSTM:
    """Long Short-Term Memory Network"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Khởi tạo embeddings
        self.W_emb = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # LSTM weights (gộp tất cả gates)
        # W_f, W_i, W_o, W_g cho forget, input, output, và cell gate
        self.W = np.random.randn(embedding_dim + hidden_dim, 4 * hidden_dim) * 0.01
        self.b = np.zeros(4 * hidden_dim)
    
    def sigmoid(self, x):
        """Hàm sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def tanh(self, x):
        """Hàm tanh"""
        return np.tanh(x)
    
    def forward_step(self, x_t, h_prev, c_prev):
        """
        Forward một bước thời gian
        
        Args:
            x_t: Input tại time t (embedding)
            h_prev: Hidden state trước đó
            c_prev: Cell state trước đó
        
        Returns:
            h_t: Hidden state mới
            c_t: Cell state mới
        """
        # Concatenate input và hidden state
        concat = np.concatenate([x_t, h_prev])
        
        # Tính tất cả gates cùng lúc
        gates = np.dot(concat, self.W) + self.b
        
        # Chia thành 4 gates
        f_t = self.sigmoid(gates[:self.hidden_dim])  # Forget gate
        i_t = self.sigmoid(gates[self.hidden_dim:2*self.hidden_dim])  # Input gate
        o_t = self.sigmoid(gates[2*self.hidden_dim:3*self.hidden_dim])  # Output gate
        g_t = self.tanh(gates[3*self.hidden_dim:])  # Candidate values
        
        # Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
        c_t = f_t * c_prev + i_t * g_t
        
        # Update hidden state: h_t = o_t * tanh(c_t)
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t, (f_t, i_t, o_t, g_t)
    
    def forward(self, sequence):
        """Forward pass qua toàn bộ sequence"""
        hidden_states = []
        cell_states = []
        gates_list = []
        
        h_prev = np.zeros(self.hidden_dim)
        c_prev = np.zeros(self.hidden_dim)
        
        for word_idx in sequence:
            x_t = self.W_emb[word_idx]
            h_t, c_t, gates = self.forward_step(x_t, h_prev, c_prev)
            
            hidden_states.append(h_t)
            cell_states.append(c_t)
            gates_list.append(gates)
            
            h_prev = h_t
            c_prev = c_t
        
        return hidden_states, cell_states, gates_list

class GRU:
    """Gated Recurrent Unit"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Khởi tạo embeddings
        self.W_emb = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # GRU weights: reset gate và update gate
        self.W_r = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.01
        self.W_z = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.01
        self.W_h = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.01
        
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward_step(self, x_t, h_prev):
        """
        Forward một bước thời gian cho GRU
        
        Args:
            x_t: Input tại time t
            h_prev: Hidden state trước đó
        
        Returns:
            h_t: Hidden state mới
        """
        # Concatenate
        concat = np.concatenate([x_t, h_prev])
        
        # Reset gate: r_t = sigmoid(W_r * [x_t, h_{t-1}] + b_r)
        r_t = self.sigmoid(np.dot(concat, self.W_r) + self.b_r)
        
        # Update gate: z_t = sigmoid(W_z * [x_t, h_{t-1}] + b_z)
        z_t = self.sigmoid(np.dot(concat, self.W_z) + self.b_z)
        
        # Candidate hidden state: h_tilde = tanh(W_h * [x_t, r_t * h_{t-1}] + b_h)
        h_tilde = self.tanh(
            np.dot(np.concatenate([x_t, r_t * h_prev]), self.W_h) + self.b_h
        )
        
        # Final hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t, (r_t, z_t)
    
    def forward(self, sequence):
        """Forward pass qua toàn bộ sequence"""
        hidden_states = []
        gates_list = []
        
        h_prev = np.zeros(self.hidden_dim)
        
        for word_idx in sequence:
            x_t = self.W_emb[word_idx]
            h_t, gates = self.forward_step(x_t, h_prev)
            
            hidden_states.append(h_t)
            gates_list.append(gates)
            
            h_prev = h_t
        
        return hidden_states, gates_list

# Ví dụ so sánh RNN, LSTM, GRU
def compare_architectures(sequence, vocab_size):
    """So sánh khả năng giữ thông tin của các kiến trúc"""
    from SimpleRNN import SimpleRNN
    
    rnn = SimpleRNN(vocab_size, hidden_dim=128)
    lstm = LSTM(vocab_size, hidden_dim=128)
    gru = GRU(vocab_size, hidden_dim=128)
    
    # Forward pass
    rnn_hidden, _ = rnn.forward(sequence)
    lstm_hidden, lstm_cell, _ = lstm.forward(sequence)
    gru_hidden, _ = gru.forward(sequence)
    
    print(f"RNN final hidden state norm: {np.linalg.norm(rnn_hidden[-1]):.4f}")
    print(f"LSTM final hidden state norm: {np.linalg.norm(lstm_hidden[-1]):.4f}")
    print(f"LSTM final cell state norm: {np.linalg.norm(lstm_cell[-1]):.4f}")
    print(f"GRU final hidden state norm: {np.linalg.norm(gru_hidden[-1]):.4f}")
```






   



