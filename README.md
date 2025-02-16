# webdev-assignment
# Golden Owl Web Application

Ứng dụng này quản lý dữ liệu điểm thi THPT 2024 với các tính năng:
- Tra cứu điểm thi theo số báo danh (SBD).
- Báo cáo xếp loại thí sinh theo các mức (>=8, 8 > && >=6, 6 > && >=4, <4) bằng biểu đồ (Chart.js).
- Danh sách top 10 thí sinh khối A (Toán, Vật lý, Hóa học).

## Nội dung
- [Yêu Cầu](#yêu-cầu)
- [Cài Đặt và Chạy trên Localhost](#cài-đặt-và-chạy-trên-localhost)
- [Chạy Ứng Dụng với Docker](#chạy-ứng-dụng-với-docker)
- [Hướng Dẫn Up Code Lên GitHub từ VS Code](#hướng-dẫn-up-code-lên-github-từ-vs-code)
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
   ```python -m venv env
   # Trên Windows:
   env\Scripts\activate
   # Trên macOS/Linux:
   source env/bin/activate
   pip install -r requirements.txt
3. **Cấu Hình Cơ Sở Dữ Liệu (goldenowl/settings.py):**
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


