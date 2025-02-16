# webdev-assignment
# Golden Owl Web Application
- [[Demo video]](https://drive.google.com/drive/folders/1PEn-pywJsZgHDjLfDqDk_tOWraOmpBRr?usp=sharing)
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






   



