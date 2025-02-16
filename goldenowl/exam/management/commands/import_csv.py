import csv
import itertools
from django.core.management.base import BaseCommand
from exam.models import Student

def safe_float(value):
    try:
        return float(value) if value.strip() != "" else 0.0
    except ValueError:
        return 0.0

class Command(BaseCommand):
    help = "Import data from diem_thi_thpt_2024.csv (chỉ lấy 20.000 dòng đầu tiên)"

    def handle(self, *args, **kwargs):
        file_path = "F:\go_trainer\goldenowl\diem_thi_thpt_2024.csv"  
        with open(file_path, encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  
            for row in itertools.islice(reader, 20000):
                Student.objects.get_or_create(
                    sbd=row[0],
                    defaults={
                        "toan": safe_float(row[1]),
                        "ngu_van": safe_float(row[2]),
                        "ngoai_ngu": safe_float(row[3]),
                        "vat_ly": safe_float(row[4]),
                        "hoa_hoc": safe_float(row[5]),
                        "sinh_hoc": safe_float(row[6]),
                        "lich_su": safe_float(row[7]),
                        "dia_ly": safe_float(row[8]),
                        "giao_duc_cong_dan": safe_float(row[9]),
                        "ma_ngoai_ngu": row[10],
                    }
                )

        self.stdout.write(self.style.SUCCESS("Finished import 20.000 dòng đầu tiên"))
