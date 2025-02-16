from django.db import models

class Student(models.Model):
    sbd = models.CharField("Số báo danh", max_length=20, unique=True)
    toan = models.FloatField("Toán")
    ngu_van = models.FloatField("Ngữ văn")
    ngoai_ngu = models.FloatField("Ngoại ngữ")
    vat_ly = models.FloatField("Vật lý")
    hoa_hoc = models.FloatField("Hóa học")
    sinh_hoc = models.FloatField("Sinh học")
    lich_su = models.FloatField("Lịch sử")
    dia_ly = models.FloatField("Địa lý")
    giao_duc_cong_dan = models.FloatField("Giáo dục công dân")
    ma_ngoai_ngu = models.CharField("Mã ngoại ngữ", max_length=10)

    def average_score(self):
        """Tính điểm trung bình 9 môn (không tính mã ngoại ngữ)."""
        total = (
            self.toan + self.ngu_van + self.ngoai_ngu +
            self.vat_ly + self.hoa_hoc + self.sinh_hoc +
            self.lich_su + self.dia_ly + self.giao_duc_cong_dan
        )
        return round(total / 9, 2)

    def score_level(self):
        """Xếp loại theo điểm trung bình."""
        avg = self.average_score()
        if avg >= 8:
            return ">=8 điểm"
        elif avg >= 6:
            return "8 > && >=6 điểm"
        elif avg >= 4:
            return "6 > && >=4 điểm"
        else:
            return "< 4 điểm"

    def group_a_score(self):
        """Điểm tổng 3 môn khối A: Toán, Lý, Hóa."""
        return self.toan + self.vat_ly + self.hoa_hoc

    def __str__(self):
        return f"{self.sbd} - {self.average_score()}"
