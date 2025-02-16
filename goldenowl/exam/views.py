import json
from django.shortcuts import render, get_object_or_404
from .models import Student

def dashboard_view(request):
    return render(request, "exam/dashboard.html")

def search_scores_view(request):
    student = None
    if request.method == "POST":
        sbd = request.POST.get('sbd')
        if sbd:
            student = get_object_or_404(Student, sbd=sbd)

    return render(request, "exam/search_scores.html", {"student": student})

def report_view(request):
    students = Student.objects.all()
    levels_dict = {
        ">=8 điểm": 0,
        "8 > && >=6 điểm": 0,
        "6 > && >=4 điểm": 0,
        "< 4 điểm": 0,
    }
    for s in students:
        levels_dict[s.score_level()] += 1

    levels_json = json.dumps(levels_dict, ensure_ascii=False)

    return render(request, "exam/report.html", {"levels_json": levels_json})

def top_group_a_view(request):
    students = Student.objects.all()
    students_sorted = sorted(students, key=lambda s: s.group_a_score(), reverse=True)
    top_10 = students_sorted[:10]
    return render(request, "exam/top_group_a.html", {"top_10": top_10})
