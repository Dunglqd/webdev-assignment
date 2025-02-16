from django import forms

class StudentLookupForm(forms.Form):
    sbd = forms.CharField(label="Số báo danh", max_length=20)
