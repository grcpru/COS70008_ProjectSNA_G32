from django import forms

class DatasetUploadForm(forms.Form):
    dataset = forms.FileField(label='Select a dataset file')