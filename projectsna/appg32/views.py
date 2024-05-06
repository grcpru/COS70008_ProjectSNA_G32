import os
import pandas as pd
import dtale
import dtale.views
from django.shortcuts import render, HttpResponseRedirect
from .forms import DatasetUploadForm
from django.conf import settings

def landing_page(request):
    return render(request, 'landing_page.html')

def upload_data(request):
    if request.method == 'GET':
        return render(request, 'upload_form.html')

    elif request.method == 'POST':
        return HttpResponseRedirect('/success/')

def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset_file = form.cleaned_data['dataset']
            file_path = os.path.join(settings.MEDIA_ROOT, dataset_file.name)
            with open(file_path, 'wb') as destination:
                for chunk in dataset_file.chunks():
                    destination.write(chunk)
            return HttpResponseRedirect('/dataset-success/')
        else:
            return render(request, 'upload_dataset_form.html', {'form': form})
    else:
        form = DatasetUploadForm()
        return render(request, 'upload_dataset_form.html', {'form': form})

# """ def explore_datasets(request):
#     file_path = r'C:\Users\grcp_\Documents\GitHub\COS70008_ProjectSNA_G32\projectsna\media\trade_data_rice.csv'
#     if os.path.exists(file_path):
#         # Read the uploaded file as a pandas DataFrame
#         df = pd.read_csv(file_path)
        
#         # Start a D-Tale web server and get the URL
#         dtale_instance = dtale.show(df)
#         dtale_url = dtale_instance.main_url()
        
#         return render(request, 'explore_datasets.html', {'dtale_url': dtale_url})
#     else:
#         return render(request, 'no_dataset_uploaded.html') """

def explore_datasets(request):
    # Path to the media folder
    media_folder = 'media'
    # List all files in the media folder
    files = os.listdir(media_folder)
    return render(request, 'explore_datasets.html', {'files': files})

def explore_file(request, file):
    # Path to the selected file in the media folder
    file_path = os.path.join('media', file)
    # Read the selected file as a pandas DataFrame
    df = pd.read_csv(file_path)
    # Start a D-Tale web server and get the URL
    dtale_instance = dtale.show(df)
    dtale_url = dtale_instance.main_url()
    return render(request, 'explore_file.html', {'dtale_url': dtale_url})

def upload_success(request):
    return render(request, 'upload_success.html')
