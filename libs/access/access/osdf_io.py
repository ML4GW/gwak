import os
from typing import Optional
from pathlib import Path
from pelicanfs.core import OSDFFileSystem
from time import time
import h5py



def pelican_read_latest_project_image(
    project_name: str,
    cit_user_name: Optional[str]=None
):
    #ToDo: test if project_name is under ["data", "train", "deploy"]
    project_images = []
    pelfs = OSDFFileSystem()
    if cit_user_name is None:
        cit_user_name = os.environ['USER']
    path = f"osdf:///igwn/cit/staging/{cit_user_name}/Container/GWAK/images/{project_name}"

    files = pelfs.ls(path)

    for file in files:
        project_images.append(file["name"])
    project_images = sorted(project_images)
    return f"osdf://{project_images[-1]}"