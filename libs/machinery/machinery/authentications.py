import logging
import subprocess


def check_scitoken():
    """
    # Activate the SciToken for GW data access.
    # This is required to access the data from the GW datafind service.
    # """

    print("")
    print("Check SciToken status.")

    result = subprocess.run(
        [
            "htgettoken", 
            "-a", 
            "vault.ligo.org", 
            "-i", 
            "igwn",
        ],
        text=True
    )

    if result.returncode == 0:
        print("    SciToken successfully activated. ")
    else:
        print("    SciToken failed...")
    print("")