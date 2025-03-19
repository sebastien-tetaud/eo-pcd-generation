#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
File Downloader Tool
===========================================================

This script provides a function to download files from a given URL using wget.
The primary functionality includes:

- Downloading files from a URL
- Saving the downloaded files to a specified directory

Dependencies:
- os
- subprocess

Author: Sébastien Tétaud
Date: 2025-03-14
License: Apache 2.0
"""

import os
import subprocess

def download_file(url, download_dir):
    """
    Downloads a file from a given URL using wget.

    Parameters:
        url (str): The URL of the file to download.
        download_dir (str): The directory where the file will be saved.

    Returns:
        str: Path to the downloaded file, or None if the download fails.
    """
    file_name = os.path.join(download_dir, os.path.basename(url))

    try:
        # Construct and execute the wget command
        subprocess.run(['wget', '-O', file_name, url], check=True)
        print(f"Downloaded: {file_name}")
        return file_name
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        return None
