from config import thumbnail_dir
import urllib
from pdf2image import convert_from_path
from PIL import Image
from os.path import join, basename, exists
import os


def get_url(arxiv_id):
    # Get url from arxiv_id
    # Id should be in YYMM.NNNNN format but sometimes it's not
    # so we have to add a leading 0 if necessary
    arxiv_id = str(arxiv_id)
    try:
        first_part = arxiv_id.split(".")[0]
        second_part = arxiv_id.split(".")[1]
    except:
        # Weird ids like quant-ph/0207118
        return "https://arxiv.org/abs/" + arxiv_id

    if len(first_part) != 4:
        while len(first_part) < 4:
            first_part = "0" + first_part

    if len(second_part) != 5:
        while len(second_part) < 5:
            second_part = "0" + second_part

    return "https://arxiv.org/abs/" + first_part + "." + second_part


def get_thumbnail(arxiv_url):
    """Generate a thumbnail for the given arxiv url"""

    # Get the Paths
    arxiv_id = arxiv_url.split("/")[-1]
    pdf_path = join(thumbnail_dir, arxiv_id + ".pdf")
    thumbnail_path = join(thumbnail_dir, arxiv_id + ".png")

    # If thumbnail already exists, return
    if exists(thumbnail_path):
        return thumbnail_path

    # Get the thumbnail url
    # NOTE: There is also an e-print url, which contains the latex source code. This could be used to do some cool stuff
    thumbnail_url = arxiv_url.replace("abs", "pdf")

    # Download pdf, concatenate the pages horizontally and convert to png
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(thumbnail_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        with open(pdf_path, "wb") as f:
            f.write(resp.content)

        # Read pdf
        images = convert_from_path(pdf_path)
    except Exception as e:
        if exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                pass
        return None

    # Generate thumbnail by concatenating the pdf pages horizontally and converting to png

    if len(images) > 8:
        images = images[:8]

    # Concatenate the pages horizontally
    width, height = images[0].size
    new_im = Image.new("RGB", (width * len(images), height))
    for i, im in enumerate(images):
        new_im.paste(im, (i * width, 0))

    # Save the image with less quality
    new_im.save(thumbnail_path, dpi=(100, 100))

    # Delete the pdf
    os.remove(pdf_path)

    print(f"[+] Generated thumbnail for {arxiv_id}")

    return thumbnail_path
