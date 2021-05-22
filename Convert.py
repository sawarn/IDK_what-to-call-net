import os
import PyPDF2
import pandas as pd
import glob



folder_path = r"C:\Users\Shivam\Desktop\New folder (3)"
# Specify the output jpg/png folder path
jpg_folder_path = r"C:\Users\Shivam\Desktop\New folder (3)"

for i in os.listdir(folder_path):
    images_path=glob.glob(os.path.join(folder_path,"*","*.pdf"))
    print(images_path)

for n, image in enumerate(images_path):
    ds = Image.open(os.path.join(folder_path, image))
    print(ds)
    """
    pixel_array_numpy = ds.pixel_array
    if PNG == True:
        image = image.replace('.pgm', '.png')
    else:
        image = image.replace('.pgm', '.png')
    cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
    if n % 50 == 0:
        print('{} image converted'.format(n))
"""