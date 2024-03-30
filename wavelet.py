import pywt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew, kurtosis, entropy
import os

class waveletTransformer():
    def __init__(self):
        self.counter = 0

    def analyze_image_wavelet_stats(self,image_path):
        # Load the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)

        # Perform a single-level discrete 2D wavelet transform
        coeffs = pywt.dwt2(img_array, 'haar')
        cA, (cH, cV, cD) = coeffs

        # Calculate statistics for the approximation coefficients
        variance = np.var(cA)
        skewness = skew(cA.flatten())
        curtosis = kurtosis(cA.flatten())
        ent = entropy(cA.flatten())

        return variance, skewness, curtosis, ent

    def process_images_in_directory(self,directory_path, output_csv_path,flag):
        results = []

        # Process each image in the directory
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_path = os.path.join(directory_path, filename)
                variance, skewness, curtosis, ent = self.analyze_image_wavelet_stats(image_path)
                results.append([self.counter, variance, skewness, curtosis, ent,flag%2])
                self.counter += 1
    
        # Save the results to a CSV file
        df = pd.DataFrame(results, columns=['ID', 'Variance', 'Skewness', 'Curtosis', 'Entropy','Flag'])
        if flag == 0:
            fileName = 'fake_r_c'
        elif flag == 1:
            fileName = 'real_r'
        elif flag == 2:
            fileName = 'fake_r_bw'
        elif flag == 3:
            fileName = 'real_c'
        elif flag == 4:
            fileName = 'fake_c_c'
        elif flag == 6:
            fileName = 'fake_c_bw'
        else:
            raise ValueError("Invalid flag")
        df.to_csv(f"{output_csv_path}/{fileName}.csv", index=False)
        
wT = waveletTransformer()
output_csv_path = "/home/cc/data/bills"
for i in range(7):
    flag = i
    if flag == 0:
        directory_path = "/home/cc/data/bills/fakeBill_raw/Colored Note"
    elif flag == 1:
        directory_path = "/home/cc/data/bills/realBill_raw"
    elif flag == 2:
        directory_path = "/home/cc/data/bills/fakeBill_raw/Black and White Note"
    elif flag == 3:
        directory_path = "/home/cc/data/bills/realBill_cropped"
    elif flag == 4:
        directory_path = "/home/cc/data/bills/fakeBill_cropped/Colored Note"
    elif flag == 5:
        continue
    elif flag == 6:
        directory_path = "/home/cc/data/bills/fakeBill_cropped/Black and White Note"

    wT.process_images_in_directory(directory_path,output_csv_path,flag)

print("Done")