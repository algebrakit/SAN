from strokes.inkml_parser import InkMLParser
from strokes.stroke2img import strokes_to_image, save_as_bmp
from strokes.scale_strokes import rescale_strokes
from inference_single import Inference
import numpy as np
import time

root = 'data/CHROME/CROHME_test_2011'
file1 = 'formulaire050-equation040'
file2 = 'formulaire050-equation070'
file3 = 'Inkdata_temp_InkFR_HPR_EQU_NOC_scc2_fi4_db135763'
file4 = 'Inkdata_temp_InkFR_HPR_EQU_NOC_scc54_fi4_db138003'
file5 = 'Inkdata_temp_InkFR_HPR_EQU_NOC_scc3_fi4_db135773'
file = file1

STROKE_LENGTH = 25 # in pixels

# get the strokes
parser = InkMLParser()
strokes, label = parser.parse_file(f'{root}/{file}.inkml')

# rescale based on the stroke lengths
strokes_norm, size = rescale_strokes(strokes, STROKE_LENGTH)
print('Image size=', size)

# convert to image
img = strokes_to_image(strokes_norm, image_size=(int(size[0])+4, int(size[1])+4), line_thickness=2, padding=2)
# save_as_bmp(img, 'test.bmp')

# convert to latex
print("\n=== Profiling Inference ===")
start_time = time.time()
inf = Inference(confPath='/Users/martijnslob/github/SAN/config.yaml')
setup_time = time.time() - start_time
print(f"Inference setup time: {setup_time:.4f} seconds")

start_time = time.time()
latex = inf.convert2latex(img)
conversion_time = time.time() - start_time
print(f"Conversion time: {conversion_time:.4f} seconds")
print(f"Total time: {setup_time + conversion_time:.4f} seconds")
print(f"\nPredicted LaTeX: {latex}")