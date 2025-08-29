import os
import glob
import cv2
import pickle as pkl
from tqdm import tqdm

# Convert test images to pickle format
image_path = 'data/14_test_images'
image_out = 'data/test_image.pkl'

images = glob.glob(os.path.join(image_path, '*.bmp'))
image_dict = {}

print(f"Processing {len(images)} test images...")
for item in tqdm(images):
    img = cv2.imread(item)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Remove '_0.bmp' from filename to match label format
        key = os.path.basename(item).replace('_0.bmp', '')
        image_dict[key] = img

with open(image_out, 'wb') as f:
    pkl.dump(image_dict, f)

print(f"Saved {len(image_dict)} images to {image_out}")

# Note: For test labels, the inference script expects a different format
# than what's in test_caption.txt. The model expects hybrid tree format
# with parent-child relationships, not just LaTeX expressions.
print("\nNote: The test_caption.txt contains LaTeX expressions, but the model")
print("expects hybrid tree format. You would need to run gen_hybrid_data.py")
print("to convert the LaTeX to the proper format for full inference.")