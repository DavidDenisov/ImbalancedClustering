import cv2
from skimage.color import rgb2lab, deltaE_ciede2000


def analyze_accuracy_ciede2000(img_gt, img_quantized):
    # Convert OpenCV BGR to standard RGB
    gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    q_rgb = cv2.cvtColor(img_quantized, cv2.COLOR_BGR2RGB)

    # Convert to standard CIELAB space float coordinates
    gt_lab = rgb2lab(gt_rgb)
    q_lab = rgb2lab(q_rgb)

    # Compute the perceptually weighted CIEDE2000 error array
    delta_e_matrix = deltaE_ciede2000(gt_lab, q_lab)

    # Return the mean error across all pixels
    return delta_e_matrix.mean()


# Load images
orig = cv2.imread('orig.png')
our = cv2.imread('our.png')
choice = cv2.imread('choice.png')
kmeans = cv2.imread('kmeans.png')

# Print evaluations
for name, img in [("Our", our), ("Choice", choice), ("K-Means", kmeans)]:
    color_err = analyze_accuracy_ciede2000(orig, img)
    print(f"--- {name} vs Ground Truth ---")
    print(f"Color Error (CIEDE2000):  {color_err:.4f} (Lower = Closer hue/saturation)\n")