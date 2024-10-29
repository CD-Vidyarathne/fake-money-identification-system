import cv2
import math


# -------- Nadeesha---------------
def resize_image(image):
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_height / original_width
    new_height = int(500 * aspect_ratio)
    r_image = cv2.resize(image, (500, new_height), interpolation=cv2.INTER_AREA)
    # cv2.imshow("Resized Image",r_image)
    return r_image


# --------- Yeshara---------------
def grayScale_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def enhance_image(image):
    gray = grayScale_image(image)
    # cv2.imshow("GreyScale Image",gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # cv2.imshow("Enhanced Image",enhanced)
    return enhanced


def blur_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


# ---------- Aadhya-------------------------------
def preprocess_image(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Binary Image",binary)
    return binary


def detect_edges_image(image):
    edges = cv2.Canny(image, 100, 200)
    # cv2.imshow("Edges",edges)
    return edges


# ------------- Thenuja ----------------------
def find_largest_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def check_if_cropping_needed(image, color_image):
    largest_contour = find_largest_contour(image)

    if largest_contour is None:
        print("No contour detected")
        return False

    x, y, w, h = cv2.boundingRect(largest_contour)
    printable = color_image.copy()

    cv2.drawContours(printable, [largest_contour], -1, (0, 255, 0), 2)
    cv2.rectangle(printable, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Original Image with Contour", printable)
    img_height, img_width = image.shape[:2]

    bounding_box_area = w * h
    image_area = img_height * img_width

    ratio = bounding_box_area / image_area

    if ratio < 0.9 and ratio > 0.3:
        return True
    else:
        return False


def crop_image(image, color_image):
    # cv2.imshow("Image", image)
    large = find_largest_contour(image)
    x, y, w, h = cv2.boundingRect(large)
    crop = image[y : y + h, x : x + w]
    crop_color = color_image[y : y + h, x : x + w]
    res = resize_image(crop)
    res_color = resize_image(crop_color)

    cv2.imshow("Cropped Currency Note", res_color)
    return res, res_color


# --------------------- Chamindu ------------------------------------
def detect_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
    # sift = cv2.SIFT_create()
    # keypoints, descriptors = sift.detectAndCompute(image, None)
    # return keypoints, descriptors


def compare_images(image1, image2):
    keypoints1, descriptors1 = detect_features(image1)
    keypoints2, descriptors2 = detect_features(image2)

    if descriptors1 is None or descriptors2 is None:
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches, keypoints1, keypoints2


# ------------------ Nadeesha ----------------------------
def compare_histograms(image1, image2):
    gray1 = grayScale_image(image1)
    gray2 = grayScale_image(image2)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(similarity)
    return similarity


# ---------------- Chamindu -----------------------------
def check_currency_validity(ref_image, test_image, threshold=0.55):
    ref_resized_img = resize_image(ref_image)
    ref_enhanced_img = enhance_image(ref_resized_img)
    ref_blur_img = blur_image(ref_enhanced_img)
    ref_pre_img = preprocess_image(ref_enhanced_img)
    ref_edges = detect_edges_image(ref_pre_img)

    test_resized_img = resize_image(test_image)
    test_enhanced_img = enhance_image(test_resized_img)
    test_blur_img = blur_image(test_enhanced_img)
    test_pre_img = preprocess_image(test_enhanced_img)
    test_edges = detect_edges_image(test_pre_img)
    if check_if_cropping_needed(test_edges, test_resized_img):
        test_cropped, test_cropped_color = crop_image(test_edges, test_resized_img)
    else:
        test_cropped = test_edges
        test_cropped_color = test_resized_img
    matches, kp1, kp2 = compare_images(ref_edges, test_cropped)
    histo_similarity = compare_histograms(ref_resized_img, test_cropped_color)

    if matches is None:
        print("Could not find matches.")
        return "Error: Unable to compare images"

    if matches and kp1 and kp2:
        output_img = ref_image.copy()
        matched_image = cv2.drawMatches(
            ref_resized_img,
            kp1,
            test_cropped_color,
            kp2,
            matches[:20],
            output_img,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        matched_image = None

    total_matches = len(matches) if matches else 0
    similarity = total_matches / min(len(kp1), len(kp2)) if total_matches > 0 else 0
    total_similarity = math.sqrt(similarity**2 + histo_similarity**2)
    print(total_similarity)

    if total_similarity >= threshold:
        return "Real Currency", matched_image
    else:
        return "Fake Currency", matched_image


REF_IMAGE_PATH = "C:/Users/chami/Code/Uni_Work/DIP/project/image/ref.png"
TEST_IMAGE_PATH = "C:/Users/chami/Code/Uni_Work/DIP/project/image/real.jpg"
FAKE_IMAGE_PATH = "C:/Users/chami/Code/Uni_Work/DIP/project/image/fake.jpg"
ref_image = cv2.imread(REF_IMAGE_PATH, cv2.IMREAD_COLOR)
real_image = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_COLOR)
fake_image = cv2.imread(FAKE_IMAGE_PATH, cv2.IMREAD_COLOR)

result, matched_image = check_currency_validity(ref_image, real_image)

print(result)
if result == "Real Currency":
    cv2.imshow("Matched Features", matched_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
