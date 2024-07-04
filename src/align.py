from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2 as cv


def align_images(images):
    gray_images = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in images]
    orb = cv.ORB.create(nfeatures=1000)  # Limit features for speed

    def process_image(i):
        if i == 0:
            return images[0]

        kp1, des1 = orb.detectAndCompute(gray_images[0], None)
        kp2, des2 = orb.detectAndCompute(gray_images[i], None)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        matches = matches[:100]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        sz = (images[0].shape[1], images[0].shape[0])
        M, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
        return cv.warpPerspective(images[i], M, sz)

    with ThreadPoolExecutor() as executor:
        return list(executor.map(process_image, range(len(images))))


def stack_images(images):
    return np.mean(images, axis=0).astype(np.uint8)


def load_images(files):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda f: cv.imread(str(f)), files))


def save_image(path, image):
    cv.imwrite(path, image)
