import cv2


def check_focus(image, threshold=55):
    """Check whether the image is in focus or not using variance of Laplacian

    Args:
        image (UMat): image array

    Returns:
        bool: True if the image is in focus, False otherwise
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variance = cv2.Laplacian(image, cv2.CV_64F).var()

    return variance > threshold
