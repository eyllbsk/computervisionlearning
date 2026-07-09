import cv2

image = cv2.imread("images/shapes.jpg")

if image is None:
    print("Resim yuklenemedi.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Bulunan kontur sayisi:", len(contours))

for contour in contours:
    area = cv2.contourArea(contour)

    if area > 500:
        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        corner_count = len(approx)

        x, y, w, h = cv2.boundingRect(approx)

        if corner_count == 3:
            shape = "Triangle"

        elif corner_count == 4:
            ratio = w / float(h)

            if 0.90 <= ratio <= 1.10:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif corner_count > 6:
            shape = "Circle"

        else:
            shape = "Polygon"

        print(shape, "Alan:", area, "Kose:", corner_count)

        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(
            image,
            shape,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

cv2.imshow("Shapes", image)
cv2.imshow("Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()