import cv2

cars_classifier = cv2.CascadeClassifier('cars.xml')
bus_classifier = cv2.CascadeClassifier('Bus_front.xml')
bikes_classifier = cv2.CascadeClassifier('two_wheeler.xml')

cap = cv2.VideoCapture("../assets/pettah.mp4")

while (True):

    ret, img = cap.read()

    blur = cv2.blur(img, (3, 3))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    bikes = bus_classifier.detectMultiScale(gray)

    for (x, y, w, h) in bikes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Bike', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('LIVE', img)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
