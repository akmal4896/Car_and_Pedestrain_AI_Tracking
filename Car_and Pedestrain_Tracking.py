# import opencv module
import cv2

def main():
    # open video
    # video = cv2.VideoCapture('Tesla Dashcam 2.mp4')
    video1 = cv2.VideoCapture('Motorcycle Dashcam.mp4')

    # our pre-trained car and pedestrian classifier
    classifier_file_car = 'car_detector.xml'
    classifier_file_pedestrian = 'haarcascade_fullbody.xml'

    initialize(classifier_file_car, classifier_file_pedestrian, video1)

    # Release the VideoCapture object
    video1.release()
    print('Code Completed')

def initialize(classifier_file_car, classifier_file_pedestrian, video1):
    # Create car classifier
    car_tracker = cv2.CascadeClassifier(classifier_file_car)
    pedestrian_tracker = cv2.CascadeClassifier(classifier_file_pedestrian)

    # Run forever until car shut down or crash
    while True:

        # read the current frame
        (read_successful, frame) = video1.read()

        # error checking
        if read_successful:
            # must convert frame to grayscale
            greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # detect cars
        cars = car_tracker.detectMultiScale(greyscale_frame)
        pedestrian = pedestrian_tracker.detectMultiScale(greyscale_frame)

        # Draw rectangles around the cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw rectangles around the cars
        for (x, y, w, h) in pedestrian:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Display the image with the faces spotted
        cv2.imshow('Car and Pedestrian Detection', frame)

        # Don,t auto close (Wait here in the code and listen for a key press)
        key = cv2.waitKey(1)

        # Break the loop if button 'Q' or 'q' is pressed
        if key == 81 or key == 113:
            break

# If file got import, dont run immediately
if __name__ == '__main__':
    main()

"""
# our image
img_file = 'Car Image 2.jpg'

# create opencv image
img = cv2.imread(img_file)  # read image pixel data into array

# convert to grayscale (needed for haar cascade)
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(grayscale)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


# Display the image with the faces spotted
cv2.imshow('Clever Programmer Car Detector', img)

# Dont autoclose (Wait here in the code and listen for a key press)
cv2.waitKey()

print('Code Completed')

"""
