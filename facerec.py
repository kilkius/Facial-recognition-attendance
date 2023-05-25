import face_recognition, os, cv2, numpy, csv
from datetime import datetime

# input from webcam
capture = cv2.VideoCapture(0)

# check photos in db
elon_pic = face_recognition.load_image_file("photos/elon.jpeg")
elon_encoding = face_recognition.face_encodings(elon_pic)[0]

bezos_pic = face_recognition.load_image_file("photos/bezos.jpeg")
bezos_encoding = face_recognition.face_encodings(bezos_pic)[0]

known_face_encodings = [
    elon_encoding,
    bezos_encoding
]

known_face_names = [
    "elon",
    "bezos"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# open csv file
f = open(current_date+'.csv', 'w+', newline= '')
lnwriter = csv.writer(f)

while True:
    _, frame = capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

    if s:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match = numpy.argmin(face_distance)
            if matches[best_match]:
                name = known_face_names[best_match]

            # add name to csv
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H,%M,%S")
                    print(name, current_time)
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Attendance system", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
f.close()