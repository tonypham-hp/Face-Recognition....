# import face_recognition
# import cv2
# import numpy as np
# import glob
# import time
# import csv  
# import pickle



# f=open("ref_name.pkl","rb")
# ref_dictt=pickle.load(f)         #ref_dict=ref vs name
# f.close()

# f=open("ref_embed.pkl","rb")
# embed_dictt=pickle.load(f)      #embed_dict- ref  vs embedding 
# f.close()

# ############################################################################  encodings and ref_ids 
# known_face_encodings = []  #encodingd of faces
# known_face_names = []	   #ref_id of faces



# for ref_id , embed_list in embed_dictt.items():
# 	for embed in embed_list:
# 		known_face_encodings +=[embed]
# 		known_face_names += [ref_id]
   												


# #############################################################frame capturing from camera and face recognition
# video_capture = cv2.VideoCapture(0)
# # Initialize some variables
# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True

# while True  :
# 	# Grab a single frame of video
# 	ret, frame = video_capture.read()

# 	# Resize frame of video to 1/4 size for faster face recognition processing
# 	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# 	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
# 	#rgb_small_frame = small_frame[:, :, ::-1]
# 	rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
# 	print("small_frame:", small_frame.shape, small_frame.dtype)
# 	print("rgb_small_frame:", rgb_small_frame.shape, rgb_small_frame.dtype)
	




# 	# Only process every other frame of video to save time
# 	if process_this_frame:
# 		# Find all the faces and face encodings in the current frame of video
# 		face_locations = face_recognition.face_locations(rgb_small_frame)
# 		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

# 		face_names = []
# 		for face_encoding in face_encodings:
# 			# See if the face is a match for the known face(s)
# 			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
# 			name = "Unknown"

# 			# # If a match was found in known_face_encodings, just use the first one.
# 			# if True in matches:
# 			#     first_match_index = matches.index(True)
# 			#     name = known_face_names[first_match_index]

# 			# Or instead, use the known face with the smallest distance to the new face
# 			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
# 			best_match_index = np.argmin(face_distances)
# 			if matches[best_match_index]:
# 				name = known_face_names[best_match_index]
# 			face_names.append(name)

# 	process_this_frame = not process_this_frame


# 	# Display the results
# 	for (top, right, bottom, left), name in zip(face_locations, face_names):
# 		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
# 		top *= 4
# 		right *= 4
# 		bottom *= 4
# 		left *= 4

# 		              #updating in database

# 		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

# 		# Draw a label with a name below the face
# 		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
# 		font = cv2.FONT_HERSHEY_DUPLEX
# 		cv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
# 	font = cv2.FONT_HERSHEY_DUPLEX
# 	# cv2.putText(frame, last_rec[0], (6,20), font, 1.0, (0,0 ,0), 1)

# 	# Display the resulting imagecv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
# 	cv2.imshow('Video', frame)

# 	# Hit 'q' on the keyboard to quit!
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		# t.cancel()
# 		break

# 		# break

# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()

# import face_recognition
# import cv2
# import numpy as np
# import pickle

# # Load name dict
# with open("ref_name.pkl", "rb") as f:
#     ref_dictt = pickle.load(f)

# # Load embedding dict
# with open("ref_embed.pkl", "rb") as f:
#     embed_dictt = pickle.load(f)

# # Prepare lists
# known_face_encodings = []
# known_face_names = []

# for ref_id, embed_list in embed_dictt.items():
#     for embed in embed_list:
#         known_face_encodings.append(embed)
#         known_face_names.append(ref_id)

# # Open camera
# video_capture = cv2.VideoCapture(0)
# cv2.namedWindow("Video", cv2.WINDOW_NORMAL)


# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True

# print("Press 'q' to quit recognition")

# while True:
#     ret, frame = video_capture.read()   

#     # webcam failed → skip
#     if not ret or frame is None:
#         continue

#     # normalize dtype
#     if frame.dtype != "uint8":
#         frame = frame.astype("uint8")

#     # Resize for faster recognition
#     try:
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     except:
#         continue

#     # If BGRA → convert to BGR
#     if len(small_frame.shape) == 3 and small_frame.shape[2] == 4:
#         small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGRA2BGR)

#     # If gray → convert to BGR
#     if len(small_frame.shape) == 2:
#         small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2BGR)

#     # Convert BGR → RGB (face_recognition requirement)
#     try:
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#     except:
#         # fallback
#         rgb_small_frame = small_frame[:, :, ::-1]

#     # Process every 2nd frame
#     if process_this_frame:
#         try:
#             face_locations = face_recognition.face_locations(rgb_small_frame)
#             face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
#         except:
#             continue

#         face_names = []

#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"

#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)

#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]   # name = ref_id

#             face_names.append(name)

#     process_this_frame = not process_this_frame

#     # Draw overlay
#         # Draw overlay
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         display_name = ref_dictt[name] if name in ref_dictt else name

#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
#                       (0, 0, 255), cv2.FILLED)
#         cv2.putText(frame, display_name, (left + 6, bottom - 6),
#                     cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)

#     # ---- ALWAYS SHOW FRAME (Fix freezing) ----
#     cv2.imshow('Video', frame)
#     cv2.moveWindow("Video", 100, 100)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()

import face_recognition
import cv2
import numpy as np
import pickle

# Load name dictionary
with open("ref_name.pkl", "rb") as f:
    ref_dictt = pickle.load(f)

# Load embeddings dictionary
with open("ref_embed.pkl", "rb") as f:
    embed_dictt = pickle.load(f)

# Prepare encoded data
known_face_encodings = []
known_face_ids = []

for ref_id, embed_list in embed_dictt.items():
    for emb in embed_list:
        known_face_encodings.append(emb)
        known_face_ids.append(ref_id)

# Open webcam
video_capture = cv2.VideoCapture(0)

# Force stable webcam format
video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
video_capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

print("Press 'q' to quit recognition")

while True:
    ret, frame = video_capture.read()

    if not ret or frame is None:
        continue

    # Ensure correct dtype
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Convert grayscale → BGR
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Convert BGRA → BGR
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Make memory contiguous (dlib REQUIREMENT)
    frame = np.ascontiguousarray(frame)

    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = np.ascontiguousarray(small_frame)

    # Convert to RGB (dlib requirement)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

    # Face detection
    try:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    except Exception as e:
        print("ERROR in face_recognition:", e)
        face_locations = []
        face_encodings = []

    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Choose best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            id_detected = known_face_ids[best_match_index]
            name = ref_dictt.get(id_detected, id_detected)

        face_names.append(name)

    # Draw boxes
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                      (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)

    # Display video
    cv2.imshow("Video", frame)
    cv2.moveWindow("Video", 100, 100)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
