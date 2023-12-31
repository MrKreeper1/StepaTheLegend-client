 #! /usr/bin/python

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
from flask import Flask, request, Response
import requests
from threading import Thread
import pybase64
import os, os.path
import urllib3

def read_config():
	with open("./config", "r") as f:
		lines = f.readlines()
	#print(lines)
	
	return lines[0][:-1], int(lines[1][:-1]), int(lines[2][:-1]), int(lines[3][:-1]), int(lines[4][:-1]), lines[5][:-1], int(lines[6])
	
def write_config(ip=-1, port=-1, show_im=-1, update_ol=-1, toff_ol=-1, serv_ip=-1, serv_port=-1):
	ip1, port1, show_im1, update_ol1, toff_ol1, serv_ip1, serv_port1 = read_config()
	ip2 = ip if ip != -1 else ip1
	port2 = port if port != -1 else port1
	show_im2 = show_im if show_im != -1 else show_im1
	update_ol2 = update_ol if update_ol != -1 else update_ol1
	toff_ol2 = toff_ol if toff_ol != -1 else toff_ol1
	serv_ip2 = serv_ip if serv_ip != -1 else serv_ip1
	serv_port2 = serv_port if serv_port != -1 else serv_port1
	
	with open("./config", "w") as f:
		f.write(f"{ip2}\n{port2}\n{show_im2}\n{update_ol2}\n{toff_ol2}\n{serv_ip2}\n{serv_port2}")
	print(f"[CONFIG]: Updating config to: {ip2} {port2} {show_im2} {update_ol2} {toff_ol2} {serv_ip2} {serv_port2}")
STEPA_KRUTOY_IP, STEPA_KRUTOY_PORT, SHOW_IMAGE, UPDATE_ON_LOAD, TURNOFF_UPDATING_OL, STEPA_SERVER_IP, STEPA_SERVER_PORT = read_config()

print(f"[CONFIG]: {STEPA_KRUTOY_IP} {STEPA_KRUTOY_PORT} {SHOW_IMAGE} {UPDATE_ON_LOAD} {TURNOFF_UPDATING_OL} {STEPA_SERVER_IP} {STEPA_SERVER_PORT}")

app = Flask("StepaKrutoyClient")

#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
def reload_encodings(rel = True):
	global data, encodingsP, flag
	flag = True
	if rel: print("[INFO] training -", os.system("python ./train_model.py"))
	print("[INFO] loading encodings + face detector...")
	with open(encodingsP, "rb") as f:
		data = pickle.loads(f.read())
	print("[INFO] loading encodings done")
	flag = False

# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam, could be your laptop webcam
# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
#vs = VideoStream(src=2,framerate=10).start()

reload_encodings(rel=UPDATE_ON_LOAD)
if UPDATE_ON_LOAD and TURNOFF_UPDATING_OL: write_config(update_ol=0, toff_ol=0)

#192.168.139.46:5000

frame2 = ""

def fl_serv():
	global app, frame2, STEPA_KRUTOY_IP, STEPA_KRUTOY_PORT, flag
	
	@app.route("/")
	def index():
		global frame2, flag
		if flag:
			print("[FLAG] =", flag)
			return Response(status=501)
		return pybase64.b64encode(cv2.imencode(".jpg", frame2)[1].tobytes())
	
	@app.route("/save/", methods=["POST"])
	def savephoto():
		global flag
		if flag:
			print("[FLAG] =", flag)
			return Response(status=501)
		name, data = request.form["name"], request.form["data"]
		print("FILE SIZE =", len(data))
		if not os.path.exists(os.getcwd() + "/dataset/" + name):
			os.mkdir(os.getcwd() + "/dataset/" + name)
		l = os.listdir(os.getcwd() + "/dataset/" + name)
		#print(l)
		num = -1
		for el in l:
			if "image_" in el:
				num = max(num, int(el[6:-4]))
		with open("./dataset/" + name + f"/image_{num + 1}.jpg", "wb") as f:
			#print(data)
			f.write(pybase64.b64decode(data))
		return Response(status=200)
	@app.route("/reload/", methods=["POST"])
	def rel():
		global flag
		
		if flag:
			print("[FLAG] =", flag)
			return Response(status=501)
		#write_config(update_ol=1, toff_ol=1)
		return Response(status=200)
	
	app.run(port=STEPA_KRUTOY_PORT, host=STEPA_KRUTOY_IP)
	
	
		
flag = False

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

th = Thread(target=fl_serv)

th.start()

camlist = [1]

namesprev = []


# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	# Detect the fce boxes
	boxes = face_recognition.face_locations(frame)
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			#If someone in your dataset is identified, print their name on the screen
			if currentname != name:
				currentname = name
				print("Found:", currentname)

		# update the list of names
		names.append(name)
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)
	if namesprev != names and names not in namesprev and not flag:
		names1 = names[:]
		if "Unknown" in names:
			names1.remove("Unknown")
			if len(names1):
				print("Posting to", f"http://{STEPA_SERVER_IP}:{STEPA_SERVER_PORT}/names/:", {"names": ";".join(names1), "photo": str(pybase64.b64encode(cv2.imencode('.jpg', frame2)[1].tobytes()))[2:-1][:20] + "..." + str(pybase64.b64encode(cv2.imencode('.jpg', frame2)[1].tobytes()))[2:-1][-20:], "date": time.ctime()}, "-", end="")
				r = requests.post(f"http://{STEPA_SERVER_IP}:{STEPA_SERVER_PORT}/names/", data={"names": ";".join(names1), "photo": str(pybase64.b64encode(cv2.imencode('.jpg', frame2)[1].tobytes()))[2:-1], "date": time.ctime()})
				print("", r)
		else:
			print("Posting to", f"http://{STEPA_SERVER_IP}:{STEPA_SERVER_PORT}/names/:", {"names": ";".join(names), "photo": str(pybase64.b64encode(cv2.imencode('.jpg', frame2)[1].tobytes()))[2:-1][:20] + "..." + str(pybase64.b64encode(cv2.imencode('.jpg', frame2)[1].tobytes()))[2:-1][-20:], "date": time.ctime()}, "-", end="")
			r = requests.post(f"http://{STEPA_SERVER_IP}:{STEPA_SERVER_PORT}/names/", data={"names": ";".join(names), "photo": str(pybase64.b64encode(cv2.imencode('.jpg', frame2)[1].tobytes()))[2:-1], "date": time.ctime()})
			print("", r)
		
	elif flag:
		print("[FLAG] =", flag)


	# display the image to our screen
	frame2 = frame
	if SHOW_IMAGE:
		cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF

	# quit when 'q' key is pressed
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
