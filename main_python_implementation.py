from numpy import uint8, argmax
import cv2

# Initialize the parameters
confidenceThreshold = 0.7  #Confidence threshold
nmsThreshold 		= 0.4  #Non-maximum suppression threshold
inputWidth 			= 94  #Width of network's input image
inputHeight 		= 94  #Height of network's input image

datasetClasses_File = "coco.names"
classes 			= None

# Read and store classes' labels
with open(datasetClasses_File, 'rt') as readline:
	classes = readline.read().rstrip('\n').split('\n')

model_cfg_File 				= "yolov3.cfg"
model_weights_trained_File 	= "yolov3.weights"

trained_net = cv2.dnn.readNetFromDarknet(model_cfg_File, model_weights_trained_File)
trained_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
trained_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Webcam initialization
webcam_cap = cv2.VideoCapture(0)
 

# Get the names of the output layers
def getOutputsNames(arg_net):

    # Get the names of all the layers in the network
    layersNames = arg_net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in arg_net.getUnconnectedOutLayers()]



# Draw the predicted bounding box
def drawPrediction(classId, conf, left, top, right, bottom):

    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
     
    label = '%.2f' % conf

         
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
 
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))



# Remove the bounding boxes with low confidence using non-maxima suppression method
def ensureReliability(frame, outs):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
 
    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold)

    for i in indices:
    	i = i[0]
    	box = boxes[i]
    	eft = box[0]
    	top = box[1]
    	width = box[2]
    	height = box[3]
    	drawPrediction(classIds[i], confidences[i], left, top, left + width, top + height)



while 1:

	# get frame from the video
    _ , frame = webcam_cap.read()

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inputWidth, inputHeight), [0,0,0], 1, crop = True)
 
    # Sets the blob input to the network
    trained_net.setInput(blob)
 
    # Forward pass to get output of the output layers
    outputs = trained_net.forward(getOutputsNames(trained_net))

    # Remove low confidence bounding boxes
    ensureReliability(frame, outputs)

    # Put efficiency information. The function getPerfProfile returns the 
    # overall time for inference (time) and the timings for each of the layers (in layersTimes)
    time, _ = trained_net.getPerfProfile()

    info_label = 'Inference time: %.2f ms' % (time * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, info_label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv2.imshow('Real-Time Object Detection', frame.astype(uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam_cap.release()
cv2.destroyAllWindows()
 
