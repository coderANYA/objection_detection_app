# pyqt5 app for showing the main window after loading the ui

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
import cv2
import numpy as np
import time
import numpy as np
import mediapipe as mp
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the UI file
main_screen = "ui/app_screen.ui"
about_screen = "ui/about_screen.ui"
obj_dect_image_screen = "ui/odis.ui"
obj_dect_video_screen = "ui/odvs.ui"
obj_dect_webcam_screen = "ui/odwc.ui"

# model path 
model1 = 'models/efficientdet.tflite'
model2 = 'models/efficientdet_lite2.tflite'
model3 = 'models/ssd_mobilenet_v2.tflite'

Ui_MainWindow, QtBaseClass = uic.loadUiType(main_screen)
Ui_AboutWindow, QtBaseClass = uic.loadUiType(about_screen)
Ui_OdisWindow, QtBaseClass = uic.loadUiType(obj_dect_image_screen)
Ui_OdvsWindow, QtBaseClass = uic.loadUiType(obj_dect_video_screen)
Ui_OdwcWindow, QtBaseClass = uic.loadUiType(obj_dect_webcam_screen)

# constants
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 0)  # red
ROOT_DIR = '.'

# visualization functions
def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image


# Create the main window
class MyApp(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.show()
        self.pushButton.clicked.connect(self.show_about)
        self.pushButton_2.clicked.connect(self.show_odis)
        self.pushButton_4.clicked.connect(self.show_odwc)
        # show mouse position on status bar
        self.statusBar().showMessage("Ready")

    def show_about(self):
        self.about_window = AboutWindow()
        self.about_window.show()   
        # hide the main window
        self.hide()

    def show_odis(self):
        self.odis_window = OdisWindow()
        self.odis_window.show()   
        # hide the main window
        self.hide()
    
    def show_odwc(self):
        self.odwc_window = OdwcWindow()
        self.odwc_window.show()   
        # hide the main window
        self.hide()
   
# Create the about window
class AboutWindow(QMainWindow, Ui_AboutWindow):
        
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_AboutWindow.__init__(self)
        self.setupUi(self)
        self.show()

    # on clicking of cross button 
    def closeEvent(self, event):
        self.main_window = MyApp()
        self.main_window.show()
        # hide the ab
        self.hide()

# Create the object detection image screen
class OdisWindow(QMainWindow, Ui_OdisWindow):
            
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_OdisWindow.__init__(self)
        self.setupUi(self)
        self.show()
        # 3 radio buttons click event
        self.radioButton.clicked.connect(self.radioButtonClicked)
        self.radioButton_2.clicked.connect(self.radioButtonClicked)
        self.radioButton_3.clicked.connect(self.radioButtonClicked)
        # default model
        self.model = model1
        self.image = None
        self.path = None
        # browse button click event
        self.toolButton.clicked.connect(self.browseImage)
        # detect button click event
        self.toolButton_2.clicked.connect(self.detectInImage)

    def browseImage(self):
        # open the file dialog
        path = self.openFileNameDialog()
        # load the image
        self.image = cv2.imread(path)
        self.path = path
        # show the image
        self.showImage(self.image)

    def openFileNameDialog(self):
        # show a file dialog to select image type file and the return full path of the image
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Image", "","Image Files (*.jpg *.png *.jpeg)", options=options)
        return fileName
    
    def showImage(self, image):
        try:
            # convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # convert the image to QImage
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            # scale image with a height of 600 and width of 600 pixels
            qimage = qimage.scaled(800, 600, Qt.KeepAspectRatio)
            # show the image
            self.label.setPixmap(QPixmap.fromImage(qimage))
            # show the status bar message
            self.statusBar().showMessage(f'Image loaded: {self.image.shape[1]}x{self.image.shape[0]}')
        except:
            pass

    def radioButtonClicked(self):
        if self.radioButton.isChecked():
            self.model = model1
        elif self.radioButton_2.isChecked():
            self.model = model2
        elif self.radioButton_3.isChecked():
            self.model = model3
        self.statusBar().showMessage(f'Model selected: {self.model}')

    def detectInImage(self):
        try:
            model = self.model
            base_options = python.BaseOptions(model_asset_path=model)
            options = vision.ObjectDetectorOptions(base_options=base_options,score_threshold=0.5)
            detector = vision.ObjectDetector.create_from_options(options)
            print("file",self.path)
            image = mp.Image.create_from_file(self.path)
            detection_result = detector.detect(image)
            image_copy = np.copy(image.numpy_view())
            annotated_image = visualize(image_copy, detection_result)
            rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            self.showImage(rgb_annotated_image)
            self.statusBar().showMessage(f'Objects detected: {len(detection_result.detections)}')

            # show names of detected objects in the textbrowser
            self.textBrowser.clear()
            for detection in detection_result.detections:
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                self.textBrowser.append(result_text)
            if len(detection_result.detections) == 0:
                self.textBrowser.append(f"No objects detected by {self.model}")
        except Exception as e:
            print(e)
            self.statusBar().showMessage(f'Error: {e}')
            self.textBrowser.clear()
            self.textBrowser.append(f'Error: {e}')

    # on clicking of cross button 
    def closeEvent(self, event):
        self.main_window = MyApp()
        self.main_window.show()
        # hide the ab
        self.hide()

# Create the object detection webcam screen
class OdwcWindow(QMainWindow, Ui_OdwcWindow):
            
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_OdwcWindow.__init__(self)
        self.setupUi(self)
        self.show()
        # 3 radio buttons click event
        # default model
        self.model = model1
        self.camera_id = 0
        self.is_camera_running = False
        # stop button click event
        self.radioButton.clicked.connect(self.radioButtonClicked)
        self.radioButton_2.clicked.connect(self.radioButtonClicked)
        self.radioButton_3.clicked.connect(self.radioButtonClicked)
        self.lineEdit.textChanged.connect(self.camera_id_changed)
        self.toolButton_2.clicked.connect(self.startDetection)
        self.toolButton_3.clicked.connect(self.stopDetection)
            
    def run(self, model: str, camera_id: int, width: int, height: int) -> None:
        """Continuously run inference on images acquired from the camera.

        Args:
            model: Name of the TFLite object detection model.
            camera_id: The camera id to be passed to OpenCV.
            width: The width of the frame captured from the camera.
            height: The height of the frame captured from the camera.
        """

        # Variables to calculate FPS
        counter, fps = 0, 0
        start_time = time.time()

        # Start capturing video input from the camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Visualization parameters
        row_size = 20  # pixels
        left_margin = 24  # pixels
        text_color = (0, 0, 255)  # red
        font_size = 1
        font_thickness = 1
        fps_avg_frame_count = 10

        detection_result_list = []
        def visualize_callback(result: vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms: int):
            result.timestamp_ms = timestamp_ms
            detection_result_list.append(result)
            # Initialize the object detection model
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.ObjectDetectorOptions(base_options=base_options,
                    running_mode=vision.RunningMode.LIVE_STREAM,
                    score_threshold=0.5,
                    result_callback=visualize_callback)
        detector = vision.ObjectDetector.create_from_options(options)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.'  )
            counter += 1
            image = cv2.flip(image, 1)

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Run object detection using the model.
            detector.detect_async(mp_image, counter)
            current_frame = mp_image.numpy_view()
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

            # Calculate the FPS
            if counter % fps_avg_frame_count == 0:
                end_time = time.time()
                fps = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()

            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(fps)
            text_location = (left_margin, row_size)
            cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)
            if detection_result_list:
                vis_image = visualize(current_frame, detection_result_list[0])
                # cv2.imshow('object_detector', vis_image)
                self.showImage(vis_image)
                self.statusBar().showMessage(f'Objects detected: {len(detection_result_list)}')
                self.textBrowser.clear()
                for detection in detection_result_list:
                    try:
                        category = detection.detections[0].categories[0]
                        category_name = category.category_name
                        probability = round(category.score, 2)
                        result_text = category_name + ' (' + str(probability) + ')'
                        self.textBrowser.append(result_text)
                    except Exception as e:
                        print(detection, e)
                detection_result_list.clear()
            else:
                self.showImage(current_frame)
                # cv2.imshow('object_detector', current_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                detector.close()
                cap.release()
                cv2.destroyAllWindows()
                self.statusBar().showMessage(f'Camera stopped')
                self.label.pixmap().fill(Qt.black)
                break
            if not self.is_camera_running:
                detector.close()
                cap.release()
                cv2.destroyAllWindows()
                self.statusBar().showMessage(f'Camera stopped')
                self.label.pixmap().fill(Qt.black)
                break
        
    def radioButtonClicked(self):
        if self.radioButton.isChecked():
            self.model = model1
        elif self.radioButton_2.isChecked():
            self.model = model2
        elif self.radioButton_3.isChecked():
            self.model = model3
        self.statusBar().showMessage(f'Model selected: {self.model}')

    def stopDetection(self):
        self.is_camera_running = False
        self.statusBar().showMessage(f'Camera stopped')

    def camera_id_changed(self):
        self.camera_id = int(self.lineEdit.text())
        self.statusBar().showMessage(f'Camera ID: {self.camera_id}')

    def startDetection(self):
        self.is_camera_running = True
        self.run(self.model, self.camera_id, 640, 480)
        self.statusBar().showMessage(f'Camera started')

    def showImage(self, image): 
        try:
            # convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # convert the image to QImage
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimage))
            # show the status bar message
        except Exception as e:
            print(e)
            self.statusBar().showMessage(f'Error: {e}')
             
    def closeEvent(self, event):
        self.main_window = MyApp()
        self.main_window.show()
        # hide the ab
        self.hide()

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())
