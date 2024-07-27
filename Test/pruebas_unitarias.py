import unittest
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pydicom as dicom
from detector_neumonia import grad_cam, predict, read_dicom_file, read_jpg_file, preprocess  # Asegúrate de que la importación es correcta

class TestImageProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Cargar el modelo en el setUpClass si se utiliza en todas las pruebas
        cls.model = tf.keras.models.load_model('conv_MLP_84.h5')

    def setUp(self):
        # Prepara una imagen de prueba simple
        self.test_image = np.random.rand(512, 512, 3) * 255
        self.test_image = self.test_image.astype(np.uint8)
        
        # Prepara un archivo DICOM y JPEG ficticios para pruebas
        self.dicom_image = np.random.rand(512, 512) * 255
        self.dicom_image = self.dicom_image.astype(np.uint8)
        self.jpg_image = np.random.rand(512, 512, 3) * 255
        self.jpg_image = self.jpg_image.astype(np.uint8)

    def test_grad_cam(self):
        """ Test grad_cam function """
        result = grad_cam(self.test_image)
        self.assertEqual(result.shape, (512, 512, 3))  # Asegúrate de que el tamaño de la imagen es correcto
        self.assertIsInstance(result, np.ndarray)

    def test_predict(self):
        """ Test predict function """
        label, proba, heatmap = predict(self.test_image)
        self.assertIn(label, ["bacteriana", "normal", "viral"])
        self.assertTrue(0 <= proba <= 100)
        self.assertEqual(heatmap.shape, (512, 512, 3))  # Asegúrate de que el tamaño de la imagen es correcto
        self.assertIsInstance(heatmap, np.ndarray)

    def test_read_dicom_file(self):
        """ Test read_dicom_file function """
        # Guardar imagen de prueba DICOM en un archivo temporal
        dicom_file = 'test.dcm'
        dicom_file_temp = dicom.Dataset()
        dicom_file_temp.PixelData = self.dicom_image.tobytes()
        dicom_file_temp.file_meta = dicom.FileMetaDataset()
        dicom_file_temp.file_meta.MediaStorageSOPClassUID = dicom.uid.ImplicitVRLittleEndian
        dicom_file_temp.file_meta.MediaStorageSOPInstanceUID = dicom.uid.generate_uid()
        dicom_file_temp.is_little_endian = True
        dicom_file_temp.is_implicit_VR = True
        dicom_file_temp.Rows, dicom_file_temp.Columns = self.dicom_image.shape
        dicom_file_temp.save_as(dicom_file)
        
        img_RGB, img2show = read_dicom_file(dicom_file)
        self.assertEqual(img_RGB.shape[0], 512)
        self.assertEqual(img_RGB.shape[1], 512)
        self.assertIsInstance(img2show, Image.Image)

    def test_read_jpg_file(self):
        """ Test read_jpg_file function """
        # Guardar imagen de prueba JPG en un archivo temporal
        jpg_file = 'test.jpg'
        cv2.imwrite(jpg_file, self.jpg_image)
        
        img2, img2show = read_jpg_file(jpg_file)
        self.assertEqual(img2.shape[0], 512)
        self.assertEqual(img2.shape[1], 512)
        self.assertIsInstance(img2show, Image.Image)

if __name__ == '__main__':
    unittest.main()
s