import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from detector_neumonia import preprocess, read_dicom_file, predict

class TestDetectorNeumonia(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Esta función se llama una vez al inicio de todas las pruebas
        cls.dicom_path = os.path.join(os.path.dirname(__file__), '..', 'Imagenes de prueba', 'DICOM', 'viral (2).dcm')
        cls.model_path = os.path.join(os.path.dirname(__file__), '..', 'conv_MLP_84.h5')

    def setUp(self):
        # Esta función se llama antes de cada prueba.
        self.dummy_image = np.random.rand(512, 512, 3) * 255
        self.dummy_image = self.dummy_image.astype(np.uint8)

    def test_preprocess(self):
        preprocessed_image = preprocess(self.dummy_image)
        self.assertEqual(preprocessed_image.shape, (1, 512, 512, 1))
        self.assertTrue(np.max(preprocessed_image) <= 1.0)
        self.assertTrue(np.min(preprocessed_image) >= 0.0)

    def test_read_dicom_file(self):
        if os.path.exists(self.dicom_path):
            img_rgb, img2show = read_dicom_file(self.dicom_path)
            self.assertEqual(img_rgb.shape[-1], 3)
        else:
            self.skipTest(f"DICOM file not found at {self.dicom_path}")

    def test_predict(self):
        if os.path.exists(self.model_path):
            label, proba, heatmap = predict(self.dummy_image)
            self.assertIn(label, ["bacteriana", "normal", "viral"])
            self.assertTrue(0 <= proba <= 100)
            self.assertEqual(heatmap.shape[-1], 3)
        else:
            self.skipTest(f"Model file not found at {self.model_path}")

if __name__ == '__main__':
    unittest.main()
