import unittest as ut
from main import sigmoida
import numpy as np

class Sigmoida(ut.TestCase):
	def test_numbers_8(self):
		self.assertEqual(sigmoida(0), 0.5)
	# def test_vectorself(self):		
	# 	self.assertEqual(sigmoida(np.array([2,5,6]), [ 0.88079708,  0.99330715,  0.99752738])

	
if __name__ == '__main__':
	ut.main()