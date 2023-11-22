import unittest
from twin4build.examples.minimal_example import minimal_example
from twin4build.examples.space_co2_no_controller_example import space_co2_no_controller_example
from twin4build.examples.space_co2_controller_example import space_co2_controller_example

class TestExamples(unittest.TestCase):
    @unittest.skipIf(False, 'Currently not used')
    def test():
        minimal_example()
        space_co2_no_controller_example()
        space_co2_controller_example()