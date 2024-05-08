import unittest

import numpy as np
from src.data_preprocess.point import GridDiscretizer, CoordinateMap

class GridDiscretizerTest(unittest.TestCase):
    '''
    unit test of `GridDiscretizer`
    '''

    def test_discretize(self):
        '''
        toy example
        '''
        d = GridDiscretizer(xrange=(0, 1), yrange=(0, 1), dim=(2, 2))
        out = d.discretize(np.array([[.9, 1], [0, .3]]))

        self.assertTrue(np.allclose(out, np.array([[.75, .75], [.25, .25]])))

    def test_list_points(self):
        '''
        test list_points
        '''
        d = GridDiscretizer(xrange=(0, 1), yrange=(0, 1), dim=(2, 2))
        out = d.list_points()

        self.assertTrue(np.allclose(
            out,
            np.array([
                [.25, .25],
                [.75, .25],
                [.25, .75],
                [.75, .75]
            ])
        ))

    def test_geolife(self):
        '''
        sample from geolife
        '''
        d = GridDiscretizer(xrange=(39.442078, 41.058964), yrange=(115.416827, 117.508251), dim=(8, 8))
        out = d.discretize(np.array([[40.957909, 117.377537]]))

        self.assertTrue(np.allclose(out, np.array([[ 40.95790863, 117.377537 ]])))

class CoordinateMapTest(unittest.TestCase):
    '''
    unit test of `CordinateMap`
    '''

    def to_cartesian_test(self):
        '''
        example from T-drive
        '''
        coord_map = CoordinateMap(ref=(39.915, 116.395))

        out = coord_map.to_cartesian(
            np.array([
                (39.92123, 116.51172),
                (39.93883, 116.51135)
            ])
        )

        self.assertTrue(
            np.allclose(out, [
                [9965.75180957,  693.5201112 ],
                [9934.16058125, 2652.7422552 ]
            ])
        )

    def to_latlong_test(self):
        '''
        example from T-drive
        '''
        coord_map = CoordinateMap(ref=(39.915, 116.395))

        out = coord_map.to_latlong(
            np.array([
                [9965.75180957,  693.5201112 ],
                [9934.16058125, 2652.7422552 ]
            ])
        )

        self.assertTrue(
            np.allclose(out, [
                (39.92123, 116.51172),
                (39.93883, 116.51135)
            ])
        )

if __name__ == '__main__':
    unittest.main()