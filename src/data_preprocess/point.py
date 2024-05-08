import numpy as np
from typing import Union
from sklearn.neighbors import NearestNeighbors

LatLong = tuple[float, float]
Vec2 = tuple[float, float]

#region discretizer
class Discretizer:
    '''
    abstract interface for discretizing points
    '''
    
    def discretize(self, points: np.ndarray) -> np.ndarray:
        '''
        discretize points
        '''
        return points
    
    def list_points(self) -> Union[np.ndarray, None]:
        '''
        list points in discretizer.

        Returns:
        ---
        - points if any
        '''
        return None
    
class GridDiscretizer(Discretizer):
    '''
    discretize by mapping to center of grid containing the point
    '''

    def __init__(self, xrange: tuple[float, float], yrange: tuple[float, float], dim: tuple[int, int]):
        self.xrange = xrange
        self.yrange = yrange
        self.dim = dim

        self.window_width = (
            (xrange[1] - xrange[0]) / dim[0],
            (yrange[1] - yrange[0]) / dim[1]
        )

        self.intercept = (
            xrange[0] + self.window_width[0] / 2,
            yrange[0] + self.window_width[1] / 2
        )

        self.slope = (
            (xrange[1] - xrange[0] - self.window_width[0]) / (dim[0] - 1),
            (yrange[1] - yrange[0] - self.window_width[1]) / (dim[1] - 1)
        )

    def discretize(self, points: np.ndarray) -> np.ndarray:
        assert np.all((self.xrange[0] <= points[:, 0]) & (points[:, 0] <= self.xrange[1]))
        assert np.all((self.yrange[0] <= points[:, 1]) & (points[:, 1] <= self.yrange[1]))

        out = np.zeros_like(points, dtype=float)

        out[:, 0] = self.intercept[0] + self.slope[0] * ((points[:, 0] - self.xrange[0]) // self.window_width[0]).clip(0, self.dim[0] - 1)
        out[:, 1] = self.intercept[1] + self.slope[1] * ((points[:, 1] - self.yrange[0]) // self.window_width[1]).clip(0, self.dim[1] - 1)
        
        return out
    
    def list_points(self):
        x = self.intercept[0] + self.slope[0] * np.arange(0, self.dim[0])
        y = self.intercept[1] + self.slope[1] * np.arange(0, self.dim[1])
        
        x, y = np.meshgrid(x, y)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        return np.concatenate([x, y], axis=1)

class NearestNeighborDiscretizer(Discretizer):
    '''
    discretize by mapping to nearest support
    '''

    def __init__(self, points: np.ndarray):
        self.points = points
        self.nn = NearestNeighbors(n_neighbors=1).fit(points)

    def discretize(self, points: np.ndarray) -> np.ndarray:
        _, indices = self.nn.kneighbors(points)
        return self.points[indices[:, 0], :]
    
    def list_points(self) -> np.ndarray:
        return self.points
#endregion

# average distance between 1 deg difference in latitude
DEG_DIST = 111319.44

class CoordinateMap:
    '''
    project lat-long to a rectangular plane tangent to curved surface centered at the reference point.

    stolen from Erik:
    https://github.com/erik-buchholz/RAoPT/blob/main/raopt/preprocessing/coordinates.py
    '''
    
    def __init__(self, ref: LatLong):
        '''
        Args:
        ---
        - ref: reference point
        '''
        self.ref = ref

    def to_cartesian(self, latlong: np.ndarray) -> np.ndarray:
        '''
        convert from lat-long to cartesian coordinates relative to a reference point

        Args:
        ---
        - latlong: [b, (lat, long)]

        Returns:
        ---
        - coords: [b, (x meter, y meter)]
        '''
        assert len(latlong.shape) == 2
        assert latlong.shape[1] == 2

        cart = np.zeros_like(latlong)

        lat0 = self.ref[0]
        long0 = self.ref[1]

        dist_lat = DEG_DIST
        dist_long = DEG_DIST * np.cos(np.radians(lat0))

        cart[:, 0] = dist_long * (latlong[:, 1] - long0)
        cart[:, 1] = dist_lat * (latlong[:, 0] - lat0)

        return cart
    
    def to_latlong(self, cart: np.ndarray) -> np.ndarray:
        '''
        convert from relative cartesian coordinate to lat-long

        Args:
        ---
        - cart: [b, (x meter, y meter)]
        - ref: reference point in (lat, long)

        Returns:
        ---
        - latlong: [b, (lat, long)]
        '''
        assert len(cart.shape) == 2
        assert cart.shape[1] == 2

        latlong = np.zeros_like(cart)

        lat0 = self.ref[0]
        long0 = self.ref[1]

        dist_lat = DEG_DIST
        dist_long = DEG_DIST * np.cos(np.radians(lat0))

        latlong[:, 1] = cart[:, 0] / dist_long + long0
        latlong[:, 0] = cart[:, 1] / dist_lat + lat0

        return latlong