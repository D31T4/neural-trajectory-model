import matplotlib.pyplot as plt
import folium

from src.data_preprocess.trajectory import Trajectory

def get_shanghai_map() -> folium.Map:
    '''
    get shanghai map
    '''
    return folium.Map(location=[31.11, 121.49], tiles="CartoDB Positron", zoom_start=8.5)

def plot_trajectory(trajectory: Trajectory, color: str, map: folium.Map = None, marker: bool = True) -> folium.Map:
    '''
    plot trajectory on map

    Args:
    ---
    - trajectory
    - color
    - map
    '''
    map = map or get_shanghai_map()

    if marker:
        for p in set(p for p in trajectory.points if p is not None):
            folium.CircleMarker(p, radius=2, weight=5, color=color).add_to(map)

    prev: int = 0

    for i in range(1, len(trajectory)):
        if trajectory[i] == None:
            continue
        
        if trajectory[prev] != None:
            if prev == i - 1:
                folium.PolyLine(locations=[trajectory[prev], trajectory[i]], weight=2, color=color).add_to(map)
            else:
                folium.PolyLine(locations=[trajectory[prev], trajectory[i]], weight=2, color=color, dash_array=[10]).add_to(map)

        prev = i

    return map
