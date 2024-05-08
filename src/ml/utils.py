import torch
import torch.nn.functional as F

from src.data_preprocess.point import DEG_DIST

def accuracy(logits: torch.FloatTensor, target: torch.IntTensor):
    '''
    prediction accuracy
    '''
    probs = F.softmax(logits, dim=-1)
    predicted = probs.argmax(dim=-1)
    return (predicted == target).float().mean()

def haversine(p0: torch.FloatTensor, p1: torch.FloatTensor) -> torch.FloatTensor:
    '''
    batch compute euclidean distance between 2 points (lat, long)

    Args:
    ---
    - p0: [b, (lat, long)]
    - p1: [b, (lat, long)]

    Returns:
    ---
    - euclidean distance (meter)
    '''
    R = 6371; # volumetric radius of earth (km)

    p0 = torch.deg2rad(p0)
    p1 = torch.deg2rad(p1)

    delta = p1 - p0

    # https://en.wikipedia.org/wiki/Haversine_formula
    a = torch.sin(delta[:, 0] / 2) ** 2 + torch.cos(p0[:, 0]) * torch.cos(p1[:, 0]) * torch.sin(delta[:, 1] / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(a))
    d = R * c
    return d * 1000

def mean_deviation(logits: torch.FloatTensor, target: torch.FloatTensor, all_candidates: torch.FloatTensor):
    '''
    euclidean distance between predicted point and actual point

    Args:
    ---
    - logits
    - target: target classes
    - all_candidates: candidate locations

    Returns:
    ---
    - euclidean distance between predicted point and actual point
    '''
    logits = logits.view(-1, logits.shape[-1])
    probs = F.softmax(logits, dim=-1)

    target = target.view(-1)
    target = all_candidates[target, :]

    predicted = probs.argmax(dim=-1)
    predicted = all_candidates[predicted, :]

    return haversine(predicted, target).mean()

def to_cartesian(latlong: torch.FloatTensor, ref_point: tuple[float, float]):
    '''
    convert lat-long to cartesian by tangent plane projection
    '''
    cart = torch.zeros_like(latlong)

    dist_lat = DEG_DIST
    dist_long = DEG_DIST * torch.cos(torch.deg2rad(ref_point[0]))

    cart[:, 0] = dist_long * (latlong[:, 1] - ref_point[1])
    cart[:, 1] = dist_lat * (latlong[:, 0] - ref_point[0])

    return cart

def create_shanghai_preprocessor(x_range: tuple[float, float], y_range: tuple[float, float], ref_point: tuple[float, float]):
    min_y, max_y = y_range
    min_x, max_x = x_range
    
    def normalize_coords_inplace(tensor: torch.FloatTensor):
        '''
        normalize coordinates in-place
        '''
        tensor[:, 0] = 2 * (tensor[:, 0] - min_x) / (max_x - min_x) - 1
        tensor[:, 1] = 2 * (tensor[:, 1] - min_y) / (max_y - min_y) - 1
    
    def preprocess(tensor: torch.FloatTensor) -> torch.FloatTensor:
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, 2)

        # convert to cartesian
        tensor = to_cartesian(tensor, ref_point)

        # min-max normalization
        normalize_coords_inplace(tensor)

        return tensor.reshape(original_shape)

    return preprocess