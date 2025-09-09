import numpy as np

def _strokeLength(strokes):
    Metrics = []
    max_x=-1
    max_y=-1
    min_x=1e10
    min_y=1e10
    for ii in range(len(strokes)):
        s = np.array(strokes[ii])
        max_x = max(max_x, s[:, 0].max())
        max_y = max(max_y, s[:, 1].max())
        min_x = min(min_x, s[:, 0].min())
        min_y = min(min_y, s[:, 1].min())
        ds = np.diff(s, axis=0)
        length = ((ds**2).sum(axis=1)**0.5).sum()
        Metrics.append(length)
    Metrics.sort()
    pos = len(Metrics) // 4
    return Metrics[pos], min_x, min_y, max_x, max_y

def _rescale(strokes, scale, minX, minY):
    new_strokes = []
    for ii in range(len(strokes)):
        s = np.array(strokes[ii])
        s[:, 0] = (s[:, 0] - minX) * scale
        s[:, 1] = (s[:, 1] - minY) * scale
        new_strokes.append(s.tolist())
    return new_strokes

def rescale_strokes(strokes, stroke_length=25):
    # get the 25% percentile of stroke lengths
    stroke_len, minX, minY, maxX, maxY = _strokeLength(strokes)
    scale = stroke_length / stroke_len

    # rescale based on stroke length to get a suggested image size
    strokes_norm = _rescale(strokes, scale, minX, minY)

    stroke_len_norm, minX, minY, maxX, maxY = _strokeLength(strokes_norm)
    return strokes_norm, [maxX, maxY]
