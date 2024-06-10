import numpy as np
cimport numpy as np

cpdef np.ndarray[np.float64_t, ndim=2] box_iou(np.ndarray[np.float64_t, ndim=2] box1, np.ndarray[np.float64_t, ndim=2] box2):
    cdef int N = box1.shape[0]
    cdef int M = box2.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] ious = np.zeros((N, M), dtype=np.float64)
    
    cdef double x1, y1, w1, h1
    cdef double x2, y2, w2, h2
    cdef double inter_area, union_area
    
    for i in range(N):
        x1, y1, w1, h1 = box1[i]
        for j in range(M):
            x2, y2, w2, h2 = box2[j]
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)
            
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            
            ious[i, j] = inter_area / union_area if union_area > 0 else 0.0
    
    return ious