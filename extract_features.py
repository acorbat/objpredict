import pandas as pd

from skimage.measure import regionprops

import moment_invariants as inv


def crop_img(labeled_img, min_row, min_col, max_row, max_col, extra=10):
    """Returns a crop of the image with a little extra if needed."""
    this_crop = labeled_img.copy()
    min_row = min_row-extra
    if min_row<0:
        min_row = 0
    max_row = max_row+extra
    if max_row>this_crop.shape[0]:
        max_row = this_crop.shape[0]
    min_col = min_col-extra
    if min_col<0:
        min_col = 0
    max_col = max_col+extra
    if max_col>this_crop.shape[1]:
        max_col = this_crop.shape[1]
    
    this_crop = this_crop[min_row:max_row, min_col:max_col]
    return this_crop


def extract_features(labeled, int_img=None):
    """Returns a list with the extracted features from labeled and intensity
    image"""
    out = []
    for label, props in enumerate(regionprops(labeled)):
        this_crop = crop_img(labeled, *props.bbox)
        
        this_zer = inv.zernike_invariants(this_crop, deg=8)
        
        this_hu = inv.hu_invariants(this_crop)
        
        this_out = [label]
        this_out.extend(this_hu)
        this_out.extend(this_zer)
        if int_img is not None:
            this_int_img = this_crop(int_img, *props.bbox)
            this_out.append(this_int_img)
        out.append(this_out)
    return out


def analyze_images(labeled_series, int_series=None):
    """Generates a DataFrame with the features extracted from the list of
    labeled images and int_series if given.

    Parameters
    ----------
    labeled_series : list of np.arrays
        list of labeled images from which to extract features
    int_series : list of np.arrays
        list of intensity images from which to extract features

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the extracted features for each image
    """
    if int_series is None:
        int_series = [None] * len(labeled_series)
    assert labeled_series.shape == int_series.shape
    
    cols = ('n_img', 'label')
    for i in range(1, 8+1):
        cols = cols + ('hu'+str(i),)
    for i in range(1, 41+1):
        cols = cols + ('S'+str(i),)
    if int_series is not None:
        cols = cols + ('int_img')
    
    out = {}
    for col in cols:
        out[col] = []
    
    for n_img, labeled_img, int_img in enumerate(zip(labeled_series, int_series)):
        cells_feats = extract_features(labeled_img, int_img)
        out['n_img'].append(n_img)
        for cell_feats in cells_feats:
            for col, feature in zip(cols[1:], cell_feats):
                out[col].append(feature)
    
    df = pd.DataFrame(out)
    return df
