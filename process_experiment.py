# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:50:33 2017

@author: Agus
"""
import os
import pathlib
import appdirs
import shutil

import pandas as pd
import skimage.io as skio

import extract_features as exf
import classify as clf

data_dir = pathlib.Path(appdirs.user_data_dir('objpredict', 'LEC'))
data_dir.mkdir(parents=True, exist_ok=True)

#%% Load 
def get_installed_pipelines():
    pipelines = {'./%s' % p.stem : p for p in pathlib.Path('.').glob('*.pipe')}
    pipelines.update({'%s' % p.stem : p for p in pathlib.Path(data_dir).glob('*.pipe')})
    return pipelines


def install(path, name=None):
    src = pathlib.Path(path)
    if name is None:
        name = src.name
    src = src.with_suffix('.pipe')
    dst = data_dir.joinpath(name + '.pipe')
    shutil.copy(str(src), str(dst))


def list_attrs():
    cols = []
    for i in range(1, 8+1):
        cols = cols + ['hu'+str(i),]
    for i in range(1, 41+1):
        cols = cols + ['S'+str(i),]
    return cols


def gen_filemap(filename):
    image_paths = pd.read_table(filename)
    
    img_types = [col.split('_', 1)[1] for col in image_paths.columns if col.startswith('FileName_')] 
    obj_types = [col.split('_', 1)[1] for col in image_paths.columns if col.startswith('ObjectsFileName_')] 
    

    output = []
    
    for i in image_paths.index:
        imgs = [os.path.join(image_paths['PathName_' + col][i], image_paths['FileName_' + col][i])
                for col in img_types]
        objs = [os.path.join(image_paths['ObjectsPathName_' + col][i], image_paths['ObjectsFileName_' + col][i])
                for col in obj_types]

        output.append((image_paths.ImageNumber[i], imgs, objs))
    
    # TODO: check if only one img type is generated
    return output, img_types, obj_types


def process_experiment(filenames, do_append=False, pipeline='PCA_ExtraTrees'):
    for filename in filenames:
        images_map, img_types, obj_types = gen_filemap(filename) # Receives the *_Image.txt file and paths should be parsed
        
        if do_append:
            obj_csv_dir = '_'.join([filename.split('_')[0], obj_types[0]]) + '.txt'
            pre_csv = pd.read_csv(obj_csv_dir, sep='\t')
            df_csv = pd.DataFrame()
        
        df = pd.DataFrame()
        for img_num, img_path, labeled_path  in images_map:
            labeled = skio.imread(labeled_path)
            
            this_df = exf.analyze_images(labeled)
            
            this_df = clf.classify(this_df, pipeline)
            
            this_df['ImageNumber'] = img_num
            if do_append:
                this_csv = pd.merge(pre_csv, this_df, on=['ImageNumber', 'ObjectNumber'], how='outer')
                df_csv.append(this_csv)
            else:
                df = df.append(this_df)
        
        if do_append:
            df_csv.to_csv(obj_csv_dir)
        else:
            df.to_csv('Classification.csv')


def train(folders, pipeline='PCA_ExtraTrees'):
    cols = ['c']
    cols.extend(list_attrs())
    for folder in folders:
        classes = {}
        folder = pathlib.Path(folder)
        classes = {p.name: p for p in folder.iterdir() if p.is_dir()}
        
        df = pd.DataFrame()
        for this_class, this_folder in classes.items():
            for file in this_folder.glob('*.tif'):
                img = skio.imread(file) # TODO: add try for corrupted image
                attrs = exf.extract_features(img)[0] # I only have one label in this image
                
                attrs[0] = this_class # Replace img label (always 0) with class
                
                this_df = pd.DataFrame([attrs], columns=cols)
                df = df.append(this_df, ignore_index=True)
        
        pipedict = clf.train_classifier(df) # TODO: if more than one folder, it overwrites
        savepath = folder.joinpath(pipeline+'.pipe')
        clf.save_pipeline(savepath, pipedict)