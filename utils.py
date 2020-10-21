import json
import numpy as np
import pandas as pd

import os
from PIL import Image


question_path = 'dataset/dataset_v7w_telling.json'
image_path = 'dataset/v7w_telling_answers.json'

def bound_box_position_handler(file_path):
    '''

    :param file_path:
            file: v7w_telling_answers.json
            content: position of bounding boxes for objects, query by qa_id
    :return: dictionary: {qa_id:[object_names, object_numbers, bounding_box_positions]}
            different_object: all object names are different
            duplicated_object: exists same object names
            same_object: all object names are the same
    '''
    with open(file_path,'r',encoding='utf-8') as f:
        data = json.load(f)

        object_freq = {}
        for object in data['boxes']:

            if object['qa_id'] not in object_freq:
                position = []
                position.append([object['height'],object['width'],object['y'],object['x']])
                object_freq[object['qa_id']] = [[object['name']],1,position]
            else:
                l = object_freq[object['qa_id']][0]
                l.append(str(object['name']))
                position = list(object_freq[object['qa_id']][2])
                position.append([object['height'],object['width'],object['y'],object['x']])
                object_freq[object['qa_id']] = [l, object_freq[object['qa_id']][1]+1,position]


    different_object = {}
    duplicated_object = {}
    same_object = {}
    for object in object_freq:

        dic = {}.fromkeys(object_freq[object][0])
        if object_freq[object][1] >1:
            if len(dic) ==len(object_freq[object][0]): # all objects are different
                different_object[object] = object_freq[object]
            elif len(dic) > 1: # exists same objects
                duplicated_object[object] = object_freq[object]
            else: # all objects are exactly the same
                same_object[object] = object_freq[object]


    return different_object,duplicated_object,same_object

def qa_text_handler(file_path,objects):
    '''

    :param file_path:
            file: dataset_v7w_telling.json
            content: questions & answers, query by qa_id
    :param objects: filtered qa pairs return by bound_box_position_handler()
    :return: list: [image_id,qa_id,question,answer, question_type]
            filtered by question type(in this case 'who')
    '''


    with open(file_path,'r',encoding='utf-8') as f:
        data = json.load(f)


    qa_pair_who_list = []
    for qa_pairs in data['images']:

        for qa_pair in qa_pairs['qa_pairs']:
            if qa_pair['qa_id'] in objects.keys():
                if qa_pair['type'] =='who':
                    qa_pair_who_list.append([qa_pair['image_id'],qa_pair['qa_id'],qa_pair['question'],
                                        qa_pair['answer'],qa_pair['type']])



    return qa_pair_who_list


def size_handler(path):

    with Image.open(path) as img:

        return img.size #(width,height)


def dataset_generation(qa_list,objects):
    '''
        write filtered training samples into csv file
    :param qa_list:
                return by qa_text_handler()
    :param objects:
                return by bound_box_position_handler()
    '''

    data = []
    for sample in qa_list:
        image_id = sample[0]
        qa_id = sample[1]
        question = sample[2]
        answer = sample[3]

        object1 = objects[qa_id][0][0]
        object2 = objects[qa_id][0][-1]
        positon = objects[qa_id][2]

        question = question.lower().lstrip('who ').rstrip('?')
        answer = answer.lower().strip('.')
        qa = answer +' ' + question

        path = os.path.join('dataset/visual_7w/images', 'v7w_'+str(image_id)+'.jpg')
        size = size_handler(path) #(width,height)


        data.append({'ImageID':image_id,'LabelName1':object1,'LabelName2':object2,'Height':size[1],'Width':size[0],
        'XMin1':positon[0][3], 'XMax1': positon[0][3]+positon[0][1], 'YMin1':positon[0][2], 'YMax1':positon[0][2]+positon[0][0],
        'XMin2':positon[-1][3], 'XMax2':positon[-1][3]+positon[-1][1],'YMin2': positon[-1][2],'YMax2':positon[-1][2]+positon[-1][0],
        'ReferringExpression':qa})



    df2 = pd.DataFrame(data, columns=['ImageID', 'LabelName1', 'LabelName2', 'Height','Width','XMin1','XMax1','YMin1','YMax1',
                                      'XMin2','XMax2','YMin2','YMax2','ReferringExpression'])

    df2.to_csv('output/sample23245.csv',mode='a', index=False)

if __name__ == '__main__':


    for i,objects in enumerate(bound_box_position_handler(image_path)):
        qa = qa_text_handler(question_path, objects)
        if i!=2:
            dataset_generation(qa, objects)
        else:
            dataset_generation(qa[:1300], objects)
