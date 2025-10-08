import os
import segmentation
import joblib

current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

classification_result = []
for each_character in segmentation.characters:
    each_character = each_character.reshape(1, -1)
    result = model.predict(each_character)
    classification_result.append(result)

print(classification_result)

plate_string = ''
for each_predict in classification_result:
    plate_string += each_predict[0]

print(plate_string)

column_list_copy = segmentation.column_list[:]
segmentation.column_list.sort()
rightplate_string = ''
for each in segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print(rightplate_string)