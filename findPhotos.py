from config import DROPBOX_API


def execute():
    import os
    import tensorflow.keras
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
    from tensorflow.keras.models import Model
    import time

    model = tensorflow.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
    import sklearn.metrics
    import tensorflow
    import numpy as np
    def load_image(path):
        img = tensorflow.keras.utils.load_img(path, target_size=model.input_shape[1:3])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x

    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    feat_extractor.summary()

    import random
    from matplotlib import pyplot as plt
    from scipy.spatial import distance

    def get_closest_images(query_image_idx, num_results=5):
        distances = [distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features]
        print(distances)
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results + 1]
        print(idx_closest)
        return idx_closest

    def get_concatenated_images(idx_closest_paths, thumb_height):
        thumbs = []
        for idx in idx_closest_paths:
            img = image.load_img(idx)
            img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
            thumbs.append(img)
        concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
        return concat_image

    import sklearn
    from sklearn.decomposition import PCA
    import pickle as pk

    with open('content/features_images_short.p', 'rb') as pickle_file:
        images, pca_features, pca = pk.load(pickle_file)
        # do a query on a random image
        # query_image_idx = int(len(images) * random.random())

        imagetime = []
        for i in os.listdir("static/files"):
            imagetime.append(os.path.getmtime("static/files/"+str(i)))
        imagetime.sort()

        for i in range(len(imagetime)):
            if imagetime[-1] == os.path.getmtime("static/files/"+str(os.listdir("static/files")[i])):
                userimage = "static/files/"+str(os.listdir("static/files")[i])
            else: continue


        new_image, x = load_image(userimage)

        new_features = feat_extractor.predict(x)
        new_pca_features = pca.transform(new_features)[0]
        distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:20]  # grab 20 occurrences

        '''FOR DROPBOX'''
        idx_closest_paths = [images[idx_closest[i]] for i in range(len(idx_closest))]
        #content/all_images_short/ISIC_0028968.jpg --> /all_images_short/ISIC_0028968.jpg
        idx_closest_paths = [idx_closest_path[7:] for idx_closest_path in idx_closest_paths]

        import requests
        import json
        import base64
        from PIL import Image
        from io import BytesIO

        def convert_base64(image_base, name):
            # Convert Base64 to image
            image = Image.open(BytesIO(base64.b64decode(image_base)))
            # Save image
            image.save("all_images_short/" + name)

        url = "https://content.dropboxapi.com/2/files/get_thumbnail_batch"
        headers = {
            "Authorization": DROPBOX_API,
            "Content-Type": "application/json",
            "Dropbox-Api-Select-User": ""
        }
        data = {"entries": []}
        for i in idx_closest_paths:
            data["entries"].append({"path": i, "format": {".tag": "jpeg"}, "size": {".tag": "w480h320"}})
        r = requests.post(url, headers=headers, data=json.dumps(data))
        info = r.json()
        for i in range(len(info["entries"])):
            convert_base64(info["entries"][i]["thumbnail"], info["entries"][i]["metadata"]["name"])
        '''FOR DROPBOX'''
        
        idx_closest_paths = [idx_closest_path[1:] for idx_closest_path in idx_closest_paths]
        results_image = get_concatenated_images(idx_closest_paths[0:5], 200) # grab 5 photos

        '''DELETE DROPBOX FILES'''
        import os
        folder_path = "all_images_short"
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(e)
        '''DELETE DROPBOX FILES'''

        import pandas as pd
        df = pd.read_csv('content/HAM10000_metadata_short')
        lesion = []

        lesion_type_dict = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }

        for i in idx_closest:
            x = df['dx'].loc[df.index[i]]
            lesion.append(lesion_type_dict[x])
        print(lesion)

        lesion_percentage = []
        lesions_stat = {}
        for i in range(len(lesion)):
            try:
                lesions_stat[lesion[i]] += 1
            except:
                lesions_stat[lesion[i]] = 1
        for i in lesions_stat.keys():
            stat = int(lesions_stat[i]) / len(lesion)
            lesion_percentage.append(str(i) + "--"+str(format((stat * 100), ".4g")) + "%")

        from PIL import Image
        import base64
        from io import BytesIO


        def encode_img(img):
            pil_img = Image.fromarray(img)
            buff = BytesIO()
            pil_img.save(buff, format="JPEG")
            results_image = base64.b64encode(buff.getvalue()).decode("utf-8")
            return results_image


        for i in idx_closest:
            x = df['dx'].loc[df.index[i]]
            lesion.append(lesion_type_dict[x])

        def encode_img1(image, im_type="JPEG"):
            buffered = BytesIO()
            image.save(buffered, format=im_type)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str


    new_image = encode_img1(new_image)
    results_image = encode_img(results_image)


    main_result = [lesion[0:5], new_image,results_image, lesion_percentage]
    return main_result
