# A Port of SSD: Single Shot MultiBox Detector to Google Cloud Machine Learning Engine.

For more details, please refer to [arXiv paper](http://arxiv.org/abs/1512.02325) and [Keras implementation](https://github.com/rykov8/ssd_keras).

## Usage

```sh
$ pip install git+https://github.com/monochromegane/starchart.git
$ git clone https://github.com/monochromegane/ssd_mlengine.git
$ starchart train -m ssd_mlengine \
                  -M trainer.task \
                  -s BASIC_GPU \
                  -- \
                  --annotation_path=gs://PATH_TO_ANNOTATION_FILE \
                  --prior_path=gs://PATH_TO_PRIOR_FILE \
                  --weight_path=gs://PATH_TO_WEIGHT_FILE \
                  --images_path=gs://PATH_TO_IMAGES_FILE \
                  --model_dir=TRAIN_PATH/model
```

When the training is over,

```sh
$ starchart expose -m ssd_mlengine
```

When the prediction API is published,

```sh
$ python predict.py
# {'predictions': [{'key': '1',
#    'objects': [[8.0,        # class
#      0.45196059346199036,   # confidence
#      104.90727233886719,    # xmin
#      97.99836730957031,     # ymin
#      212.12222290039062,    # xmax
#      315.3045349121094]]}]} # ymax
```

The code of `prediction.py` is like the following:

```python
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
from PIL import Image

project = YOUR_PROJECT_ID
model   = 'ssd_mlengine'
version = YOUR_MODEL_VERSION

credentials = GoogleCredentials.get_application_default()
ml = discovery.build('ml', 'v1', credentials=credentials)

size = (300, 300)
image_path = TARGET_IMAGE_PATH
img = Image.open(image_path)
original_size = img.size
resized_img = img.resize(size)

data = []
for i in range(size[1]):
    data.append([])
    for j in range(size[0]):
        data[i].append(resized_img.getpixel((j, i)))

body = {'instances': [{'key': '1',
    'data': data,
    'keep_top_k': 1,
    'original_size': original_size,
    'confidence_threshold': 0.45
    }]}
request = ml.projects().predict(name='projects/{}/models/{}/versions/{}'.format(project, model, version), body=body)
try:
    response = request.execute()
    output = response['predictions']
    print(output)
except errors.HttpError as err:
    print(err._get_reason())
```

## Preparation

- A account for Google Cloud Machine Learning Engine
- Install [monochromegane/starchart](https://github.com/monochromegane/starchart)
- Put these files to Google Cloud Storage.
    - Ground truth file. You can generate by the [code](https://github.com/rykov8/ssd_keras/blob/master/PASCAL_VOC/get_data_from_XML.py)
    - Images file. The source images of ground truth (tar.gz).
    - `weights_SSD300.hdf5`. Pre-trained weight file. Download from [here](https://github.com/rykov8/ssd_keras)
    - `prior_boxes_ssd300.pkl`. Pre-calcuration default boxes. Download from [here](https://github.com/rykov8/ssd_keras). And re-packaging for this repository.
      ```py
      import pickle
      pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))[4332:].dump('prior_boxes_ssd300_min.pkl')
      ```
