# imgTagger
A small and simple image tagger that saves your images into new folder, based on a classifier trained on the ImageNet. You specify a input path and the integrated classifier will sort your pictures in folder. The classifying is achived with pretrained CNNs, trained on the ImageNet-Dataset. This means there are 1000 possible classes. Your images will COPIED not moved! So you have a backup if the classifier goes mad ;)

When you run the application for the first time or trying another model it will need to download the weights of the CNN. Do not worry, as far as I know Google/Facebook did not have neural network weights that spy on you, yet.

## Install

You need Python3 for this software.

```bash
pip install -r requirements.txt
```

## Your CLI-Interface 

| Option                | Description       		      			                      |  		Default                	    |
| --------------------- |---------------------------------------------------------| --------------------------------|
| --out\_path           | Output path of the sorted image                     	  | images                          |
| --in\_path            | The path to the input image                     	      |   images\_sorted		              |
| --model               | Choose a model for classifying your images, currently supported: resnet18 (smaller resnet), mobilenet\_v2 (very small and reliable), resnet152 (big with good results), resnext101\_32x8d (best top 1 acc)                                 |   resnet152			    |

