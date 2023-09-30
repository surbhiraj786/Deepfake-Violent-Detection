import numpy as np
from PIL import Image,ImageFile,ImageOps, ImageEnhance
import os
import torch
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import cv2
from torch import nn
import torch.optim as optim
from torch import tensor
from model import Model
import imgaug
from sklearn.metrics import classification_report, average_precision_score
import copy
from torch.autograd import Variable
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
from PIL import Image
import torchvision.transforms.functional as F

torch.set_num_threads(1)

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

def load_image_with_error_handling(image_path, transform=None):
    try:
        image = Image.open(image_path)
        if transform:
            image = transform(image)
        return image
    except (Image.UnidentifiedImageError, OSError):
        print(f"Skipping {image_path} due to UnidentifiedImageError or OSError.")
        return None


def compute_prnu(image, n=3):
    # Compute the noise residual pattern of the image
    noise = np.zeros_like(image, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            noise += (image - cv2.GaussianBlur(image, (n,n), n/5))**2
    noise = np.sqrt(noise / (n*n))

    # Compute the PRNU by normalizing the noise residual pattern
    prnu = noise / np.mean(noise)
    return prnu

	
def convertImageToTensor(imagePath):
  #image = Image.open(imagePath)
  #print(imagePath)
  imageTensor = readImage(imagePath).float()
  imageTensor = Variable(imageTensor,requires_grad=False)
  imageTensor = imageTensor.unsqueeze(0)
  return imageTensor
  

def readImage(imagePath):
  # Load image
  #img = cv2.imread(imagePath, 0)
  img = load_image_with_error_handling(imagePath,transform=None)
  #print(img)
  img_filtered = None
  prnu = None
  result = None
  
  if img is None:
      os.remove(imagePath)
      
  if img is not None:      
      image = img.resize((256,256)) 
      
      # Apply the hue transformation to the image      
      img = transform(image)
      img = np.array(img)
  
      #img = cv2.resize(img, (112, 112))
      # Apply 2D Fourier Transform
      f = np.fft.fft2(img)
  
      # Shift zero frequency to the center of the spectrum      
      dm_frequency_domain = np.fft.fftshift(f)
      dm_reduced_domain = dm_frequency_domain.copy()
      # Set a threshold for magnitude to retain only the most significant coefficients
      threshold = 0.001 * np.max(np.abs(dm_reduced_domain))
      dm_reduced_domain[np.abs(dm_reduced_domain) < threshold] = 0
      transformed_image = np.log(1 + np.abs(dm_reduced_domain))
      img_filtered = torch.tensor(transformed_image)
      image2 = np.array(image.convert("YCbCr"))[:,:,0]
      prnu = compute_prnu(image2)
      prnu = torch.tensor(prnu)
      result = torch.stack([img_filtered,prnu],0)
  return result

def testImage(nnModel,imageTensor):
  nnModel.eval()
  with torch.no_grad():
    imageTensor = imageTensor.to(device) 
    #print(imageTensor.is_cuda)
    dfe = nnModel(imageTensor)
    dfe = dfe.cpu()
  return torch.argmax(dfe, 1)
  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("XXXXX")
#load trained model
trainedModel = torch.load('/home/surbhi/docker_dfdv/fft_2Patcehes_main_model_ViolenceDS_15K_images_100epoch.pth')
trainedModel = trainedModel.to(device)
for param in trainedModel.parameters():
    param.requires_grad = False
    
    

#load images
fake_images_path = "/home/surbhi/docker_dfdv/test set/fake/"
fakeImagesList = os.listdir(fake_images_path)
fakeTrueLabel = [0]*len(fakeImagesList)
real_images_path = "/home/surbhi/docker_dfdv/test set/real/"
realImagesList = os.listdir(real_images_path)
realTrueLabel = [1]*len(realImagesList)


pred_labels=[]
true_labels=[]
image_names=[]
for image in fakeImagesList:  
    imageTensor = convertImageToTensor(fake_images_path+"/"+image)
    pred = testImage(trainedModel,imageTensor)[0]
    true_labels.append(0)
    pred_labels.append(pred)
    image_names.append(image)

for image in realImagesList:  
    imageTensor = convertImageToTensor(real_images_path+"/"+image)
    pred = testImage(trainedModel,imageTensor)[0]
    true_labels.append(1)
    pred_labels.append(pred)
    image_names.append(image)
	
labels = [image_names,true_labels,pred_labels] 
with open("/home/surbhi/docker_dfdv/text.txt", "w") as file:
     file.write("image_name\ttrue_label\tpred_label")
     for x in zip(*labels):
        file.write("{0}\t{1}\t{2}\n".format(*x))

report = classification_report(true_labels, pred_labels)
print(report)
AP = average_precision_score(true_labels, pred_labels)
print(AP) 









