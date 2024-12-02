from PIL import Image
from torchvision import models


vgg_weights = models.VGG19_Weights.DEFAULT

categories = vgg_weights.meta['categories']
tramsform = vgg_weights.transforms()

img = Image.open('488.jpg').convert('RGB')
img_net = tramsform(img).unsqueeze(0)

model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
model.eval()
p = model(img_net).squeeze()
res = p.softmax(dim=0).sort(descending=True)

for s, i in zip(res[0][:5], res[1][:5]):
    print(f'{categories[i]}: {s:.4f}')
