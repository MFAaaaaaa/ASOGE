import numpy as np

img = np.random.randint(1, 10, size=[5, 4, 3])
img2 = np.random.randint(1,10,size=[5,4])
# print(img.shape[0])
print(img)
print(img2)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img2[i][j] = round(np.mean(img[i][j]),2)

print(img2)