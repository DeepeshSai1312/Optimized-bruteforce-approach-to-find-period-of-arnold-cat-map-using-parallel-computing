import cupy as cp
original_image = cv2.imread('lena.png')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

original_image = cp.asarray(original_image)
current_image = cp.copy(original_image)
iteration = 0
while True:
    current_image = arnold_cat_map(current_image)
    iteration += 1
    if cp.all(current_image == original_image):
        print(f'Number of iterations required to get original: {iteration}')
        break

original_image = cp.asnumpy(original_image)
current_image = cp.asnumpy(current_image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(current_image)
plt.title('After Nth iteration')
plt.axis('off')

plt.show()
