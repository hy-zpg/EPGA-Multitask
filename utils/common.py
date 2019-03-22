def load_image(paths: np.ndarray, size: int, input_size):
    images=[]
    for img_path in paths:
        image = cv2.imread('{}'.format(img_path))
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        # data augmentation
        image = image.img_to_array(image)
        image = image.random_rotation(image,rg=10)
        image = image.random_shift(image,wrg=0.1, hrg=0.1)
        image = image.random_zoom(image,zoom_range=0.1)
        image = flip_axis(image, axis=0)
        images.append(image)
    return np.array(images, dtype='uint8')