import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


# real speckle process
def process_image(image_path, d=15, kernel_size=(7, 7), sigma=2, crop=3600):
    # read speckle
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # calculate low-pass version
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # 中心位置

    # create mask
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - d:crow + d, ccol - d:ccol + d] = 1

    fshift_filtered = fshift * mask

    # inverse fourier transform
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # avoid divine zero
    img_filtered = img / (img_back + 1e-5)

    # apply gaussian filter to denoise
    gaussian_blur = cv2.GaussianBlur(img_filtered, kernel_size, sigma)

    height, width = img.shape[:2]

    start_x = width // 2 - crop // 2
    start_y = height // 2 - crop // 2

    # crop image
    cropped_img = gaussian_blur[start_y:start_y + crop, start_x:start_x + crop]

    # rescale image value to [0,1]
    processed_image = (cropped_img - np.min(cropped_img)) / (np.max(cropped_img) - np.min(cropped_img))

    return processed_image


# implementation of MORE algorithm
def MORE(speckle_list, num_epoch=5, sup_size=(200, 200)):
    speckles_FFTM = list()
    speckles_FFTP = list()

    for speckle in speckle_list:
        FFT = np.fft.fft2(speckle)

        # reduce the zero-frequency
        FFT[1, 1] = FFT[2, 2]

        # pre-calculate mag and phase of FFT
        speckles_FFTM.append(np.fft.fftshift(np.abs(FFT)))
        speckles_FFTP.append(np.fft.fftshift(np.fft.fft2(speckle)))

    height, width = speckle.shape[:2]

    # init guess of OTF
    s = np.random.rand(height, width)
    OTF = np.fft.fft2(s)

    # create mask of support mat
    sup_mat = np.zeros((height, width))
    rx, ry = sup_size
    sup_mat[(height // 2) - (rx // 2):(height // 2) + (rx // 2), (width // 2) - (ry // 2):(width // 2) + (ry // 2)] = 1

    # outer loop of MORE
    for i in range(num_epoch):
        # inner loop of MORE
        for j in range(len(speckle_list)):
            k_space = speckles_FFTM[j] * np.exp(1j * (np.angle(speckles_FFTP[j]) - np.angle(OTF)))
            r_space = np.real(np.fft.ifft2(np.fft.ifftshift(k_space)))

            # apply real and non-negative constraints
            r_sp = r_space * sup_mat
            r_sp[r_sp < 0] = 0

            # update OTF
            k_space = np.fft.fftshift(np.fft.fft2(r_sp))
            OTF = speckles_FFTP[j] / (k_space + 1e-5)

    # recover object with OTF
    objs_list = list()
    for i in range(len(speckle_list)):
        obj = np.real(np.fft.ifft2(
            np.fft.ifftshift(speckles_FFTM[i] * np.exp(1j * (np.angle(speckles_FFTP[i]) - np.angle(OTF))))))
        obj[obj < 0] = 0
        obj = obj * sup_mat
        obj = obj[(height // 2) - (rx // 2):(height // 2) + (rx // 2),
              (width // 2) - (ry // 2):(width // 2) + (ry // 2)]
        obj = (obj - np.min(obj)) / (np.max(obj) - np.min(obj))
        objs_list.append(obj)

    return objs_list, OTF


def savefigs(images, title, save_path):
    plt.figure(figsize=(16, 4))
    plt.suptitle(title, fontsize=20)

    for i, img in enumerate(images, start=1):
        plt.subplot(1, 5, i)
        plt.imshow(img, cmap='hot')
        plt.axis('off')

    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':

    speckles = list()

    for i, img_path in enumerate(glob.glob("experiment_speckles/*.bmp")):
        speckles.append(process_image(img_path))

    savefigs(speckles, 'processed speckles', './recovery_result/processed_speckles.png')

    objs, _ = MORE(speckles)

    savefigs(objs, 'recovered objects', './recovery_result/recovered_objects.png')
