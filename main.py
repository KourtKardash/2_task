import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import sobel, gaussian
from scipy.interpolate import RectBivariateSpline, splprep, splev
import skimage.io as skio


def display_image_in_actual_size(img, dpi = 80):
    height = img.shape[0]
    width = img.shape[1]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, cmap='gray')
    return fig, ax

def save_mask(fname, snake, img):
    plt.ioff()
    fig, ax = display_image_in_actual_size(img)
    ax.fill(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    fig.savefig(fname, pad_inches=0, bbox_inches='tight', dpi='figure')
    plt.close(fig)
    
    mask = skio.imread(fname)
    blue = ((mask[:,:,2] == 255) & (mask[:,:,1] < 255) & (mask[:,:,0] < 255)) * 255
    blue = np.array(blue, dtype='uint8')
    skio.imsave(fname, blue)
    plt.ion()
    
def display_snake(img, init_snake, result_snake):
    fig, ax = display_image_in_actual_size(img)
    ax.plot(init_snake[:, 0], init_snake[:, 1], '-r', lw=2)
    ax.plot(result_snake[:, 0], result_snake[:, 1], '-b', lw=2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

def CreateMatrix(n, alpha, beta, tau) :
    a = 2 * np.eye(n) + np.roll(-1 * np.eye(n), 1, axis=1) + np.roll(-1 * np.eye(n), -1, axis=1)
    b = 6 * np.eye(n) + np.roll(-4 * np.eye(n), 1, axis=1) + np.roll(-4 * np.eye(n), -1, axis=1) + \
        np.roll(np.eye(n), 2, axis=1) + np.roll(np.eye(n), -2, axis=1)
    A = alpha * a + beta * b
    return np.linalg.inv(np.eye(n) + tau * A) 
    
def GetPotential(img, w_line, w_edge) :
    P_line = gaussian(img, 3)
    P_edge = sobel(P_line)
    return w_line * P_line + w_edge * P_edge

def GetSegmentedImage(img, init_snake, alpha, beta, tau, w_line, w_edge, kappa) :
    x_cur, y_cur = init_snake[:, 0], init_snake[:, 1]
    n = init_snake.shape[0]
    norm_x = np.zeros(n)
    norm_y = np.zeros(n)
    matrix = CreateMatrix(n, alpha, beta, tau)
    potential = GetPotential(img, w_line, w_edge)
    height, width = potential.shape[0], potential.shape[1]
    interpolated_image = RectBivariateSpline(np.arange(width), np.arange(height), potential.T)
    for _ in range (800) :
        fx = interpolated_image(x_cur, y_cur, dx=1, grid=False)
        fx /= np.linalg.norm(fx)
        fy = interpolated_image(x_cur, y_cur, dy=1, grid=False)
        fy /= np.linalg.norm(fy)
        x_cur_shifted = np.roll(x_cur, -1)
        y_cur_shifted = np.roll(y_cur, -1)
        norm_x = y_cur_shifted - y_cur
        norm_y = x_cur - x_cur_shifted
        norm_x[-1] = 0
        norm_y[-1] = 0
        x_next = np.dot(matrix, x_cur + tau * (kappa * norm_x + fx))
        y_next = np.dot(matrix, y_cur + tau * (kappa * norm_y + fy))
        x_cur, y_cur = x_next, y_next
        x_cur[-1] = x_cur[0]
        y_cur[-1] = y_cur[0]
        tck, u = splprep([x_cur, y_cur], s=0, per=True)
        xi, yi = splev(np.linspace(0, 1, n), tck)
        xi[xi < 0] = 0
        yi[yi < 0] = 0
        xi[xi > width - 1] = width - 1
        yi[yi > height - 1] = height - 1
        x_cur, y_cur = xi, yi
    return np.concatenate((x_cur.reshape(-1, 1), y_cur.reshape(-1, 1)), axis=1)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image')
    parser.add_argument('initial_snake')
    parser.add_argument('output_image')
    parser.add_argument('alpha')
    parser.add_argument('beta')
    parser.add_argument('tau')
    parser.add_argument('w_line')
    parser.add_argument('w_edge')
    parser.add_argument('kappa')
    args = parser.parse_args()

alpha, beta = float(args.alpha), float(args.beta) 
tau, kappa = float(args.tau), float(args.kappa)
w_line, w_edge = float(args. w_line), float(args.w_edge)
img = np.array(cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)).astype('float')
init_snake = np.loadtxt(args.initial_snake)
snake = GetSegmentedImage(img, init_snake, alpha, beta, tau, w_line, w_edge, kappa)

save_mask(args.output_image, snake, img)

