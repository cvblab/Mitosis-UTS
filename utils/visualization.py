import matplotlib.pyplot as plt

from skimage import measure


def visualize_gt(im, mask):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    ax.imshow(im)

    labels = measure.label(mask)
    props = measure.regionprops(labels)  # centroid: (row, column) (y, x)

    for i_prop in props:
        circ = plt.Circle((i_prop.centroid[1], i_prop.centroid[0]), 25, color='g', linewidth=3, fill=False)  # (xy)
        ax.add_patch(circ)

    ax.axis('off')
    plt.show()