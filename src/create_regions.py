# function to create regions for the given image
from skimage.measure import label, regionprops
from skimage import exposure


def create_regions(mask, parameters):
    # find end of monolayer and beginning of sprouting region based on post position

    label_mask = label(segmentation_dilation)

    left_post_x_centroids = []
    left_post_x_left_border = []
    right_post_x_centroids = []

    for region in regionprops(label_mask):
    # to determine start of sprouting region:
    ## use centroids of objects that are between 15% and 50% of the image
    # to determine where monolayer ends:
    ## use left border of bounding boxes of objects that are between 15% and 50% of the image
    if region.centroid[1] < label_mask.shape[1]/2 and region.centroid[1] > label_mask.shape[1]*0.1:
        left_post_x_centroids.append(region.centroid[1])
        left_post_x_left_border.append(region.bbox[1])

    # to determine end of sprouting region:
    ## use centroids of objects that are between 50% and 85% of the image
    elif region.centroid[1] > label_mask.shape[1]/2 and region.centroid[1] < label_mask.shape[1]*0.85:
        right_post_x_centroids.append(region.centroid[1])

    sprouting_start = np.mean(left_post_x_centroids)
    sprouting_end = np.mean(right_post_x_centroids)
    monolayer_end = np.mean(left_post_x_left_border)

    fig, ax = plt.subplots(1,2, sharey=True)
    ax[0].imshow(exposure.equalize_adapthist(image_with_af), cmap='hot')
    ax[0].vlines(sprouting_start, label_mask.shape[1]*0.05, label_mask.shape[1]*0.95, color='White')
    ax[0].vlines(sprouting_end, label_mask.shape[1]*0.05, label_mask.shape[1]*0.95, color='White')
    ax[0].vlines(monolayer_end, label_mask.shape[1]*0.05, label_mask.shape[1]*0.95, color='White')
    ax[1].imshow(image_af_removed, cmap='hot')
    plt.savefig('drive/MyDrive/segmentation_posts/Output/' + IMAGE_NAME + '.png')