import numpy as np


def norm_HnE(img, Io=240, alpha=1, beta=0.15):
    ######## Step 1: Convert RGB to OD ###################
    ## reference H&E OD matrix.
    # Can be updated if you know the best values for your image.
    # Otherwise use the following default values.
    # Read the above referenced papers on this topic.
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    ### reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])

    # extract the height, width and num of channels of image
    h, w, c = img.shape

    # reshape image to multiple rows and 3 columns.
    # Num of rows depends on the image size (wxh)
    img = img.reshape((-1, 3))

    # calculate optical density
    # OD = −log10(I)
    # OD = -np.log10(img+0.004)  #Use this when reading images with skimage
    # Adding 0.004 just to avoid log of zero.

    OD = -np.log10((img.astype(float) + 1) / Io)  # Use this for opencv imread
    # Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)

    ############ Step 2: Remove data with OD intensity less than β ############
    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)]  # Returns an array where OD values are above beta
    # Check by printing ODhat.min()

    ############# Step 3: Calculate SVD on the OD tuples ######################
    # Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    ######## Step 4: Create plane from the SVD directions with two largest values ######
    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])  # Dot product

    ############### Step 5: Project data onto the plane, and normalize to unit length ###########
    ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
    # find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T

    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    ###### Step 8: Convert extreme values back to OD space
    # recreate the normalized image using reference mixing matrix

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # Separating H and E components

    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    return (Inorm, H, E)


# img=cv2.imread('images/HnE_Image.jpg', 1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def norm_Masson(img, Io=240, alpha=1, beta=0.15):
    ######## Step 1: Convert RGB to OD ###################
    # Masson's Trichrome OD matrix (adjusted)

    # ESTA ES LA ORIGINAL NO BORRAR
    # TRRef = np.array([[0.6176, 0.2065],  # Adjust these values based on your knowledge of the staining
    #                 [0.0200, 0.9095],
    #                 [0.2500, 0.6144]])

    TRRef = np.array([[0.7, 0.1],  # Adjust these values based on your knowledge of the staining
                      [0.15, 0.9],
                      [0.2, 0.8]])
    # reference maximum stain concentrations for Masson
    maxCRef = np.array([2.0, 2])

    # extract the height, width and num of channels of image
    h, w, c = img.shape

    # reshape image to multiple rows and 3 columns.
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log10((img.astype(float) + 1) / Io)  # Use float instead of np.float

    ############ Step 2: Remove data with OD intensity less than β ############
    ODhat = OD[~np.any(OD < beta, axis=1)]  # Remove transparent pixels

    ############# Step 3: Calculate SVD on the OD tuples ######################
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    ######## Step 4: Create plane from the SVD directions with two largest values ######
    That = ODhat.dot(eigvecs[:, 1:3])  # Project on the plane

    ############### Step 5: Normalize to unit length ###########
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        TR = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        TR = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T

    C = np.linalg.lstsq(TR, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    ###### Step 8: Convert extreme values back to OD space
    Inorm = np.multiply(Io, np.exp(-TRRef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # Separating components (collagen and other stains)
    Collagen = np.multiply(Io, np.exp(np.expand_dims(-TRRef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    Collagen[Collagen > 255] = 254
    Collagen = np.reshape(Collagen.T, (h, w, 3)).astype(np.uint8)

    Other = np.multiply(Io, np.exp(np.expand_dims(-TRRef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    Other[Other > 255] = 254
    Other = np.reshape(Other.T, (h, w, 3)).astype(np.uint8)

    return (Inorm, Collagen, Other)

def norm_Masson_modified(img, TRRef, maxCRef,  Io=240, alpha=1, beta=0.15):
    """
        Normalizes an image based on the Masson staining method and separates its components.

        This function calculates the optical density of the input image, performs Singular Value
        Decomposition (SVD) to find the directions of maximum variance, normalizes the stain
        concentrations, and reconstructs the image along with its collagen and other stain components.

        Parameters:
            img (numpy.ndarray): A 3D NumPy array representing the input image with shape (height, width, channels).
            TRRef (numpy.ndarray): A 2D array of reference transformation values for the stains, with shape (3, 2).
            maxCRef (numpy.ndarray): Un array 1D de concentraciones máximas de colorantes, con forma (2,).
            Io (float, optional): The intensity of light used during imaging. Defaults to 240.
            alpha (float, optional): The percentile for determining the range of angles used in normalization. Defaults to 1.
            beta (float, optional): The threshold below which pixels are considered transparent and excluded. Defaults to 0.15.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The normalized image in optical density space (Inorm).
                - numpy.ndarray: The collagen component extracted from the image.
                - numpy.ndarray: The other stain component extracted from the image.
        """

    # extract the height, width and num of channels of image
    h, w, c = img.shape

    # reshape image to multiple rows and 3 columns.
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log10((img.astype(float) + 1) / Io)

    ############ Step 2: Remove data with OD intensity less than β ############
    ODhat = OD[~np.any(OD < beta, axis=1)]  # Remove transparent pixels

    ############# Step 3: Calculate SVD on the OD tuples ######################
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    ######## Step 4: Create plane from the SVD directions with two largest values ######
    That = ODhat.dot(eigvecs[:, 1:3])  # Project on the plane

    ############### Step 5: Normalize to unit length ###########
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        TR = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        TR = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T

    C = np.linalg.lstsq(TR, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    ###### Step 8: Convert extreme values back to OD space
    Inorm = np.multiply(Io, np.exp(-TRRef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # Separating components (collagen and other stains)
    Collagen = np.multiply(Io, np.exp(np.expand_dims(-TRRef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    Collagen[Collagen > 255] = 254
    Collagen = np.reshape(Collagen.T, (h, w, 3)).astype(np.uint8)

    Other = np.multiply(Io, np.exp(np.expand_dims(-TRRef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    Other[Other > 255] = 254
    Other = np.reshape(Other.T, (h, w, 3)).astype(np.uint8)

    return (Inorm, Collagen, Other)
