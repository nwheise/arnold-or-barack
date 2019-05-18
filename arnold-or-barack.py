import cv2
import os
import numpy as np
import scipy.misc
import shutil


DATA_FOLDER = 'data'
FACES_FOLDER = 'faces'
ROI_IMAGE_FOLDER = 'images_with_roi'
HAAR_CASCADE_FILENAME = 'haarcascade_frontalface_alt.xml'


def detect_and_save_faces(name: str, roi_size: tuple):
    '''
    Use haar cascades to detect the face in images in the folder specified
    by the name parameter. Resize to roi_size and save the face image.

    Parameters
    name: str, folder name containing images
    roi_size: tuple, pixel dimensions for resized cropped face image
    '''

    # define source and output directories
    dir_images = os.path.join(DATA_FOLDER, name)
    dir_faces = os.path.join(DATA_FOLDER, name, FACES_FOLDER)
    if not os.path.isdir(dir_faces):
        os.makedirs(dir_faces)

    # get face cascade for face detection
    cascade_filepath = os.path.join(DATA_FOLDER, HAAR_CASCADE_FILENAME)
    face_cascade = cv2.CascadeClassifier(cascade_filepath)

    # detect for each image the face and store this in the face directory
    image_names = image_names_in_dir(dir_images)
    for i in range(len(image_names)):
        # Get image and convert to grayscale
        img = cv2.imread(os.path.join(dir_images, image_names[i]))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the face and create an image of it
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        face = faces[0]
        x, y, width, height = face[0], face[1], face[2], face[3]
        face_img = img[y:y+height, x:x+width]

        # Rescale image to fit roi_size
        face_img = cv2.resize(face_img, roi_size, interpolation=cv2.INTER_CUBIC)

        # Save the face image in the face directory
        cv2.imwrite(os.path.join(dir_faces, image_names[i]), face_img)


def image_names_in_dir(dir_name: str):
    '''
    Return a list of string names of all files in a directory with .jpg file
    extension.

    Parameters
    dir_name: str, directory name
    '''

    image_names = []
    for filename in os.listdir(dir_name):
        if filename.endswith('.jpg'):
            image_names.append(filename)

    return image_names


def visualize_roi_on_images(name: str):
    '''
    Draw a rectangle around the detected faces in the images in the folder.
    Save these images to another folder.

    Parameters
    name: str, folder name containing images
    '''

    # define directories for finding images and saving the images with ROI's visualized
    dir_images = os.path.join(DATA_FOLDER, name)
    dir_images_with_roi = os.path.join(DATA_FOLDER, name, ROI_IMAGE_FOLDER)
    if not os.path.isdir(dir_images_with_roi):
        os.makedirs(dir_images_with_roi)

    # get face cascade for face detection
    cascade_filepath = os.path.join(DATA_FOLDER, HAAR_CASCADE_FILENAME)
    face_cascade = cv2.CascadeClassifier(cascade_filepath)

    # detect for each image the face and draw a rectangle around it
    image_names = image_names_in_dir(dir_images)
    for i in range(len(image_names)):
        # Get image and convert to grayscale
        img = cv2.imread(os.path.join(dir_images, image_names[i]))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the face
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        face = faces[0]
        x, y, width, height = face[0], face[1], face[2], face[3]

        # Overlay rectangle ROI on original image
        img_with_roi = np.copy(img)
        cv2.rectangle(img=img_with_roi,
                      pt1=(x, y),
                      pt2=(x+width, y+height),
                      color=(0, 255, 0),
                      thickness=2)

        # Save to a new directory
        cv2.imwrite(os.path.join(dir_images_with_roi, image_names[i]),
                    img_with_roi)


def construct_data_matrix(name: str, roi_size: tuple, numbers: list):
    # define where to look for the detected faces
    dir_faces = os.path.join(DATA_FOLDER, name, FACES_FOLDER)

    # put all faces in a list
    names_faces = [f'{n}.jpg' for n in numbers]

    # put all faces as data vectors in a data matrix X
    N = len(names_faces) # number of faces
    P = roi_size[0]*roi_size[1] # total number of pixels
    X = np.zeros(shape=(N, P))

    # For each image, place all rows of image data into one row of X
    for i in range(N):
        img = cv2.imread(os.path.join(dir_faces, names_faces[i]))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        num_rows = gray_img.shape[0]
        num_cols = gray_img.shape[1]
        for row_num in range(num_rows):
            X[i][(row_num*num_cols):(row_num+1)*num_cols] = gray_img[row_num]

    return X

def do_pca_and_build_model(name: str, roi_size: tuple, numbers: list):
    '''
    Build a matrix of the images given by the parameter numbers. Perform PCA
    on the matrix and return the mean, eigenvalues, and eigenvectors.

    Parameters
    name: str, folder name containing images
    roi_size: tuple, pixel dimensions for resized cropped face image
    numbers: list, names of the files to be included in the model
    '''

    X = construct_data_matrix(name, roi_size, numbers)

    # perform pca on X
    number_of_components = N
    mean, eigenvalues, eigenvectors = pca(X, number_of_components)

    return [mean, eigenvalues, eigenvectors]


def test_images(name, roi_size, numbers, models):
    '''
    Compare images given by the parameter numbers to the models via projection
    and reconstruction. Return the reconstructions and the MSE between them
    and the true images.

    Parameters
    name: str, folder name containing images
    roi_size: tuple, pixel dimensions for resized cropped face image
    numbers: list, names of the files to be included in the model
    models:
    '''

    X = construct_data_matrix(name, roi_size, numbers)

    # reconstruct the images in X with each of the models provided and also calculate the MSE
    # store the results as [[results_model_arnold_reconstructed_X, results_model_arnold_MSE],
    #                       [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    results = []
    for model in models:
        # Get mean-subtracted images
        X_ms = X
        X_ms = X_ms - model[0]
        # Project, reconstruct, find mse
        projections, reconstructions = project_and_reconstruct(X_ms, model)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        results.append([reconstructions, mse])

    return results


def pca(X, number_of_components):
    '''

    Parameters
    X:
    number_of_components:
    '''

    # Get the mean of each column of X
    mean = np.mean(X, axis=0)

    # Center matrix by subtracting mean from each image
    X_ctr = X
    X_ctr = X_ctr - mean

    # Get covariance matrix
    X_cov = np.cov(X_ctr.T)

    # Get eigenvals and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(X_cov)

    # Make a list of (mean, eigenvalue, eigenvector) tuples
    pca_components = [(mean[i], np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(eigenvalues.shape[0])]

    # Sort the tuples from high to low by abs(eigenvalue), remove small eigenvalues  
    pca_components = sorted(pca_components, key=lambda x: x[1], reverse=True)
    pca_components = pca_components[0:number_of_components]

    # Reconstruct individual arrays of eigenvalues, and eigenvectors
    eigenvalues = pca_components[:][1]
    eigenvectors = np.zeros(shape=(number_of_components, X.shape[1]))
    for i in range(len(pca_components)):
        eigenvectors[i] = pca_components[i][2]
    eigenvectors = eigenvectors.T

    # NOTE: eigenvectors[:,i] corresponds to eigenvalue[i]
    # i.e. eigenvectors are stored as columns

    return [mean, eigenvalues, eigenvectors]


def project_and_reconstruct(X, model):
    '''

    Parameters
    X:
    model:
    '''

    # projection Z = XV
    projections = np.dot(X, model[2])
    # reconstruction = ZV.T = XVV.T
    reconstructions = np.dot(projections, model[2].T)

    # Add mean to get final reconstruction
    for i in range(reconstructions.shape[0]):
        reconstructions[i] = np.add(reconstructions[i], model[0])

    return [projections, reconstructions]


def visualize_model(name, model, roi_size):
    '''

    Parameters
    name:
    model:
    roi_size:
    '''

    eig_vecs = model[2].T
    num_of_faces = eig_vecs.shape[0]

    # Make a folder to save eigfaces to, or recreate it if already created
    # since we don't want eigenvalues from previous PCAs to remain
    dir_eigfaces = "data/{}/eigfaces".format(name)
    if os.path.isdir(dir_eigfaces): shutil.rmtree(dir_eigfaces)
    os.makedirs(dir_eigfaces)

    # Make list of names
    eigface_names = ["eigface" + str(i) + ".png" for i in range(num_of_faces)]

    # Each eigenvector is an eigenface, so create images of these
    eig_faces = np.zeros(shape=(num_of_faces, roi_size[0], roi_size[1]))
    face_row_size = eig_faces.shape[2]
    face_column_size = eig_faces.shape[1]
    for i in range(num_of_faces):
        for j in range(face_column_size):
            eig_faces[i][j] = eig_vecs[i][ j*face_row_size: (j+1)*face_row_size ]

    # save images in the folder
    for i in range(num_of_faces):
        scipy.misc.imsave(os.path.join(dir_eigfaces, eigface_names[i]),
                          eig_faces[i])


def visualize_reconstructions(name, model_name, reconstructions, roi_size):
    '''

    Parameters
    name:
    model_name:
    reconstructions:
    roi_size:
    '''

    num_of_recs = reconstructions.shape[0]
    dir_reconstructions = "data/{}/reconstructions".format(name)
    if not os.path.isdir(dir_reconstructions):
        os.makedirs(dir_reconstructions)

    # Make list of names
    recface_names = [model_name + "reconstruction" + str(i) + ".png" for i in range(len(reconstructions))]

    # Make images from the reconstructions
    rec_faces = np.zeros( (num_of_recs, roi_size[0], roi_size[1]) )
    face_row_size = rec_faces.shape[2]
    face_column_size = rec_faces.shape[1]
    for i in range(num_of_recs):
        for j in range(face_column_size):
            rec_faces[i][j] = reconstructions[i][ j*face_row_size: (j+1)*face_row_size ]

    # save images in the folder
    for i in range(num_of_recs):
        scipy.misc.imsave(os.path.join(dir_reconstructions, recface_names[i]), rec_faces[i])


def main():
    roi_size = (50, 50)  # for reasonably quick computation time

    # Detect all faces in all the images in the folder of a person (in this case "arnold" and "barack") and save them in a subfolder "faces" accordingly
    detect_and_save_faces("arnold", roi_size=roi_size)
    detect_and_save_faces("barack", roi_size=roi_size)

    # visualize detected ROIs overlayed on the original images and copy paste these figures in a document 
    ########################################################################################################################
    # See function written above. The visualize_roi_on_images function follows almost exactly the detect_and_save_faces,   #
    # but instead of cropping, resizing, and saving the face images it simply draws a rectangle over                       #
    # the original images and saves them. It would be more efficient to include this in the detect_and_save_faces          #
    # function, but for clarity for the assignment I wrote a new function to be called here.                               #
    ########################################################################################################################
    visualize_roi_on_images("arnold")
    visualize_roi_on_images("barack")

    # Perform PCA on the previously saved ROIs and build a model=[mean, eigenvalues, eigenvectors] for the corresponding person's face making use of a training set
    model_arnold = do_pca_and_build_model(name="arnold", roi_size=roi_size,
                                          numbers=[1, 2, 3, 4, 5, 6])
    model_barack = do_pca_and_build_model(name="barack", roi_size=roi_size,
                                          numbers=[1, 2, 3, 4, 5, 6])

    # visualize these models
    ################################################################################################
    # Each model can be visualized by its eigenfaces, which are images created by each eigenvector #
    # in the model. These functions create each eigenface and save them in the directory           #
    ################################################################################################
    visualize_model("arnold", model_arnold, roi_size)
    visualize_model("barack", model_barack, roi_size)

    # Test and reconstruct "unseen" images and check which model best describes it (wrt MSE)
    # results=[[results_model_arnold_reconstructed_X, results_model_arnold_MSE], [results_model_barack_reconstructed_X, results_model_barack_MSE]]
    # The correct model-person combination should give best reconstructed images and therefore the lowest MSEs
    results_arnold = test_images(name="arnold",
                                 roi_size=roi_size,
                                 numbers=[7, 8],
                                 models=[model_arnold, model_barack])
    results_barack = test_images(name="barack",
                                 roi_size=roi_size,
                                 numbers=[7, 8, 9, 10],
                                 models=[model_arnold, model_barack])

    # visualize the reconstructed images
    ########################################################################################
    # The function visualize_reconstructions simply takes each reconstruction, and creates #
    # the corresponding image out of it.                                                   #
    ########################################################################################
    visualize_reconstructions(name="arnold",
                              model_name="arnold",
                              reconstructions=results_arnold[0][0],
                              roi_size=roi_size)
    visualize_reconstructions(name="arnold",
                              model_name="barack",
                              reconstructions=results_arnold[1][0],
                              roi_size=roi_size)
    visualize_reconstructions(name="barack",
                              model_name="arnold",
                              reconstructions=results_barack[0][0],
                              roi_size=roi_size)
    visualize_reconstructions(name="barack",
                              model_name="barack",
                              reconstructions=results_barack[1][0],
                              roi_size=roi_size)

    # Print raw MSE data from the results
    print("MSE: Arnold unseen compared to Arnold model")
    print(results_arnold[0][1])
    print("MSE: Arnold unseen compared to Barack model")
    print(results_arnold[1][1])
    print("MSE: Barack unseen compared to Arnold model")
    print(results_barack[0][1])
    print("MSE: Barack unseen compared to Barack model")
    print(results_barack[1][1])

    # Count number of cases where the MSE for the correct image classification
    # was less than the MSE for the incorrect image classification
    correct = 0
    total = 0
    for i in range(len(results_arnold[0][1])):
        total += 1
        if results_arnold[0][1][i] < results_arnold[1][1][i]:
            correct += 1
    for i in range(len(results_barack[0][1])):
        total += 1
        if results_barack[0][1][i] > results_barack[1][1][i]:
            correct += 1

    print(f'{correct} out of {total} images correctly identified.')


if __name__ == '__main__':
    main()
