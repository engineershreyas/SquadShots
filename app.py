from flask import Flask, redirect, url_for, request, render_template, session
from flask.ext.login import LoginManager, UserMixin, login_user, logout_user,\
    current_user
import requests
import urllib
import urllib2
from instagram import client
import cv
import cv2
import os
from PIL import Image
import numpy as np
from StringIO import StringIO
from azure.storage import BlobService
from azure.storage import TableService, Entity
import subprocess
import sys
import matplotlib.pyplot as plt
#from oauth2client.client import OAuth2WebServerFlow
from googleapishtuffs import Contacts




instagram_client_id = "5a8ab4abb86640ccb32a1997b339e6cd"
instagram_client_secret = "5239d6f5cf67469e85e16e2de6ace321"
code = ""
OUTPUT_DIRECTORY = "./face_root_directory/"
CASCADE = "./haarcascade_frontalface_alt.xml"
IMAGE_SCALE = 2
haar_scale = 1.2
min_neighbors = 3
min_size = (20, 20)
haar_flags = 0
normalized_face_dimensions = (100, 100)
i = 0




app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/download')
def download():
    blob_service = BlobService(account_name='squadshots', account_key='UgxaWKAKv2ZvhHrPt0IHi4EQedPpZw35r+RXkAYB2eICPrG3TjSwk2G8gUzG/PNDDTV+4CVCYWCvZSiad5xMQQ==')
    try:
        blob_service.get_blob_to_path('album','image','static/output.png')
    except Exception as e:
        print e
    blobs = blob_service.list_blobs('album')
    for i in blob_service.list_containers():
        print "This container is " + i.name


    return render_template('album.html',filename="static/output.png")

@app.route('/upload',methods=['POST'])
def upload():

    file = request.files['fileInput']
    print "File is" + file.filename


    if file:
        data = file.read()


        blob_service = BlobService(account_name='squadshots', account_key='UgxaWKAKv2ZvhHrPt0IHi4EQedPpZw35r+RXkAYB2eICPrG3TjSwk2G8gUzG/PNDDTV+4CVCYWCvZSiad5xMQQ==')
        blob_service.create_container('album')

        blob_service.put_block_blob_from_bytes(
            'album',
            file.filename + "_blob",
            data,
            x_ms_blob_content_type='image/png'
        )

        if 'username' in session:
            un = session['username']
        else:
            print "not in session"

        blob_service.set_blob_metadata(container_name="album",
                                   blob_name=file.filename + "_blob",
                                   x_ms_meta_name_values={'metaun': un})

        blob_service.get_blob_to_path('album',file.filename + "_blob",'static/output.png')
        f = open('input_person.png','w+')
        f.write(data)
        f.close()


        [X,y] = read_images(OUTPUT_DIRECTORY, (256,256))
    # Convert labels to 32bit integers. This is a workaround for 64bit machines,
        y = np.asarray(y, dtype=np.int32)

    # Create the Eigenfaces model.
        model = cv2.createEigenFaceRecognizer()
    # Learn the model. Remember our function returns Python lists,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
        model.train(np.asarray(X), np.asarray(y))

    # Save the model for later use
        model.save("eigenModel.xml")



           # Create an Eign Face recogniser
        t = float(100000)
        model = cv2.createEigenFaceRecognizer(threshold=t)

        # Load the model
        model.load("eigenModel.xml")

       # Read the image we're looking for
        try:
            sampleImage = cv2.imread('static/output.png', cv2.IMREAD_GRAYSCALE)
            if sampleImage != None:
                sampleImage = cv2.resize(sampleImage, (256,256))
            else:
                print "sample image is  null"
        except IOError:
            print "IO error"

      # Look through the model and find the face it matches
        [p_label, p_confidence] = model.predict(sampleImage)

    # Print the confidence levels
        print "Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence)

    # If the model found something, print the file path
        if (p_label > -1):
            count = 0
            for dirname, dirnames, filenames in os.walk(OUTPUT_DIRECTORY):
                for subdirname in dirnames:
                    subject_path = os.path.join(dirname, subdirname)
                    if (count == p_label):
                        for filename in os.listdir(subject_path):
                            print "subject path = " + subject_path

                    count = count+1

    return "uploaded"


@app.route('/get_contacts')
def get_contacts():
    code = request.values['code']
    credentials = flow.step2_exchange(code)
    contact = Contacts()

@app.route('/authenticate/')
def instagram_authentication():

    redirect_url = "http://127.0.0.1:5000" + url_for('handle_authentication')


    authentication_link = "https://api.instagram.com/oauth/authorize/?client_id=" + instagram_client_id + "&redirect_uri=" + redirect_url + "&response_type=code"
    return redirect(authentication_link)

@app.route('/handle_authentication')
def handle_authentication():

    code = request.values.get('code')
    redirect_url = "http://127.0.0.1:5000" + url_for('handle_authentication')

    if not code:
        return "Sorry, could not receive code"
    else:

        instagram_client = client.InstagramAPI(client_id=instagram_client_id, client_secret=instagram_client_secret,redirect_uri=redirect_url)
        access_token, instagram_user = instagram_client.exchange_code_for_access_token(code)



        session['username']  = instagram_user['username']
        session['id'] = instagram_user['id']
        session['token'] = access_token
        api = client.InstagramAPI(access_token=access_token)
        total = ""

        #flow = OAuth2WebServerFlow(client_id='402789993508-2u63d2a8p1i9miri5ua64gspuui1dtl9.apps.googleusercontent.com',client_secret='UlG5JsEZHiuz_RHRNDjC-7Kn',scope='https://www.google.com/m8/feeds https://www.googleapis.com/auth/contacts.readonly',redirect_uri='http://127.0.0.1:5000/get_contacts')
        #auth_uri = flow.step1_get_authorize_url()
        #return redirect(auth_uri)


        next_page = "http://127.0.0.1:5000" + url_for('main_page')
        people, next = api.user_follows(instagram_user['id'])
        while next:
            people, next = api.user_follows(with_next_url=next)
            for user in people:
                photo_url = user.profile_picture
                name = user.full_name
                print name
                get_face_in_photo(photo_url, name, user.id, access_token, False, None, None)

        return redirect(next_page)


@app.route('/main_page')
def main_page():
    return redirect("http://127.0.0.1:5000" + url_for('download'))



##face-recognition methods
def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def read_images(path, sz=None):
    X,y = [], []
    count = 0
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    extension = filename[-3:]
                    if extension == "jpg" or extension == "png":
                        print os.path.join(subject_path, filename)
                        try:
                            im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                        except IOError:
                            continue
                        if im == None:
                            print "im is nonetype"
                    else:
                        continue
                    # resize to given size (if given)
                    if (sz is not None and im != None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(count)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            count = count+1
    return [X,y]




def convert_rgb_to_bgr(open_cv_image):
    try:
        new_image = cv.CreateImage((open_cv_image.width, open_cv_image.height), cv.IPL_DEPTH_8U, open_cv_image.channels)
        cv.CvtColor(open_cv_image, new_image, cv.CV_RGB2BGR)
    except:
        print "Error converting image to BGR"
        return None
    return new_image

def download_photo_as_open_cv_image(photo_url):
    try:
        if photo_url != None:
            img = urllib2.urlopen(photo_url).read()
        else:
            print "photo_url is null"
    except urllib2.HTTPError:
        # possible case of 404 on image
        print "Error fetching image: %s" % photo_url
        return None
    img = StringIO(img)
    pil_image = Image.open(img)
    try:
        open_cv_image = cv.fromarray(np.array(pil_image))[:, :]
    except TypeError:
        print "unsupported image type"
        return None
    open_cv_image = convert_rgb_to_bgr(open_cv_image)
    return open_cv_image

def normalize_image_for_face_detection(img):
    gray = cv.CreateImage((img.width, img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / IMAGE_SCALE),
                   cv.Round(img.height / IMAGE_SCALE)), 8, 1)
    if img.channels > 1:
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
    else:
        # image is already grayscale
        gray = cv.CloneMat(img[:, :])
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)
    cv.EqualizeHist(small_img, small_img)
    return small_img


def face_detect_on_photo(img, constraint_coordinate):
    cascade = cv.Load(CASCADE)
    faces = []
    small_img = normalize_image_for_face_detection(img)
    faces_coords = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                        haar_scale, min_neighbors, haar_flags, min_size)
    for ((x, y, w, h), n) in faces_coords:
        if constraint_coordinate is not None and not _is_in_bounds((x, y, w, h), constraint_coordinate, small_img):
            print "Coordinate is not in bounds"
            continue
        pt1 = (int(x * IMAGE_SCALE), int(y * IMAGE_SCALE))
        pt2 = (int((x + w) * IMAGE_SCALE), int((y + h) * IMAGE_SCALE))
        face = img[pt1[1]:pt2[1], pt1[0]: pt2[0]]
        face = normalize_face_for_save(face)
        faces.append(face)
    return faces




def _create_folder_name(name):
    split_name = name.split(" ")
    first = split_name[0]
    if(len(split_name) > 1):
        last = split_name[len(split_name) - 1]
    else:
        last = "user"
    folder_name = "%s_%s" % (first, last)
    return folder_name

def save_face(name, face):
    folder_name = _create_folder_name(name)
    if not os.path.exists(OUTPUT_DIRECTORY + folder_name):
        os.makedirs(OUTPUT_DIRECTORY + folder_name)
    names = name.split(" ")
    if len(names) > 1:
        filename = "%s_%s.jpg" % (names[0],names[1])
    else:
        filename = "%s.jpg"

    full_path = OUTPUT_DIRECTORY + folder_name + "/" + filename
    try:
        cv2.imwrite(full_path, np.asarray(face))
    except UnicodeEncodeError:
        print "Did not save picture because of unicode exception"

    print "Saving: %s" % full_path



def normalize_face_size(face):
    normalized_face_dimensions = (100, 100)
    face_as_array = np.asarray(face)
    resized_face = cv2.resize(face_as_array, normalized_face_dimensions)
    resized_face = cv.fromarray(resized_face)
    return resized_face


def normalize_face_histogram(face):
    face_as_array = np.asarray(face)
    equalized_face = cv2.equalizeHist(face_as_array)
    equalized_face = cv.fromarray(equalized_face)
    return equalized_face


def normalize_face_color(face):
    gray_face = cv.CreateImage((face.width, face.height), 8, 1)
    if face.channels > 1:
        cv.CvtColor(face, gray_face, cv.CV_BGR2GRAY)
    else:
        # image is already grayscale
        gray_face = cv.CloneMat(face[:, :])
    return gray_face[:, :]


def normalize_face_for_save(face):
    face = normalize_face_size(face)
    face = normalize_face_color(face)
    face = normalize_face_histogram(face)
    return face
"""
"""
def check_other_photos(id_,token):
    redirect_url = "http://127.0.0.1:5000" + url_for('handle_authentication')
    api = client.InstagramAPI(access_token=token)
    recent_media,next = api.user_recent_media(user_id=id_)
    face_found = False
    while next and face_found == False:
        recent_media, next = api.user_recent_media(with_next_url=next)
        for photo in recent_media:
            photo_url = photo.images['standard_resolution'].url
            face_found = get_face_in_photo(photo_url,photo.user.full_name,id_,token,True,None,None)
            if face_found == True:
                break



#  @task
def get_face_in_photo(photo_url, name, id_, token, second, x, y):
    print "Get face in photo called"
    photo_in_memory = download_photo_as_open_cv_image(photo_url)
    if photo_in_memory is None:
        print "No photo in memory"
        return False
    if x is None and y is None:
        print "x is None and y is None"
        # case for profile picture that isnt necessarily tagged
        # only return a result if exactly one face is in the image
        faces = face_detect_on_photo(photo_in_memory, None)
        if len(faces) == 0:
            print "Zero faces found"
            if second != True:
                check_other_photos(id_,token)
                return False
        elif len(faces) == 1:
            print "One face found"
            save_face(name,faces[0])
            return True
        else:
            print "more than one face found"
            save_face(name,faces[0])
            return True
        return False
    for face in face_detect_on_photo(photo_in_memory, (x, y)):
        save_face(name, face)

    return True

def images_from_random_people(all_people, max_pics, recognizer):
    num_to_train = 8
    id_counter = 2
    random.shuffle(all_people)
    all_people = all_people
    num_training_added = 0
    for person in all_people:
        label_dict[recognizer][id_counter] = person
        if "DS_STORE" in face_dir + person:
            continue
        try:
            all_pictures = os.listdir(face_dir + person + "/")
        except:
            continue
        if len(all_pictures) < 20:
            continue
        random.shuffle(all_pictures)
        all_pictures = all_pictures[:max_pics]
        for picture_name in all_pictures:
            picture_dir = face_dir + person + "/"
            full_path = picture_dir + picture_name
            try:
                face = cv.LoadImage(full_path, cv2.IMREAD_GRAYSCALE)
            except IOError:
                continue
            yield face[:, :], id_counter
        num_training_added += 1
        if num_training_added > num_to_train:
            break
        id_counter += 1


def get_name(path):
    dName = os.path.basename(os.path.dirname(path))
    firstandlast = dName.split("_")
    full_name = " ".join(firstandlast)
    return full_name

def iterate_over_random_people():
    num_people = 5
    people_names = os.listdir(face_dir)
    random.shuffle(people_names)
    people_to_use = people_names[:num_people]
    # variable faces is a full image path
    for person in people_to_use:
        picture_dir = face_dir + person + "/"
        try:
            all_pictures = os.listdir(picture_dir)
        except:
            continue
        if len(all_pictures) == 0:
            continue
        some_picture = random.choice(all_pictures)
        full_path = picture_dir + some_picture
        variable_faces.append(full_path)
    for filename in variable_faces:
        try:
            image = cv.LoadImage(filename, cv2.IMREAD_GRAYSCALE)
        except IOError:
            continue
        yield image[:, :], filename



def train_recognizers(recognizers):
    for recognizer in recognizers:
        label_dict[recognizer] = {}
    images = []
    labels = []
    num_faces = 0
    max_pics = 50

    all_people = os.listdir(face_dir)
    person = random.choice(all_people)
    all_people.remove(person)
    for face, id_counter in images_from_target_person(person, max_pics, recognizers):
        images.append(np.asarray(face))
        labels.append(id_counter)

    for recognizer in recognizers:
        image_copy = list(images)
        label_copy = list(labels)
        for face, id_counter in images_from_random_people(all_people, max_pics, recognizer):
            image_copy.append(np.asarray(face))
            label_copy.append(id_counter)
            num_faces += 1

        image_array = np.asarray(image_copy)
        label_array = np.asarray(label_copy)
        recognizer.train(image_array, label_array)
    return recognizers




app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


if __name__ == '__main__':
    app.run(debug=True)
