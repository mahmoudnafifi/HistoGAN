import os
import dlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import os.path as path
# Reference: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

def detect_face_landmarks(face_file_path=None,
                          predictor_path=None,
                          img=None):
  # References:
  # -   http://dlib.net/face_landmark_detection.py.html
  # -   http://dlib.net/face_alignment.py.html

  if predictor_path is None:
    predictor_path = './utils/shape_predictor_68_face_landmarks.dat'

  # Load all the models we need: a detector to find the faces, a shape predictor
  # to find face landmarks so we can precisely localize the face
  detector = dlib.get_frontal_face_detector()
  shape_predictor = dlib.shape_predictor(predictor_path)

  if img is None:
    # Load the image using Dlib
    print("Processing file: {}".format(face_file_path))
    img = dlib.load_rgb_image(face_file_path)

  shapes = list()

  # Ask the detector to find the bounding boxes of each face. The 1 in the
  # second argument indicates that we should upsample the image 1 time. This
  # will make everything bigger and allow us to detect more faces.
  dets = detector(img, 1)

  num_faces = len(dets)
  print("Number of faces detected: {}".format(num_faces))

  if num_faces < 1:
    raise Exception('No face found!')

  # Find the face landmarks we need to do the alignment.
  faces = dlib.full_object_detections()
  for d in dets:
    print("Left: {} Top: {} Right: {} Bottom: {}".format(
      d.left(), d.top(), d.right(), d.bottom()
    ))

    shape = shape_predictor(img, d)
    faces.append(shape)

  return faces



def recreate_aligned_images(json_data,
                            dst_dir='./temp-faces/',
                            output_size=1024,
                            transform_size=4096,
                            enable_padding=True):
    print('Recreating aligned images...')
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    for item_idx, item in enumerate(json_data.values()):
        print('\r%d / %d ... ' % (item_idx, len(json_data)), end='', flush=True)

        # Parse landmarks.
        # pylint: disable=unused-variable
        lm = np.array(item['in_the_wild']['face_landmarks'])
        filename = item['in_the_wild']['file_path']
        filename = path.split(filename)[-1]

        lm_chin = lm[0:17]  # left-right
        lm_eyebrow_left = lm[17:22]  # left-right
        lm_eyebrow_right = lm[22:27]  # left-right
        lm_nose = lm[27:31]  # top-down
        lm_nostrils = lm[31:36]  # top-down
        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        lm_mouth_outer = lm[48:60]  # left-clockwise
        lm_mouth_inner = lm[60:68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        print(eye_to_mouth.shape)
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        src_file = item['in_the_wild']['file_path']
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)),
                     int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))),
                int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
               int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0),
               max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]),
                                           (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
              1.0 - np.minimum(np.float32(x) / pad[0],
                               np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(
                np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(
              img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                      'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                            (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        # dst_subdir = os.path.join(
        #   dst_dir, '%05d' % (item_idx - item_idx % 1000))
        os.makedirs(dst_dir, exist_ok=True)
        print(f'Saving {os.path.join(dst_dir, filename)}')
        #img.save(os.path.join(dst_subdir, '%05d.png' % item_idx))
        img.save(os.path.join(dst_dir, filename))

    # All done.
    print('\r%d / %d ... done' % (len(json_data), len(json_data)))

    return



def face_extraction(face_file_path):

  faces = detect_face_landmarks(face_file_path)

  img = dlib.load_rgb_image(face_file_path)

  thumbnail_size = 512
  thumbnails = dlib.get_face_chips(img, faces, size=thumbnail_size)

  # The first face which is detected:
  # NB: we assume that there is exactly one face per picture!
  f = faces[0]

  parts = f.parts()

  num_face_landmarks=68

  v = np.zeros(shape=(num_face_landmarks, 2))
  for k, e in enumerate(parts):
    v[k, :] = [e.x, e.y]

  json_data = dict()

  item_idx = 0

  json_data[item_idx] = dict()
  json_data[item_idx]['in_the_wild'] = dict()
  json_data[item_idx]['in_the_wild']['file_path'] = face_file_path
  json_data[item_idx]['in_the_wild']['face_landmarks'] = v

  recreate_aligned_images(json_data)
