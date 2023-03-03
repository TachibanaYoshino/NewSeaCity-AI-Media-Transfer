import os, base64
from tqdm.notebook import tqdm
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))
import cv2, queue, threading
import tensorflow as tf
import numpy as np
from source.facelib.facer import FaceAna
import source.utils as utils
from PIL import Image
from source.mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()


alpha_img = Image.open("models/alpha.jpg")
facer = FaceAna(["models/detector.pb", "models/keypoints.pb"])

class Videocap:
    def __init__(self, name, limit=1024):

        vid = cv2.VideoCapture(name)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = vid.get(cv2.CAP_PROP_FPS)
        self.ori_width, self.ori_height = width, height

        # max_edge = max(width, height)
        # scale_factor = limit / max_edge if max_edge > limit else 1.
        # height = int(round(height * scale_factor))
        # width = int(round(width * scale_factor))
        # self.width, self.height = self.to_8s(width), self.to_8s(height)

        self.count = 0  # Records the number of frames entered into the queue.
        self.cap =vid
        self.ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.q =  queue.Queue(maxsize=100)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            # print("get me")
            self.ret, frame = self.cap.read()
            if not self.ret:
                break
            self.q.put(frame[:,:,::-1])
            self.count+=1
        self.cap.release()

    def read(self):
        f = self.q.get()
        self.q.task_done()
        return f


class Cartoonizer():
    def __init__(self, dataroot, name):
        self.name = name
        self.facer = facer
        self.sess_head = self.load_sess(os.path.join(dataroot, name+'_cartoon_h.pb'), 'model_head')
        self.sess_bg = self.load_sess(os.path.join(dataroot, name+'_cartoon_bg.pb'), 'model_bg')

        self.box_width = 288
        global_mask = alpha_img.resize((self.box_width, self.box_width),Image.ANTIALIAS)
        self.global_mask = np.array(global_mask).astype(np.float32) / 255.0
        # self.global_mask = cv2.cvtColor(np.array(global_mask), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def load_sess(self, model_path, name):
        # print(model_path, name)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # print(f'loading model from {model_path}')
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name=name+self.name)
            sess.run(tf.global_variables_initializer())
        # print(f'load model {model_path} done.')
        return sess


    def detect_face(self, img):
        src_h, src_w, _ = img.shape
        src_x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, landmarks, _ = self.facer.run(src_x)
        if boxes.shape[0] == 0:
            return None
        else:
            return landmarks


    def cartoonize(self, img):
        # img: RGB input
        ori_h, ori_w, _ = img.shape
        img = utils.resize_size(img, size=720)

        img_brg = img[:, :, ::-1]

        # background process
        pad_bg, pad_h, pad_w = utils.padTo16x(img_brg)

        bg_res = self.sess_bg.run(
            self.sess_bg.graph.get_tensor_by_name(
                f'model_bg{self.name}/output_image:0'),
            feed_dict={f'model_bg{self.name}/input_image:0': pad_bg})
        res = bg_res[:pad_h, :pad_w, :]

        landmarks = self.detect_face(img_brg)
        if landmarks is None:
            # print('No face detected!')
            return res

        # print('%d faces detected!'%len(landmarks))
        for landmark in landmarks:
            # get facial 5 points
            f5p = utils.get_f5p(landmark, img_brg)

            # face alignment
            head_img, trans_inv = warp_and_crop_face(
                img,
                f5p,
                ratio=0.75,
                reference_pts=get_reference_facial_points(default_square=True),
                crop_size=(self.box_width, self.box_width),
                return_trans_inv=True)

            # head process
            head_res = self.sess_head.run(
                self.sess_head.graph.get_tensor_by_name(
                    f'model_head{self.name}/output_image:0'),
                feed_dict={
                    f'model_head{self.name}/input_image:0': head_img[:, :, ::-1]
                })

            # merge head and background
            head_trans_inv = cv2.warpAffine(
                head_res,
                trans_inv, (np.size(img, 1), np.size(img, 0)),
                borderValue=(0, 0, 0))

            mask = self.global_mask
            mask_trans_inv = cv2.warpAffine(
                mask,
                trans_inv, (np.size(img, 1), np.size(img, 0)),
                borderValue=(0, 0, 0))
            mask_trans_inv = np.expand_dims(mask_trans_inv, 2)

            res = mask_trans_inv * head_trans_inv + (1 - mask_trans_inv) * res

        # res = cv2.resize(res, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
        res = Image.fromarray(np.uint8(res)).resize( (ori_w, ori_h), Image.ANTIALIAS)

        return np.array(res)[:,:,::-1]  # rgb


    def Convert_video(self, input_video_path, save_dir):
        # for op in self.sess_land.graph.get_operations():
        #     print(op.name, op.values())
        # load video
        save_path = os.path.join(save_dir, os.path.basename(input_video_path).rsplit('.',1)[0]+f"_{self.name}.mp4")
        vid = Videocap(input_video_path)
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        num = vid.total
        video_out = cv2.VideoWriter( save_path, codec, vid.fps, (vid.ori_width, vid.ori_height))
        pbar = tqdm(total=vid.total, )
        pbar.set_description(f"Running: {os.path.basename(input_video_path).rsplit('.', 1)[0] + f'_{self.name}.mp4'}")
        while num>0:
            if vid.count < vid.total and vid.ret == False and vid.q.empty():
                pbar.close()
                video_out.release()
                return "The video is broken, please upload the video again."
            frame = vid.read()
            fake_img = self.cartoonize(frame)
            video_out.write(cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR))
            pbar.update(1)
            num-=1
        pbar.close()
        video_out.release()
        # 使用ffmpeg 对生成视频进行压缩
        # cmd = f'sleep 1 && ffmpeg -i {input_video_path.rsplit(".",1)[0]+f"_{style}.mp4"} -vcodec libx264 -f mp4 -nostdin -an {input_video_path.rsplit(".",1)[0]+f"_C_{style}.mp4"}'
        # flag = os.system(cmd)
        # if flag==0: # 压缩成功，返回压缩后的视频
        #     return input_video_path.rsplit(".",1)[0]+f"_C_{style}.mp4"
        # else: # 否则返回未压缩的视频
        return save_path








