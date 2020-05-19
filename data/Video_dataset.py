import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
from PIL import ImageFilter
#import cv2
import numpy as np
from util.util import extract_smooth_areas
import torch
import PIL
import os
import random
from moviepy.editor import VideoFileClip
from imgaug import augmenters as iaa
import imgaug as ia

class VideoDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.path_videos = os.path.join(opt.dataroot, opt.phase + 'A/')
        self.annotation_path = os.path.join(opt.dataroot, opt.phase+'annot' + 'A/')


        self.A_size = len(sorted(os.listdir(self.path_videos)))



        transform_ = [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])

        ]
        transform = [transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                        ]
        if(self.opt.input_nc==1):
            transform_ = [
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5],
                   #                  [0.5, 0.5, 0.5])
                transforms.Normalize((0.5,), (0.5,))
            ]
        elif(self.opt.input_nc==3):
            transform_ = [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])
                # transforms.Normalize((0.5,), (0.5,))
            ]

        self.transform = transforms.Compose(transform_)


        videos = sorted(os.listdir(self.path_videos))
        annotations = sorted(os.listdir(self.annotation_path))

        self.Vid = [self.path_videos + v + '/' for v in videos]
        self.Ann = [self.annotation_path + a for a in annotations]



    def read_annotation_file2(self,file_path, video_path):
        info_dict = {}
        file1 = open(file_path, 'r')

        while True:
            # Get next line from file
            line = file1.readline()
            # if line is empty
            # end of file is reached
            if not line:
                break
            line = line.strip()
            line = line.split(',')

            info_dict.setdefault(line[0].split('.')[0], []).append(line[1:])
            # if line[0] not in info_dict:
            #   info_dict[line[0]] = line[1:]
            # else:
            #   print(line[1:])
            #   x = []
            #  x.append(info_dict[line[0]])
            # info_dict[line[0]] = x.append(line[1:])

        file1.close()
        info_dict2 = {}
        V = sorted(os.listdir(video_path))
        V1 = []
        for i in range(0, len(V)):
            key = V[i].split('.')[0]
            if key in info_dict:
                V1.append(V[i])

        return info_dict, V1

    def apply_transformation(self, image):
        c = self.opt.input_nc
        image = image.astype(np.uint8)

        sometimes = lambda aug: iaa.Sometimes(.1, aug)
        seq_image = iaa.Sequential([
            sometimes(iaa.Add((-10, 10), per_channel=0.5)),
            # change brightness of images (by -10 to 10 of original value)
            #sometimes(iaa.AddToHueAndSaturation((-20, 20))),
        ]
        )
        seq_all = iaa.Sequential([

            iaa.Fliplr(0.4),  # horizontally flip 50% of all images
            iaa.Flipud(0.6),  # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                #shear=(-16, 16),  # shear by -16 to +16 degrees
                #order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                #cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ])

        images_aug = seq_image.augment_images([image])

        all = np.empty([image.shape[0], image.shape[1], c], dtype=np.uint8)

        if (c == 1):
            all[:, :, 0] = images_aug[0]
        elif (c == 3):
            all[:, :, 0:c] = images_aug[0]

        all_aug = seq_all.augment_images([all])

        return Image.fromarray(np.uint8(all_aug[0]))

    def get_paired_videos(self,video_path, annotation_file):
        list_A = []
        list_B = []
        #V = sorted(os.listdir(video_path))
        annot_dict,V = self.read_annotation_file2(annotation_file,video_path)
        #for i in range(0, len(V)):
        video = V[random.randint(0, len(V) - 1)]

        annots = annot_dict[video.split('.')[0]]
        if len(annots) == 0:
            annots = [annots]
        ann = annots[random.randint(0, len(annots) - 1)]


        timeA1 = int(ann[1].split(':')[0]) * 3600 + int(ann[1].split(':')[1]) * 60 + int(ann[1].split(':')[2])
        timeA2 = int(ann[2].split(':')[0]) * 3600 + int(ann[2].split(':')[1]) * 60 + int(ann[2].split(':')[2])

        timeB1 = int(ann[3].split(':')[0]) * 3600 + int(ann[3].split(':')[1]) * 60 + int(ann[3].split(':')[2])
        timeB2 = int(ann[4].split(':')[0]) * 3600 + int(ann[4].split(':')[1]) * 60 + int(ann[4].split(':')[2])

        videoA = VideoFileClip(video_path + video)
        try:
            videoB = VideoFileClip(video_path + ann[0])
        except:
            ext = ann[0].split('.')[1]
            if ext == 'flv':
                videoB = VideoFileClip(video_path + ann[0].split('.')[0] + '.mp4')
            else:
                videoB = VideoFileClip(video_path + ann[0].split('.')[0] + '.flv')

        x = timeB1
        y = timeA1

        X = ann[0].split('.')[0]
        Y = video.split('.')[0]


        try:
            #xx = random.randint(timeB1, timeB2 - 2)
            yy = random.randint(timeA1, timeA2 - 2)
            A = videoA.subclip(yy, yy + 2)
            B = videoB.subclip(timeB1+(yy-timeA1), timeB1+(yy-timeA1) + 2)

        except:
            xa = 0
            xb = 0

            A = videoA.subclip(xa, min(xa + 2,int(videoA.duration)))
            B = videoB.subclip(xb, min(xb + 2,int(videoB.duration)))

        framesA = []
        framesB = []
        for frame in A.iter_frames():
            framesA.append(frame)

        for frame in B.iter_frames():
            framesB.append(frame)


        return framesA, framesB

    def get_unpaired_videos(self,video_path1, video_path2):
        V1 = sorted(os.listdir(video_path1))
        V2 = sorted(os.listdir(video_path2))

        video1 = V1[random.randint(0, len(V1) - 1)]
        video2 = V2[random.randint(0, len(V2) - 1)]




        videoA = VideoFileClip(video_path1 + video1)
        videoB = VideoFileClip(video_path2 + video2)

        xa = random.randint(0, int(videoA.duration) - 2)
        xb = random.randint(0, int(videoB.duration) - 2)

        try:
            videoA = videoA.subclip(xa, xa + 2)
            videoB = videoB.subclip(xb, xb + 2)
        except:
            videoA = videoA.subclip(0, min(0 + 2, int(videoA.duration)))
            videoB = videoB.subclip(0, min(0 + 2, int(videoB.duration)))



        framesA = []
        framesB = []
        for frame in videoA.iter_frames():
            framesA.append(frame)

        for frame in videoB.iter_frames():
            framesB.append(frame)

        # print(len(framesA))
        # print(len(framesB))
        # videoA.write_gif("A111.gif", fps=15)
        # videoB.write_gif("A222.gif", fps=15)

        return framesA, framesB


    def __getitem__(self, index):

        #A_path = self.A_paths[1]

        if(self.opt.input_nc==1):
            s ='L'
        elif(self.opt.input_nc==3):
            s='RGB'
        index = index%self.A_size
        ###Paired
        A1,A2 = self.get_paired_videos(self.Vid[index],self.Ann[index])
        ####Not Paired
        B1,B2 = self.get_unpaired_videos(self.Vid[random.randint(0,self.A_size-1)],self.Vid[random.randint(0,self.A_size-1)])

        list_tensorA1=[]
        list_tensorA2=[]

        for a1,a2 in zip(A1,A2):
            pil_a1 = Image.fromarray(a1).resize((128,128),Image.BICUBIC)
            a1_t = self.transform(pil_a1)
            list_tensorA1.append(a1_t)
            #pil_a2 = self.apply_transformation(np.array(Image.fromarray(a2).resize((128,128),Image.BICUBIC)))
            pil_a2 = Image.fromarray(a2).resize((128,128),Image.BICUBIC)
            a2_t = self.transform(pil_a2)
            list_tensorA2.append(a2_t)

        list_tensorB1 = []
        list_tensorB2 = []

        for b1, b2 in zip(B1, B2):
            pil_b1 = Image.fromarray(b1).resize((128, 128), Image.BICUBIC)
            b1_t = self.transform(pil_b1)
            list_tensorB1.append(b1_t)
            pil_b2 = Image.fromarray(b2).resize((128, 128), Image.BICUBIC)
            b2_t = self.transform(pil_b2)
            list_tensorB2.append(b2_t)



        A11 = torch.stack(list_tensorA1)
        A22 = torch.stack(list_tensorA2)
        B11 = torch.stack(list_tensorB1)
        B22 = torch.stack(list_tensorB2)

        #A11 = [list_tensorA1]
        #A22 = [list_tensorA2]
        #B11 = [list_tensorB1]
        #B22 = [list_tensorB2]

        return {'A1': A11,'A2': A22,'B1': B11,'B2': B22,'A_paths':self.path_videos}

    def __len__(self):
        return 100
        return 5000
        return self.A_size

    def name(self):
        return 'VideoDataset'