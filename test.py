import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from skimage.measure import compare_psnr
from util import util
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


if __name__ == '__main__':


    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads = 1
    #opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test
    psnr=[]
    model.eval()
    k=0

    def save_im_emb(im,emb,k,type=['A','0']):
        im_emb = emb.cpu().float().numpy()
        im = Image.fromarray(util.tensor2im(im))
        embedding_path = './results/' + opt.name + '/test_' + str(opt.epoch) + '/'+ type[0] + str(k) + type[1] + '.npy'
        im_path = './results/' + opt.name + '/test_' + str(opt.epoch) + '/'+ type[0] + str(k) + type[1] + '.png'
        im.save(im_path)
        np.save(embedding_path, im_emb)

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()

        im_emb = model.outA1
        im = model.A1V
        save_im_emb(im, im_emb, k, type=['A', '1'])

        im_emb = model.outA2
        im = model.A2V
        save_im_emb(im, im_emb, k, type=['A', '2'])

        im_emb = model.outB1
        im = model.B1V
        save_im_emb(im, im_emb, k, type=['B', '1'])

        im_emb = model.outB2
        im = model.B2V
        save_im_emb(im, im_emb, k, type=['B', '2'])

        k+=1


