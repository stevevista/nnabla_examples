import numpy as np
import time
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from scipy.misc import imread, imresize, imsave


def reorg(input, stride=2):
    batch_size, input_channel, input_height, input_width = input.data.shape
    output_height, output_width, output_channel = int(input_height/stride), int(input_width/stride), input_channel*stride*stride
    output = F.transpose(F.reshape(input, (batch_size, input_channel, output_height, stride, output_width, stride)), (0, 1, 2, 4, 3, 5)) 
    output = F.transpose(F.reshape(output, (batch_size, input_channel, output_height, output_width, stride*stride)), (0, 4, 1, 2, 3)) 
    output = F.reshape(output, (batch_size, output_channel, output_height, output_width))
    return output

def yolo(x, classes, num, test=False):

    final_filters = (classes+5)*num

    def bn(xx, name):
        return PF.batch_normalization(xx, batch_stat=not test, name=name)

    def maxpool2(xx):
        return F.max_pooling(xx, (2,2), stride=(2,2))

    leaky = nn.Variable([1])
    leaky.d[0] = 0.1

    def bn_cnv_w3(xx, c, name):
        return F.prelu(bn(PF.convolution(xx, c, (3, 3), pad=(1, 1), stride=(1, 1), with_bias=False, name=name), name=name), leaky)

    def bn_cnv_w1(xx, c, name):
        return F.prelu(bn(PF.convolution(xx, c, (1, 1), pad=(0, 0), stride=(1, 1), with_bias=False, name=name), name=name), leaky)


    c1 = maxpool2(bn_cnv_w3(x, 32, name='l1'))
    c2 = maxpool2(bn_cnv_w3(c1, 64, name='l2'))
    c3 = bn_cnv_w3(c2, 128, name='l3')
    c4 = bn_cnv_w1(c3, 64, name='l4')
    c5 = maxpool2(bn_cnv_w3(c4, 128, name='l5'))
    c6 = bn_cnv_w3(c5, 256, name='l6')
    c7 = bn_cnv_w1(c6, 128, name='l7')
    c8 = maxpool2(bn_cnv_w3(c7, 256, name='l8'))
    c9 = bn_cnv_w3(c8, 512, name='l9')
    c10 = bn_cnv_w1(c9, 256, name='l10')
    c11 = bn_cnv_w3(c10, 512, name='l11')
    c12 = bn_cnv_w1(c11, 256, name='l12')
    c13 = bn_cnv_w3(c12, 512, name='l13') # tag

    c14 = bn_cnv_w3(maxpool2(c13), 1024, name='l14')
    c15 = bn_cnv_w1(c14, 512, name='l15')
    c16 = bn_cnv_w3(c15, 1024, name='l16')
    c17 = bn_cnv_w1(c16, 512, name='l17')
    c18 = bn_cnv_w3(c17, 1024, name='l18')
    c19 = bn_cnv_w3(c18, 1024, name='l19')
    c20 = bn_cnv_w3(c19, 1024, name='l20')

    c21 = bn_cnv_w1(c13, 64, name='l21')
    c22 = reorg(c21, 2)
    c23 = F.concatenate(c22, c20, axis=1)
    c24 = bn_cnv_w3(c23, 1024, name='l24')
    c25 = PF.convolution(c24, final_filters, (1, 1), pad=(0, 0), stride=(1, 1), name='l25')

    batch_size, channels, nr, nc = c25.shape
    reshaped = F.reshape(c25, [batch_size, num, channels/num, nr, nc])
    splited = F.split(reshaped, axis=1)

    predicts = []
    for u in splited:
        spatials = F.split(u, axis=1)
        box_x = F.sigmoid(spatials[0])
        box_y = F.sigmoid(spatials[1])
        box_w = F.exp(spatials[2])
        box_h = F.exp(spatials[3])
        box = F.stack(box_x, box_y, box_w, box_h, axis=1)
        obj = F.sigmoid(spatials[4])
        clsfi = F.stack(*spatials[5:], axis=1)
        clsfi = F.softmax(clsfi, axis=1)
        predicts.append((box, obj, clsfi))

    return c25, predicts



def load_conv(raw, params, k, filters, window, bn, scope):

    def read_filter_size():
        buf = raw.read(filters*4)
        return np.frombuffer(buf, dtype=np.float32).reshape([1,filters,1,1])

    if bn:
        params['%s/bn/beta'%scope].d = read_filter_size()
        params['%s/bn/gamma'%scope].d = read_filter_size()
        params['%s/bn/mean'%scope].d = read_filter_size()
        params['%s/bn/var'%scope].d = read_filter_size()
    else:
        params['%s/conv/b'%scope].d = np.frombuffer(raw.read(filters*4), dtype=np.float32)

    d = raw.read(filters*k*window*window*4)
    params['%s/conv/W'%scope].d = np.frombuffer(d, dtype=np.float32).reshape([filters,k,window,window])


def load_weights(path):
    params = nn.get_parameters(grad_only=False)

    weights = open(path, 'rb')
    hdr = weights.read(4*4)

    load_conv(weights, params, 3, 32, 3, True, 'l1')
    load_conv(weights, params, 32, 64, 3, True, 'l2')
    load_conv(weights, params, 64, 128, 3, True, 'l3')
    load_conv(weights, params, 128, 64, 1, True, 'l4')
    load_conv(weights, params, 64, 128, 3, True, 'l5')
    load_conv(weights, params, 128, 256, 3, True, 'l6')
    load_conv(weights, params, 256, 128, 1, True, 'l7')
    load_conv(weights, params, 128, 256, 3, True, 'l8')
    load_conv(weights, params, 256, 512, 3, True, 'l9')
    load_conv(weights, params, 512, 256, 1, True, 'l10')
    load_conv(weights, params, 256, 512, 3, True, 'l11')
    load_conv(weights, params, 512, 256, 1, True, 'l12')
    load_conv(weights, params, 256, 512, 3, True, 'l13')
    load_conv(weights, params, 512, 1024, 3, True, 'l14')
    load_conv(weights, params, 1024, 512, 1, True, 'l15')
    load_conv(weights, params, 512, 1024, 3, True, 'l16')
    load_conv(weights, params, 1024, 512, 1, True, 'l17')
    load_conv(weights, params, 512, 1024, 3, True, 'l18')
    load_conv(weights, params, 1024, 1024, 3, True, 'l19')
    load_conv(weights, params, 1024, 1024, 3, True, 'l20')
    load_conv(weights, params, 512, 64, 1, True, 'l21')
    load_conv(weights, params, 1280, 1024, 3, True, 'l24')
    load_conv(weights, params, 1024, 425, 1, False, 'l25')


classes=80
num=5

x=nn.Variable([1,3,608,608])
out,predicts = yolo(x, 80, 5, test=True)
nr=out.shape[2]
nc=out.shape[3]



box_biases = [
    (0.57273, 0.677385),
    (1.87446, 2.06253),
    (3.33843, 5.47434),
    (7.88282, 3.52778),
    (9.77052, 9.16828)
]


thresh = 0.24

def iou(box1,box2):
    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def do_nms_obj(probs, thresh):
    def cmp(a,b):
        if a.max == b.max:
            return 0
        elif a.max > b.max:
            return -1
        else:
            return 1

    probs.sort(cmp)

    for i in range(len(probs)):
        a = probs[i]
        for j in range(i+1,len(probs)):
            b = probs[j]
            if b.max == 0: continue
            if iou(a.box, b.box) > thresh:
                b.max = 0


def get_region_box(pred, biases, j, i):
    x = (i + pred[0][0][j][i])
    y = (j + pred[0][1][j][i])
    w = pred[0][2][j][i] * biases[0]
    h = pred[0][3][j][i] * biases[1]
    return [x,y,w,h]

class region(object):
    pass


def predict(imgpath):
    start_time = time.time()

    src_img = imread('/home/lc/data/giraffe.jpg').astype('float32')
    img = imresize(src_img, (608, 608))
    img = img.transpose([2,0,1])
    x.d = img / 255.

    with nn.auto_forward():
        _, predicts = yolo(x, 80, 5, test=True)

    i=0
    probs=[]
    for box, obj, clsfi in predicts:
        for row in range(nr):
            for col in range(nc):
                scale = obj.d[0][row][col]
                maxprob = 0
                maxidx = 0
                for j in range(classes):
                    prob = scale*clsfi.d[0][j][row][col]
                    if prob > maxprob:
                        maxprob = prob
                        maxidx = j

                r = region()
                if maxprob < thresh:
                    maxprob = 0
                else:
                    r.box = get_region_box(box.d, box_biases[i], row, col)
                r.max = maxprob
                r.maxidx = maxidx
                probs.append(r)

        i=i+1

    do_nms_obj(probs, 0.4)

    h_scale = src_img.shape[0]/(nr+0.0)
    w_scale = src_img.shape[1]/(nc+0.0)


    def draw_border(img, box):
        left,right,top,bot = box
        if left < 0: left = 0
        if top < 0: top = 0
        for i in range(left,right):
            img[top][i][0:3] = 0
            img[bot][i][0:3] = 0
        for i in range(top,bot):
            img[i][left][0:3] = 0
            img[i][right][0:3] = 0



    for r in probs:
        if r.max > 0.24:
            left  = (r.box[0]-r.box[2]/2.)*w_scale
            right = (r.box[0]+r.box[2]/2.)*w_scale
            top   = (r.box[1]-r.box[3]/2.)*h_scale
            bot   = (r.box[1]+r.box[3]/2.)*h_scale
            print 'class:', r.maxidx, r.max
            draw_border(src_img, [int(left),int(right),int(top),int(bot)])

    imsave('out.png', src_img)
    print("time: %4.4f" % (time.time() - start_time,))


if __name__ == '__main__':
    import sys 
    if len(sys.argv) < 3:
        print 'Usage: yolo path_to_yolo.weights path_to_img'
        sys.exit()
    wfile = sys.argv[1]
    imgfile = sys.argv[2]
    load_weights(wfile)
    predict(imgfile)
    
