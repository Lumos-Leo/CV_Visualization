import matplotlib.pyplot as plt
import torch
import numpy as np

from PIL import Image
from torchvision.transforms import transforms 

# ===========================图像放缩======================================= #
def scale_image(path, scale):
    img = Image.open(path)
    w, h = img.size
    img_x2 = img.resize((w//scale, h//scale), resample=Image.BICUBIC)
    img_x2.save(path.split('.')[0]+'_scale.png')

# ===========================直方图======================================= #
def draw_histogram(path):
    plt.figure()
    labels = ['1', '2', '3', '4', '5']
    y1 = [0.1348,0.9999,0.9938,0.9975,0.0711,1.0000,1.0000,1.0000,0.0361,0.8649,0.1108,1.0000,0.0104,0.2340,1.0000,0.1005].sort()
    y2 = [0.9936,0.6699,0.8878,0.0479,0.2529,0.8340,0.4752,0.2570,0.9115,0.2424,0.3269,0.4807,0.8444,0.9860,0.7761,0.1936].sort()
    y3 = [0.1160,1.0000,0.2762,1.0000,0.1739,1.0000,0.3655,0.3386,0.2902,0.2745,0.2755,1.0000,1.0000,0.1414,0.7498,0.3521].sort()
    y4 = [0.4027,0.5063,0.3615,0.1793,0.3555,0.6465,0.5135,1.0000,0.4005,0.3201,0.6962,0.2700,0.5110,0.9001,0.5695,0.6446].sort()
    y5 = [0.2748,0.9998,0.3185,0.8779,0.9924,0.4448,0.9998,0.1454,0.3088,0.1400,0.9908,0.1009,0.1388,0.1151,0.1300,0.2037].sort()
    x1 = [i+1 for i in range(len(y1))]
    x2 = [i+x1[-1]+2 for i in range(len(y1))]
    x3 = [i+x2[-1]+2 for i in range(len(y1))]
    x4 = [i+x3[-1]+2 for i in range(len(y1))]
    x5 = [i+x4[-1]+2 for i in range(len(y1))]
    plt.bar(x1, y1, color='red')
    plt.bar(x2, y2, color='green')
    plt.bar(x3, y3, color='blue')
    plt.bar(x4, y4, color='orange')
    plt.bar(x5, y5, color='yellow')

    plt.xticks([i*len(y1) + 9+i for i in range(len(labels))], labels)
    plt.ylim(0, 1.2)
    plt.ylabel('Sparsity')
    plt.savefig(path, dpi = 300)

# ===========================残差图======================================= #
def draw_residual_imgs(path1, path2, path_out):
    img_hr,_,_ = Image.open(path1).convert('YCbCr').split()
    img_sr,_,_ = Image.open(path2).convert('YCbCr').split()

    residual = transforms.ToPILImage()(torch.abs(transforms.ToTensor()(img_sr) - transforms.ToTensor()(img_hr)))
    residual.save(path_out)

# ===========================伪彩色图======================================= #
def heat_imgs(path):
    plt.figure()
    img = plt.imread(path)
    img = img/255.
    plt.imshow(img, cmap = plt.cm.jet)
    plt.colorbar()
    plt.savefig(path.split('.')[0]+'_heat.png')

# ===========================不同模型参数量对比图======================================= #
def model_cmp():
    x = [1.57, 52.70, 6.00, 5.50, 2.07, 2.26, 4.55, 14.00, 29.90, 91.20]
    y = [37.27, 36.66, 37.00, 37.06, 37.21, 37.36, 36.83, 37.38, 37.52, 37.53]
    params = [9.90, 24.00, 12.46, 25.00, 9.03, 14.63, 21.18, 60.00, 813.00, 412.00]
    colors=list(np.arange(1,len(params)+1)/len(params))
    params = np.array(params)
    area = np.pi * 16 * 20 * params/(np.pi * 4 * 20 )
    plt.figure()
    plt.xlabel('Number of MACs (G)')
    plt.ylabel('PSNR (db)')
    plt.scatter(x, y, alpha=0.8, s=area, c=colors)
    plt.grid()
    plt.ylim(36.5, 37.7)
    plt.xlim(0, 100)
    # plt.legend()
    plt.annotate('SGSR-M5', (1.57,37.27), (1.34+2.5,37.27-0.08), weight="bold", color="r", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))
    plt.annotate('SRCNN', (52.70,36.66), (52.70-4.0,36.66+0.1), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate('FSRCNN', (6.00,37.00), (6.00+1.0,37.00-0.1), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate('MOREMNAS-C', (5.50,37.06), (5.50+4.0,37.06-0.02), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate('SESR-M3', (2.07,37.21), (2.07-2.0,37.21-0.1), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    # plt.annotate('ICCV21', (3.10,37.32), (3.10+2.0,37.32-0.08), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate('SGSR-M8', (2.22,37.36), (2.22-1.0,37.36+0.1), weight="bold", color="r", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))
    plt.annotate('ESPCN', (4.55,36.83), (4.55-1.0,36.83-0.1), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate('TPSR-NoGAN', (14.00,37.38), (14.00-1.0,37.38-0.1), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    # plt.annotate('VDSR', (612.60,37.53), (612.60-1.0,37.53-0.1), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate('LapSRN', (29.90,37.52), (29.90-1.0,37.52-0.1), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate('CARN-M', (91.20,37.53), (91.20-4.0,37.53-0.1), weight="bold", color="b", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.savefig('./samples/modelsPK.png', dpi = 300)

# ============================三维散点图======================================== #
def draw_3D_figs():
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.view_init(elev=14, azim=-34)
    ax.invert_xaxis()
    
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    x = [ 52.70, 6.00, 5.50, 2.07, 4.55, 14.00, 3.10, 2.34]
    y = [ 36.66, 37.00, 37.06, 37.21, 36.83, 37.38, 37.32, 37.33]
    params = [ 24.00, 12.46, 25.00, 9.03, 21.18, 60.00, 14.00, 10.20]

    x_our = [2.26]
    y_our = [37.36]
    param_our = [14.63]
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zlow, zhigh)
    ax.scatter(x, params, y, c='b', marker='o')
    ax.scatter(x_our, param_our, y_our, c='r', marker='*', s=80, depthshade=False)
    ax.text( 2.26,14.93, 37.39, "SRGFS(Ours)")
    ax.text( 52.90,24.30, 36.66, "SRCNN(2014)")
    ax.text( 6.00,12.46, 37.02, "FSRCNN(2016)")
    ax.text( 5.50,25.20, 37.00, "MOREMNAS(2020)")
    ax.text( 2.37,9.23, 37.11, "SESR(2021)")
    ax.text( 5.10,21.58, 36.83, "ESPCN(2016)")
    ax.text( 14.00,60.00, 37.38, "TPSR-NoGAN(2019)")
    ax.text( 3.45,14.50, 37.27, "ICCV(2021)")
    ax.text( 2.84,10.50, 37.23, "ACMM(2021)")

    label = ["SRCNN", "FSRCNN", "MOREMNAS", "SESR", "ESPCN", "TPSR-NoGAN", "ICCV", "ACMMM", "SRGFS(Ours)"]
    
    ax.set_xlabel('Params (K)')
    ax.set_ylabel('Number of MACs (G)')
    ax.set_zlabel('PSNR (dB)')
    
    plt.savefig('./samples/modelsPK_3d.png', dpi = 300)

# ===========================ensemble测试======================================= #
def ensemble(path):
    img_hr,_,_ = Image.open(path).convert('YCbCr').split()

    # 8种旋转方式
    img_hr.save('./samples/{}.png'.format('img_0'))
    img_90 = img_hr.rotate(90)
    img_90.rotate(-90).save('./samples/{}.png'.format('img_-90'))
    img_90.save('./samples/{}.png'.format('img_90'))
    img_180 = img_hr.rotate(180)
    img_180.save('./samples/{}.png'.format('img_180'))
    img_180.rotate(-180).save('./samples/{}.png'.format('img_-180'))
    img_270 = img_hr.rotate(270)
    img_270.save('./samples/{}.png'.format('img_270'))
    img_270.rotate(-270).save('./samples/{}.png'.format('img_-270'))
    img_hr_flip = img_hr.transpose(Image.FLIP_TOP_BOTTOM)
    img_hr_flip.save('./samples/{}.png'.format('img_hr_flip_0'))
    img_hr_flip_90 = img_hr_flip.rotate(90)
    img_hr_flip_90.save('./samples/{}.png'.format('img_hr_flip_90'))
    img_hr_flip_180 = img_hr_flip.rotate(180)
    img_hr_flip_180.save('./samples/{}.png'.format('img_hr_flip_180'))
    img_hr_flip_270 = img_hr_flip.rotate(270)
    img_hr_flip_270.save('./samples/{}.png'.format('img_hr_flip_270'))

    img_hr_flip.transpose(Image.FLIP_TOP_BOTTOM).save('./samples/{}.png'.format('img_hr_flip_-0'))
    img_hr_flip_90.rotate(-90).transpose(Image.FLIP_TOP_BOTTOM).save('./samples/{}.png'.format('img_hr_flip_-90'))
    img_hr_flip_180.rotate(-180).transpose(Image.FLIP_TOP_BOTTOM).save('./samples/{}.png'.format('img_hr_flip_-180'))
    img_hr_flip_270.rotate(-270).transpose(Image.FLIP_TOP_BOTTOM).save('./samples/{}.png'.format('img_hr_flip_-270'))
