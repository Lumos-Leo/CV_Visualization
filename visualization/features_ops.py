import torch

# ===========================获取指定层的特征图======================================= #
def extract_features(model:torch.nn.Module):
    feas = []
    import torch.nn.functional as F
    def get_fesa(module, input, output):
        feas.append(output)

    for name,sub in model.named_modules():
        if name == 'layers.5.residual_group.blocks':
            sub.register_forward_hook(get_fesa)

# ===========================指定层特征图可视化（带频域可视化）======================================= #
def visual_features(features: torch.Tensor, fre=False, path='./'):
    from einops import rearrange
    import matplotlib.pyplot as plt
    import numpy as np
    output = features.data.float().cpu().squeeze()
    C,H,W = output.shape
    if fre:
        # heatmap_fre_tot = np.zeros((H,W,3))
        magnitude_tot = np.zeros((H,W))
        for i in range(output.shape[0]):
            f = np.fft.fft2(output[i,:,:])
            f = np.fft.fftshift(f)
            magnitude = 20*np.log(np.abs(f)+1)
            magnitude_tot += magnitude
        magnitude_tot = ((magnitude_tot - magnitude_tot.min())/(magnitude_tot.max()-magnitude_tot.min()))
        magnitude_tot = (magnitude_tot * 255.0).round().astype(np.uint8)
    output_tot = np.zeros((H,W))
    for i in range(output.shape[0]):
        tmp = output[i,:,:]
        output_tot += tmp.numpy()
    output_tot = ((output_tot - output_tot.min())/(output_tot.max()-output_tot.min()))
    output_tot = (output_tot * 255.0).round().astype(np.uint8)
    plt.imshow(output_tot, cmap=plt.cm.jet)
    plt.savefig(path+'.png')
    if fre:
        # cv2.imwrite(path+'_fre.png', heatmap_fre_tot)
        plt.imshow(magnitude_tot, cmap=plt.cm.jet)
        # plt.colorbar()
        plt.savefig(path+'_fre.png')
        plt.clf()

# ===========================特征图之间相关性======================================= #
def get_the_mae(features):
    b, c,  = features.shape[0], features.shape[1]
    diff = torch.zeros((c,c))
    for i in range(c):
        for j in range(c):
            diff[i][j] = (torch.mean(torch.square(features[0,i,:,:] - features[0,j,:,:])))

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,8))
    xlabels = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16']
    ylabels = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16']
    sns.heatmap(diff, xticklabels=xlabels, yticklabels=ylabels, fmt='.2f', annot=True)
    plt.title('MSE between feature maps')
    plt.margins(0,0)
    plt.savefig('sns_heatmap_cmap.jpg', dpi=300)

    return diff
