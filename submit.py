from networks import get_model


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):
    device_id = "cuda:0" 
    if not os.path.exists(respth):
        os.makedirs(respth)

    G = get_model(self.arch, pretrained=self.indicator).cuda()
    G.load_state_dict(torch.load())
    G.eval()
    
    save_pth = osp.join('res/cp', cp)
    

    # Model loading
    metrics = SegMetric(n_classes=self.classes)
    metrics.reset()
    for index, (images, labels) in enumerate(self.data_loader):
    if (index + 1) % 100 == 0:
        print('%d batches verifiered.' % (index + 1))
        images = images.cuda()
        labels = labels.cuda()
        h, w = labels.size()[1], labels.size()[2]

        torch.cuda.synchronize()
        tic = time.perf_counter()

        with torch.no_grad():
            outputs = G(images)
            # Whether or not multi branch?
            if self.arch == 'CE2P' or 'FaceParseNet' in self.arch:
                outputs = outputs[0][-1]

            outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
            pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
            gt = labels.cpu().numpy()
            metrics.update(gt, pred)

        torch.cuda.synchronize()





if __name__ == "__main__":
    
    
    
    
    evaluate(dspth='/var/mplab_share_data/xmc112062670/CelebAMask-HQ/CelebA-HQ-img/test', cp='{}_iter.pth'.format(119999))