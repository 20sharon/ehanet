from utils import *
import time
from metrics import SegMetric
import numpy as np

class Verifier(object):

    def __init__(self, data_loader, config):

        self.data_loader = data_loader
        self.classes = config.classes
        self.arch = config.arch
        self.count = 0

    def save_anno(self, parsing, image_id, respth):
        from PIL import Image
        import numpy as np
        import os
        
        labels_celeb = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
        
        output_dir = osp.join(respth, image_id)
        os.makedirs(output_dir, exist_ok=True)

        for class_idx, class_name in enumerate(labels_celeb):

            class_mask = (parsing == class_idx).astype(np.uint8) * 255

            output_path = os.path.join(output_dir, f'{class_name}.png')
            Image.fromarray(class_mask).save(output_path)

    def validation(self, G):
        G.eval()
        time_meter = AverageMeter()

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
                
                # for i in range(len(pred)):
                #     self.save_anno(pred[i], str(self.count), './test_res')
                #     self.count += 1
            torch.cuda.synchronize()
            time_meter.update(time.perf_counter() - tic)

        print("Inference Time per image: {:.4f}s".format(
            time_meter.average() / images.size(0)))
        print(self.count)
        # mIoU metric
        score = metrics.get_scores()[0]

        return score#score["Mean IoU : \t"]
