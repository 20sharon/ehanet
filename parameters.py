import argparse

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def str2bool(v):
    return v.lower() in ('true')


def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument(
        '--arch', type=str, choices=['UNet', 'DFANet', 'DANet', 'DABNet', 'CE2P', 'FaceParseNet18',
                                     'FaceParseNet34', "FaceParseNet50", "FaceParseNet101", "EHANet18"], required=True)

    # Training setting
    parser.add_argument('--epochs', type=int, default=45, #200,
                        help='how many times to update the generator')
    parser.add_argument('--pretrained_model', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--classes', type=int, default=19)

    # Testing setting
    # parser.add_argument('--test_size', type=int, default=2824)
    # parser.add_argument('--val_size', type=int, default=2993)
    parser.add_argument('--model_name', type=str, default='model.pth')

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)

    # Path
    parser.add_argument('--img_path', type=str,
                        #default='/var/mplab_share_data/xmc112062670/CelebAMask-HQ/CelebA-HQ-img/train')
                        default='/kaggle/input/train-img-zip/train_img')
    parser.add_argument('--label_path', type=str,
                        #default='./Data_preprocessing/train_label')
                        #default='/var/mplab_share_data/xmc112062670/CelebAMask-HQ/mask/train')
                        default='/kaggle/input/train-label-zip/train_label')
                        
    parser.add_argument('--model_save_path', type=str, default='/kaggle/input/models')  #'./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    
    parser.add_argument('--val_img_path', type=str,
                        #default='/var/mplab_share_data/xmc112062670/CelebAMask-HQ/CelebA-HQ-img/test')
                        default='/kaggle/input/test-img-zip/test_img')
    parser.add_argument('--val_label_path', type=str,
                        #default='/var/mplab_share_data/xmc112062670/CelebAMask-HQ/mask/test')
                        default='/kaggle/input/test-label-zip/test_label')

    parser.add_argument('--test_image_path', type=str,
                        #default='./Data_preprocessing/test_img')
                        #default='/var/mplab_share_data/xmc112062670/CelebAMask-HQ/CelebA-HQ-img/test/')
                        default='/kaggle/input/test-img-zip/test_img')
    parser.add_argument('--test_label_path', type=str,
                        #default='/var/mplab_share_data/xmc112062670/CelebAMask-HQ/mask/test')
                        default='/kaggle/input/test-label-zip//test_label')
    parser.add_argument('--test_pred_label_path', type=str,
                        default='./test_pred_results') # test pred results path
    parser.add_argument('--test_colorful', type=str2bool,
                        default=True) #False) # color test results switch
    parser.add_argument('--test_color_label_path', type=str,
                        default='./test_color_visualize') # colorful test pred results path

    # Step size
    parser.add_argument('--sample_step', type=int, default=500) # train results sample step
    parser.add_argument('--tb_step', type=int, default=100) # tensorboard write step

    return parser.parse_args()
