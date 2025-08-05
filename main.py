import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import tqdm
import transferattack
from transferattack.utils import *
import multiprocessing
from time import time

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='sia',type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=16, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--beta', default=2.0, type=float, help='temp_hp')
    parser.add_argument('--num', default=20, type=int, help='temp_hp')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='inception_v4', type=str, help='the source surrogate model')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='/mnt/date10/public_data/ImageNet/val_sub', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default=f'./data_png', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--png', action='store_true', help='save formate')
    parser.add_argument('--GPU_ID', default='3', type=str)
    return parser.parse_args()


def delet_tensor(tensor,*index):
    b=tensor.cpu().detach().numpy()
    new_b=np.delete(b,index,axis=0)
    x=torch.Tensor(new_b).type_as(tensor)
    return x.cuda()

def main(opt):
    args = opt
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval, png=args.png)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    if not args.eval:
        if args.ensemble or len(args.model.split(',')) > 1:
            args.model = args.model.split(',')

        attacker = transferattack.load_attack_class(args.attack)(model_name=args.model, targeted=args.targeted)
        # i=0
        # avg_ce_loss,avg_ce_mean,avg_style_loss,avg_style_mean=[0 for _ in range(10)],[0 for _ in range(10)],[0 for _ in range(10)],[0 for _ in range(10)]
        all_time=0
        for images, labels, filenames in tqdm.tqdm(dataloader):
                perturbations = attacker(images, labels)
                save_images(args.output_dir, images+perturbations.cpu(), filenames,args.png)
    else:
        asr = dict()
        res = '|'
        for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper,robustbenchs):
            if model_name not in robustbenchs:
                model = wrap_model(model.eval().cuda())
            else:
                model=model.eval().cuda()
            for p in model.parameters():
                p.requires_grad = False
            correct, total = 0, 0
            for images, labels, fns in dataloader:
                if args.targeted:
                    labels = labels[1]
                ori_imgs=read_images(os.path.join(args.input_dir,'images'),fns)
                ori_logits=model(ori_imgs.cuda())
                ori_pred=torch.argmax(ori_logits,1)
                delet_index=(ori_pred!=labels.cuda()).nonzero(as_tuple=True)[0]
                images=delet_tensor(images,delet_index.cpu().detach().numpy())
                labels=delet_tensor(labels,delet_index.cpu().detach().numpy()) 
                pred = model(images.cuda())
                correct += (labels.detach().cpu().numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += labels.shape[0]
            if args.targeted:
                # correct: pred == target_label
                asr[model_name] = (correct / total) * 100
            else:
                # correct: pred == original_label
                asr[model_name] = (1 - correct / total) * 100
            print(model_name, asr[model_name])
            res += ' {:.1f} |'.format(asr[model_name])

        print(asr)
        print(res)
        with open('results.txt', 'a') as f:
            f.write(args.output_dir + res + '\n')

def read_images(root,filenames):
    batched_images=[]
    for fn in filenames:
        img_path=os.path.join(root,fn)
        img=Image.open(img_path)
        img=img.resize((img_height, img_width)).convert('RGB')
        img = np.array(img).astype(np.float32)/255
        img = torch.from_numpy(img).permute(2, 0, 1)
        batched_images.append(img.unsqueeze(0))
    return torch.cat(batched_images,0)


if __name__ == '__main__':

    opt=get_parser()
    main(opt)
