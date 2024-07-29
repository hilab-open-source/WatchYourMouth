import os
import tqdm
import json
import torch
import utils
import argparse
import numpy as np
import textblob
from textblob import TextBlob
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from modules.ctc_decode import Decoder
from torch.utils.data import DataLoader
from datasets.wym_ctc_sentences import MouthActionSentence3D
from torch.optim.lr_scheduler import CosineAnnealingLR

import models.sentence_classification as Models

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    textblob.en.spelling.update({"upcoming":100000})

    device = torch.device('cuda')

    print('Loading Data...')

    train_dataset = MouthActionSentence3D(
        root=args.data_path,
        dataset=args.train_dataset, 
        num_points=args.num_points,
        padding=False
    )
    test_dataset = MouthActionSentence3D(
        root=args.data_path,
        dataset=args.test_dataset,
        num_points=args.num_points,
        padding=False
    )

    print('Creating Data Loaders...')
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers,
                                  pin_memory=True,
                                  collate_fn=utils.ctc_collate)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.workers,
                                 pin_memory=True,
                                 collate_fn=utils.ctc_collate)
    
    Model = getattr(Models, args.model)
    model = Model(channel=args.channel,
                  in_planes=args.in_planes,
                  radius=args.radius,
                  nsamples=args.nsamples,
                  spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size,
                  temporal_stride=args.temporal_stride,
                  dim=args.dim,
                  depth=args.depth,
                  heads=args.heads,
                  dim_head=args.dim_head,
                  dropout1=args.dropout1,
                  mlp_dim=args.mlp_dim,
                  num_points=args.num_points,
                  dropout2=args.dropout2)
    
    # if any(os.scandir(args.save_model)):
    #     print('Loading Model from checkpoints...')
    #     weights = torch.load(args.save_model+'model.pth', map_location=device)
    #     model.load_state_dict(weights)
    # else:
    print('Creating Model from scratch...')
        
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device=device)

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    lr_scheduler = CosineAnnealingLR(optimizer,
                                     T_max=args.epochs*int(len(train_dataloader)),
                                     eta_min=1e-5)
    
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    decoder = Decoder(train_dataset.letters)

    print('Start Training')
    if os.path.isfile(args.save_loss):
        with open(args.save_loss) as f:
            loss = f.readline().split(':')[-1]
            best_val_loss = float(loss)
    else:
        best_val_loss = np.inf
    print('Best evaluation loss is: ', best_val_loss)
    best_wer, best_cer = 1, 1
    best_wer_corr, best_cer_corr = 1, 1

    for epoch in range(args.epochs):
        print('Epoch ', epoch)
        model.train()
        train_loss = 0
        for video, text, video_len, text_len, user, position in tqdm(train_dataloader):
            video, text = video.to(device), text.to(device)
            optimizer.zero_grad()
            logits = model(video).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss_all = criterion(logits.log_softmax(-1), text,
                                     video_len, text_len)
            loss = loss_all.mean()
            if torch.isnan(loss).any():
                print('Skipping iteration with NaN loss!!!')
                continue
            weight = torch.ones_like(loss_all)
            dlogits = torch.autograd.grad(loss_all, logits, grad_outputs=weight)[0]
            logits.backward(dlogits)
            iter_loss = loss.item()
            optimizer.step()
            lr_scheduler.step()
            train_loss += iter_loss
        train_loss /= len(train_dataloader)
        print(f'Epoch {epoch} Train Loss: {train_loss}')

        model.eval()
        val_loss = 0
        pred, gt, user_order, pos = [], [], [], []
        pred_corr = []
        with torch.no_grad():
            for video_t, text_t, video_len_t, text_len_t, user_t, position_t in test_dataloader:
                video_t, text_t = video_t.to(device), text_t.to(device)
                logits_t = model(video_t).transpose(0, 1)
                with torch.backends.cudnn.flags(enabled=False):
                    loss_all_t = criterion(logits_t.log_softmax(-1), text_t, 
                                        video_len_t, text_len_t)
                loss_t = loss_all_t.mean()
                iter_loss_t = loss_t.item()
                val_loss += iter_loss_t
                decoded = decoder.decode_greedy(logits_t, video_len_t)
                target = decoder.idx_to_string(text_t, text_len_t, test_dataset.letters)
                print('True: ', target)
                print('Pred: ', decoded, '\n')
                pred.extend(decoded)
                gt.extend(target)
                user_order.extend(user_t)
                pos.extend(position_t)
            
            for p in pred:
                corrected = [str(TextBlob(s.lower()).correct()).upper() for s in p.split()]
                pred_corr.append(' '.join(corrected))

            cer = decoder.compute_cer_level_distance(pred, gt)
            wer = decoder.compute_wer_level_distance(pred, gt)

            cer_corr = decoder.compute_cer_level_distance(pred_corr, gt)
            wer_corr = decoder.compute_wer_level_distance(pred_corr, gt)

            if cer < best_cer:
                best_cer = cer
                torch.save(model.module.state_dict(), args.save_model+'model_best_cer.pth')
            
            if wer < best_wer:
                best_wer = wer
                best_performance = {
                    'best_wer': best_wer,
                    'best_cer': best_cer,
                    'user_order': user_order,
                    'positions': pos,
                    'prediction': pred,
                    'target': gt
                }
                json_acc = json.dumps(best_performance, indent=4)
                with open(args.save_performance, "w") as outfile:
                    outfile.write(json_acc)
            
            if cer_corr < best_cer_corr:
                best_cer_corr = cer_corr
            if wer_corr < best_wer_corr:
                best_wer_corr = wer_corr

            val_loss /= len(test_dataloader)
            if val_loss < best_val_loss:
                print('Model Improved! Save the current Model!')
                best_val_loss = val_loss
                with open(args.save_loss, 'w') as f:
                    f.write('best evaluation loss:'+str(best_val_loss))

            torch.save(model.module.state_dict(), args.save_model+'model_sentence_current.pth')
            print(f'Epoch {epoch} Test Loss: {val_loss}')
            print(f'Word Error Rate: {wer}, Character Error Rate: {cer}')
            print(f'Corrected Word Error Rate: {wer_corr}, Corrected Character Error Rate: {cer_corr}')
            print(f'Best Word Error Rate: {best_wer}, Best Character Error Rate: {best_cer}')
            print(f'Best Corrected Word Error Rate: {best_wer_corr}, Best Corrected Character Error Rate: {best_cer_corr}')
    print('End Training!')
   

def parse_args():
    parser = argparse.ArgumentParser(description='Sentence Point Cloud Lipreading')

    parser.add_argument('--data-path', default='path/to/your/dataset/', type=str)
    parser.add_argument('--train-dataset', default='Sentences/Train.txt', type=str)
    parser.add_argument('--test-dataset', default='Sentences/Test.txt', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='DepthSpeechRecognition', type=str)
    parser.add_argument('--save-model', default='./Log/checkpoints/', type=str)
    parser.add_argument('--save-loss', default='./Log/checkpoints/loss.txt', type=str)
    parser.add_argument('--save-performance', default='./Log/checkpoints/best_performance.json', type=str)
    # input
    parser.add_argument('--num-points', default=1024, type=int, metavar='N')
    # TNet
    parser.add_argument('--channel', default=6, type=int, help='number of channels')
    # P4D
    parser.add_argument('--in-planes', default=3, type=float)
    parser.add_argument('--radius', default=0.05, type=float)
    parser.add_argument('--nsamples', default=32, type=int)
    parser.add_argument('--spatial-stride', default=16, type=int)
    parser.add_argument('--temporal-kernel-size', default=3, type=int)
    parser.add_argument('--temporal-stride', default=1, type=int)
    # transformer
    parser.add_argument('--dim', default=80, type=int)
    parser.add_argument('--depth', default=5, type=int)
    parser.add_argument('--heads', default=3, type=int)
    parser.add_argument('--dim-head', default=40, type=int)
    parser.add_argument('--mlp-dim', default=160, type=int)
    parser.add_argument('--dropout1', default=0.5, type=float)
    # output
    parser.add_argument('--dropout2', default=0.5, type=float)
    # training
    parser.add_argument('-b', '--batch-size', default=8, type=int)  # At least 2 samples for each GPU
    parser.add_argument('--epochs', default=300, type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N')
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2, 3]))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'
    os.environ["WANDB_MODE"] = "dryrun"
    args = parse_args()
    main(args=args)
