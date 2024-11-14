import sys
import torch
import time
from src.MultiLabel.Non_Lightning.DualCoOp.utils.helper import AverageMeter, mAP, calc_F1
from torch.cuda.amp import autocast

def official_test(data_loader, model, label_emb=None, print_freq=100, thre=0.5):
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()

    Sig = torch.nn.Sigmoid()
    Softmax = torch.nn.Softmax()
    # switch to evaluate mode
    model.eval()

    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i,   (images, _, label_vec_obs, target, idx) in enumerate(data_loader):
            #target = target.max(dim=1)[0]
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            images = images.to(device)
            #print(images.dtype)

            # compute output
           # with autocast():
                # if label_emb is not None:
                #     output = model(images, label_emb)
                # else:
                #     output = model(images)
            # output = Softmax(output).cpu()[:, 1]
            if label_emb is not None:
                output, _, _  = model(images, label_emb, device)
            else:
                    # For SST:
                    #output, intraCoOccurrence, feature = model(images)

                    # for SARB
                    #output, _, _, _, _, _ = model(images)
                images = images.to('cuda')
                model = model.to('cuda')
                # for SCP
                output, _ = model(images)

                # For DualCoOp
                #output = model(images)

            # DualCoOp
            # if output.dim() == 3:
            #     output = Softmax(output.detach())[:, 1, :]
            # else:
            #     output = Sig(output.detach())
            output = Sig(output).cpu()
            # for mAP calculation
            preds.append(output.cpu())
            targets.append(target.cpu())

            # measure accuracy and record loss
            pred = output.data.gt(thre).long()
            pred = pred.to('cpu')
            target = target.to('cpu')
            output = output.to('cpu')


            tp += (pred + target).eq(2).sum(dim=0)
            fp += (pred - target).eq(1).sum(dim=0)
            fn += (pred - target).eq(-1).sum(dim=0)
            tn += (pred + target).eq(0).sum(dim=0)
            count += images.size(0)

            this_tp = (pred + target).eq(2).sum()
            this_fp = (pred - target).eq(1).sum()
            this_fn = (pred - target).eq(-1).sum()
            this_tn = (pred + target).eq(0).sum()

            this_prec = this_tp.float() / (
                    this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
            this_rec = this_tp.float() / (
                    this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

            prec.update(float(this_prec), images.size(0))
            rec.update(float(this_rec), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                                 i] > 0 else 0.0
                   for i in range(len(tp))]
            r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                                 i] > 0 else 0.0
                   for i in range(len(tp))]
            f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
                   i in range(len(tp))]

            mean_p_c = sum(p_c) / len(p_c)
            mean_r_c = sum(r_c) / len(r_c)
            mean_f_c = sum(f_c) / len(f_c)

            p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
            r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
            f_o = 2 * p_o * r_o / (p_o + r_o)

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                      'Recall {rec.val:.2f} ({rec.avg:.2f}) \t '
                      'P_C {P_C:.2f} \t R_C {R_C:.2f} \t F_C {F_C:.2f} \t P_O {P_O:.2f} \t R_O {R_O:.2f} \t F_O {F_O:.2f}'.format(
                    i, len(data_loader), batch_time=batch_time,
                    prec=prec, rec=rec, P_C=mean_p_c, R_C=mean_r_c, F_C=mean_f_c, P_O=p_o, R_O=r_o, F_O=f_o), flush=True)

        mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())

    torch.cuda.empty_cache()
    return mAP_score