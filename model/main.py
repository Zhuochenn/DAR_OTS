import os
import torch
import configargparse
import model.components.data_loader as data_loader
import model.components.models as models
import numpy as np
import random
import utils.utils as utils
from model.components.predict_tSNE import tSNE_gen,inf_force
from utils.utils import str2bool

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, default="model/DANN/DANN.yaml")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')

    # data loading related
    parser.add_argument('--src_dir', type=str, default="data/mb1i1-wmb2i2/source/train")  
    parser.add_argument('--tar_dir', type=str, default="data/mb1i1-wmb2i2/target/train") 
    parser.add_argument('--fp', type=str, default="forcePosition/force.csv")
    parser.add_argument('--img', type=str, default="images")
    parser.add_argument('--rfimage_source', type=str, default="data/mb1i1-wmb2i2/source/ref.jpg")
    parser.add_argument('--rfimage_target', type=str, default="data/mb1i1-wmb2i2/target/ref.jpg")
    parser.add_argument('--mean_std', type=str, default="mean_std.npy")
    parser.add_argument('--pretrained', type=str, default='inference/mb1i1-wmb2i2/Source_only/source/model_r.pth')

    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--out_class', type=int, default=361)
    parser.add_argument('--out_dim', type=int, default=3)
    parser.add_argument('--early_stop', type=int, default=20)

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)

    # transfer related
    parser.add_argument('--transfer_weight', type=float, default=1)
    parser.add_argument('--clf_weight', type=float, default=1)
    parser.add_argument('--reg_weight', type=float, default=1)
    parser.add_argument('--transfer_loss', type=str, default='mmd')

    # saving related
    parser.add_argument('--model_dir', type=str, default='model.pth')
    parser.add_argument('--log_dir', type=str, default='loss.csv')
    parser.add_argument('--domain', type=str, default="mb1i1-wmb2i2")

    # inf related
    parser.add_argument('--test_source_dir', type=str, default='data/mb1i1-wmb2i2/source/test')
    parser.add_argument('--test_target_dir', type=str, default='data/mb1i1-wmb2i2/target/test')
    parser.add_argument('--inf_s', type=str, default='s.png')
    parser.add_argument('--inf_t', type=str, default='t.png')

    #tSNE
    parser.add_argument('--tSNE_source_dir', type=str, default='data/mb1i1-wmb2i2/source/ori')
    parser.add_argument('--tSNE_target_dir', type=str, default='data/mb1i1-wmb2i2/target/ori')
    parser.add_argument('--tSNE', type=str, default='tSNE.png')

    parser.add_argument('--inf_dir', type=str, default='inference/v3')
    parser.add_argument('--train', type=str2bool, default='True')
    parser.add_argument('--draw_tsne', type=str2bool, default='True')
    parser.add_argument('--draw_force', type=str2bool, default='True')
    parser.add_argument('--use_pretrained', type=str2bool, default='True')
    parser.add_argument('--use_checkpoint', type=str2bool, default='True')
    parser.add_argument('--data_save', type=str, default='source_only')

    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
  
    src_img = os.path.join(args.src_dir, args.img)
    src_fp = os.path.join(args.src_dir, args.fp)

    tgt_img = os.path.join(args.tar_dir, args.img)
    tgt_fp = os.path.join(args.tar_dir, args.fp)

    test_img = os.path.join(args.test_target_dir, args.img)
    test_fp = os.path.join(args.test_target_dir, args.fp)

    source_data_loader = data_loader.load_data_source(
        src_img, src_fp, args.rfimage_source, args.batch_size, args.mean_std, shuffle=True, num_workers=args.num_workers)
    
    target_data_loader,valid_data_loader = data_loader.load_data_target(
        tgt_img, tgt_fp, args.rfimage_target, args.batch_size, args.mean_std, shuffle=True, num_workers=args.num_workers)
    
    test_data_loader = data_loader.load_data_source(
        test_img, test_fp, args.rfimage_target, args.batch_size, args.mean_std, shuffle=False, num_workers=args.num_workers)
    
    return source_data_loader, target_data_loader, valid_data_loader, test_data_loader

def get_model(args):
    model = models.TransferNet(
        args.out_class, args.out_dim, transfer_loss=args.transfer_loss, base_net=args.backbone).to(args.device)
    return model

def get_optimizer(model, args):
    params = model.get_parameters(initial_lr = args.lr)
    optimizer = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args):
    model.eval()
    e_x_r = utils.AverageMeter()                                             
    e_y_r = utils.AverageMeter()                                             
    e_z_r = utils.AverageMeter()                                             
    e_t_r = utils.AverageMeter()                                             
    criterion_reg = torch.nn.L1Loss()                                                       
    with torch.no_grad():     
        for data, fp_label in target_test_loader:
            data, fp_label = data.to(args.device), fp_label.to(args.device)
            reg, _ = model.predict(data)
            reg[:,0] = reg[:,0]*1.5+(-0.75)
            reg[:,1] = reg[:,1]*1.5+(-0.75)
            reg[:,2] = reg[:,2]*3+(-3)
            fp = fp_label[:,:3].to(torch.float32)
            fp[:,0] = fp[:,0]*1.5+(-0.75)
            fp[:,1] = fp[:,1]*1.5+(-0.75)
            fp[:,2] = fp[:,2]*3+(-3)
            e_x = criterion_reg(reg[:,0], fp[:,0])
            e_y = criterion_reg(reg[:,1], fp[:,1])
            e_z = criterion_reg(reg[:,2], fp[:,2])
            e_t = criterion_reg(reg, fp)
            e_x_r.update(e_x.item())
            e_y_r.update(e_y.item())
            e_z_r.update(e_z.item())
            e_t_r.update(e_t.item())
    return [e_x_r.avg,e_y_r.avg,e_z_r.avg,e_t_r.avg]


def train(source_data_loader, target_data_loader, valid_data_loader, test_data_loader, model, optimizer, lr_scheduler, args, log_dir,checkpoint_dir):
    len_source_loader = len(source_data_loader)                       
    len_target_loader = len(target_data_loader)
    n_batch = max(len_source_loader, len_target_loader)   
    log = []
    stop = 0
    best_reg = 10
    for e in range(1, args.n_epoch+1):
        model.train()                                                    
        train_loss_clf = utils.AverageMeter()
        train_loss_reg = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        train_reg_mae = utils.AverageMeter()
        iter_source, iter_target = iter(source_data_loader), iter(target_data_loader)
        for _ in range(n_batch):
            try:
                source, source_fp_label = next(iter_source)     
            except StopIteration:
                iter_source = iter(source_data_loader)
                source, source_fp_label = next(iter_source)     
            try:
                target, _ = next(iter_target)     
            except StopIteration:
                iter_target = iter(target_data_loader)
                target, _ = next(iter_target)  
            source, source_fp_label = source.to(
                args.device), source_fp_label.to(args.device)
            target = target.to(args.device)
            clf_loss, reg_loss, transfer_loss,reg_mae = model(source, target, source_fp_label)   
            loss =  args.clf_weight * clf_loss +  args.reg_weight * reg_loss + args.transfer_weight * transfer_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss_clf.update(clf_loss.item())             
            train_loss_reg.update(reg_loss.item())             
            train_loss_transfer.update(transfer_loss.item())  
            train_loss_total.update(loss.item())
            train_reg_mae.update(reg_mae.item())
        
        stop += 1
        log_temp = [train_loss_clf.avg, train_loss_reg.avg, train_reg_mae.avg, train_loss_transfer.avg, train_loss_total.avg]
        info = 'Epoch: [{:2d}/{}], train_loss: clf(CE): {:.4f}, reg(MSE): {:.4f}, reg(MAE): {:.4f}, trans: {:.4f}, total: {:.4f};'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_reg.avg, train_reg_mae.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Valid
        valid_reg_loss = test(model, valid_data_loader, args)
        info += '\nVALID (MAE force): ex {:.4f} ey {:.4f} ez {:.4f} et {:.4f}'.format(valid_reg_loss[0],valid_reg_loss[1],valid_reg_loss[2],valid_reg_loss[3])    
        if(e > 0 and e%1==0):
            print(info)
            test_reg_loss = test(model, test_data_loader, args)
            log_temp.extend(valid_reg_loss)
            log_temp.extend(test_reg_loss)
            log.append(log_temp)
            np_log = np.array(log, dtype=float)
            np.savetxt(log_dir, np_log, delimiter=',', fmt='%.6f')
            print('TEST (MAE force): ex {:.4f} ey {:.4f} ez {:.4f} et {:.4f}'.format(test_reg_loss[0],test_reg_loss[1],test_reg_loss[2],test_reg_loss[3]))
            if test_reg_loss[3] < best_reg:
                stop = 0
                best_reg = test_reg_loss[3]
                torch.save(model,checkpoint_dir)
            if args.early_stop > 0 and stop >= args.early_stop:
                print(f'best reg loss : {best_reg}')
                break

def main():

    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    mtd = args.config.split(os.sep)[1]
    domain = args.config.split(os.sep)[2]
    inf_dir = os.path.join(args.inf_dir,domain, mtd, args.data_save)
    os.makedirs(inf_dir,exist_ok=True)

    log_dir = os.path.join(inf_dir,args.log_dir)
    checkpoint_dir = os.path.join(inf_dir,args.model_dir)
    inf_s_dir = os.path.join(inf_dir,args.inf_s)
    inf_t_dir = os.path.join(inf_dir,args.inf_t)
    tSNE_dir = os.path.join(inf_dir,args.tSNE)
    
    set_random_seed(args.seed)
    source_data_loader, target_data_loader, valid_data_loader, test_data_loader = load_data(args)
    print(args)
    print("len(source): {:} , len(target): {:}".format(len(source_data_loader),len(target_data_loader)))
    model = get_model(args)
    if args.use_pretrained:
        if args.use_checkpoint and os.path.exists(checkpoint_dir):
            pretrained_pth = checkpoint_dir
        else:
            pretrained_pth = args.pretrained
        model_pretrained = torch.load(pretrained_pth)
        model_dict = model.state_dict()
        pretrained_dict = model_pretrained.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    if args.train:
        train(source_data_loader, target_data_loader, valid_data_loader, test_data_loader, model, optimizer, scheduler, args, log_dir,checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        if "s" in args.inf_t or "source_only" in args.data_save:
            checkpoint_dir = args.pretrained
        else:
            raise Exception("Not found checkpoint dir")

    if args.draw_force:
        inf_force(args, checkpoint_dir,args.test_source_dir,args.rfimage_source,inf_s_dir, batch_size=1,tSNE = False, tSNE_pth = 'source')
        inf_force(args, checkpoint_dir,args.test_target_dir,args.rfimage_target,inf_t_dir, batch_size=1,tSNE = False, tSNE_pth = 'target')
    
    #tSNE
    if args.draw_tsne:
        inf_force(args, checkpoint_dir,args.tSNE_source_dir,args.rfimage_source,inf_s_dir, batch_size=1,tSNE = True, tSNE_pth = 'source')
        inf_force(args, checkpoint_dir,args.tSNE_target_dir,args.rfimage_target,inf_t_dir, batch_size=1,tSNE = True, tSNE_pth = 'target')
        tSNE_s, tSNE_t = 'source.npy','target.npy'
        tSNE_gen(tSNE_s,tSNE_t,tSNE_dir,args)
    print(f"{mtd} finished!")

if __name__ == "__main__":
    main()
