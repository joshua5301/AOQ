import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import matplotlib.pyplot as plt

# Resolve everything relative to this file so the script can be run from any
# working directory (e.g. the repo root).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, ".."))
from utils.utils import *
from utils import KD_loss
from torchvision import datasets, transforms
from torch.autograd import Variable
from resnet import *
import torchvision.models as models
import quan
from globalVal import globalVal


def run_dir(args):
    """Directory name identifying a run.

    The optimizer is part of the identity: without it an sgd run would silently
    overwrite the adam baseline's checkpoints. Only non-default optimizers add
    a suffix, so existing adam paths are unchanged.
    """
    tag = "{}_{}bit_quantize_downsample_{}".format(
        args.student, args.n_bit, args.quantize_downsample
    )
    if args.optimizer != "adam":
        tag += "_" + args.optimizer
    return tag


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser("aoq")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--epochs", type=int, default=256, help="num of training epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="init learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
parser.add_argument(
    "--save",
    type=str,
    default=os.path.join(_HERE, "models"),
    help="path for saving trained models",
)
parser.add_argument(
    "--data",
    metavar="DIR",
    default=os.path.join(_HERE, "..", ".."),
    help="path to the directory containing cifar-10-batches-py (default: repo root)",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="continue from checkpoint.pth.tar in --save (default: start from scratch)",
)
parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing")
parser.add_argument("--teacher", type=str, default="resnet20", help="teacher model")
parser.add_argument("--student", type=str, default="resnet20", help="student model")
parser.add_argument("--n_bit", type=int, default=2, help="number of bits")
parser.add_argument(
    "--quantize_downsample",
    type=str,
    default="True",
    help="quantize downsampling layer or not",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    choices=["adam", "adamw", "sgd"],
    help="optimizer (default: adam, the original). adamw decouples weight decay; "
    "sgd uses --momentum. All three keep the alpha group at lr/10",
)
parser.add_argument(
    "-j",
    "--workers",
    default=6,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
args = parser.parse_args()

resnet_dict = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet110": resnet110,
}
CLASSES = 10

LOG_DIR = os.path.join(_HERE, "log")
os.makedirs(LOG_DIR, exist_ok=True)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(LOG_DIR, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

device = torch.device(globalVal.device)


def main():
    setup_seed(42)
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)

    # load model
    model_teacher = resnet_dict[args.teacher](pretrained=True)
    # model_teacher = nn.DataParallel(model_teacher, device_ids=device_ids).cuda()
    model_teacher = model_teacher.to(device)
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    if args.quantize_downsample == "True" or args.quantize_downsample == "1":
        args.quantize_downsample = True
    else:
        args.quantize_downsample = False

    model_student = resnet_dict[args.student](pretrained=True)
    print(model_student)
    modules_to_replace = quan.find_modules_to_quantize(model_student, args.n_bit)
    model_student = quan.replace_module_by_names(model_student, modules_to_replace)
    model_student = model_student.to(device)
    logging.info("student:")
    logging.info(model_student)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.to(device)
    criterion_kd = KD_loss.DistributionLoss()

    all_parameters = model_student.parameters()
    weight_parameters = []
    alpha_parameters = []

    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 and "bias" not in pname:
            print("weight_param:", pname)
            weight_parameters.append(p)
        elif "quan_a_fn.a" in pname or "quan_a_fn.scale" in pname or "quan_a_fn.start" in pname:
            print("alpha_param:", pname)
            alpha_parameters.append(p)

    weight_parameters_id = list(map(id, weight_parameters))
    alpha_parameters_id = list(map(id, alpha_parameters))
    other_parameters1 = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    other_parameters = list(filter(lambda p: id(p) not in alpha_parameters_id, other_parameters1))

    param_groups = [
        {"params": alpha_parameters, "lr": args.learning_rate / 10},
        {"params": other_parameters, "lr": args.learning_rate},
        {
            "params": weight_parameters,
            "lr": args.learning_rate,
        },
    ]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        # weight_decay defaults to 0, so the adam branch is exactly the original
        # optimizer. AdamW's own default is 0.01, hence passing it explicitly.
        optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw" and args.weight_decay == 0:
        logging.warning(
            "--optimizer=adamw with --weight_decay=0 is exactly Adam; the whole "
            "difference between them is how the decay term is applied."
        )
    logging.info(
        "optimizer: %s lr=%g momentum=%g weight_decay=%g",
        args.optimizer,
        args.learning_rate,
        args.momentum,
        args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0 - step / args.epochs), last_epoch=-1)
    start_epoch = 0
    best_top1_acc = 0

    checkpoint_tar = os.path.join(
        args.save,
        run_dir(args),
        "checkpoint.pth.tar",
    )
    if os.path.exists(checkpoint_tar):
        if args.resume:
            logging.info("loading checkpoint {} ..........".format(checkpoint_tar))
            checkpoint = torch.load(checkpoint_tar)
            start_epoch = checkpoint["epoch"] + 1
            best_top1_acc = checkpoint["best_top1_acc"]
            model_student.load_state_dict(checkpoint["state_dict"], strict=False)
            logging.info("loaded checkpoint {} epoch = {}".format(checkpoint_tar, checkpoint["epoch"]))
        else:
            # Resuming silently continues a previous run and can skip training
            # entirely, which quietly invalidates a comparison. Opt in instead.
            logging.warning(
                "found an existing checkpoint at %s - starting from scratch and "
                "overwriting it. Pass --resume to continue that run instead.",
                checkpoint_tar,
            )

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # data augmentation
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    # The dataset ships with the repo, so only download when it is absent.
    need_download = not os.path.isdir(os.path.join(args.data, "cifar-10-batches-py"))
    logging.info("dataset root: %s (download=%s)", os.path.abspath(args.data), need_download)

    train_dataset = datasets.CIFAR10(
        args.data, train=True, transform=train_transforms, download=need_download
    )
    val_dataset = datasets.CIFAR10(args.data, train=False, transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model_student = model_student.to(device)
    epoch = start_epoch

    # ASCII on purpose: a Windows console defaults to a legacy codepage
    # (cp949, cp1252) and raises UnicodeEncodeError on anything outside it.
    logging.info("teacher accuracy")
    validate(-2, val_loader, model_teacher, criterion, args)

    while epoch < args.epochs:
        globalVal.epoch = float(epoch)

        if epoch % 10 == 0:
            fname = os.path.join(LOG_DIR, "epoch" + str(epoch) + ".png")
            plt.figure(1)
            plt.clf()
            plt.hist(
                model_student.state_dict()["layer2.2.conv2.weight"].reshape(-1).cpu(),
                bins=200,
                range=(-0.5, 0.5),
            )
            plt.savefig(fname)

        train_obj, train_top1_acc, train_top5_acc = train(
            epoch,
            train_loader,
            model_student,
            model_teacher,
            criterion_kd,
            optimizer,
            scheduler,
        )
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model_student, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model_student.state_dict(),
                "best_top1_acc": best_top1_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            os.path.join(
                args.save,
                run_dir(args),
            ),
        )

        epoch += 1

    training_time = (time.time() - start_t) / 36000
    print("total training time = {} hours".format(training_time))


def train(epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model_student.train()
    model_teacher.eval()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group["lr"]
    print("learning_rate:", cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        # compute outputy
        logits_student = model_student(images)
        logits_teacher = model_teacher(images)
        if globalVal.epoch <= 150:
            globalVal.loss = 0.0
            loss = criterion(logits_student, logits_teacher)
        else:
            loss1 = criterion(logits_student, logits_teacher)
            loss2 = globalVal.loss
            globalVal.loss = 0.0
            loss = loss1 + 0.01 * loss2

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    return losses.avg, top1.avg, top5.avg


def validate(epoch=-1, val_loader=None, model=None, criterion=None, args=None):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        print(" * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
