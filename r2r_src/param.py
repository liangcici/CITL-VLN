import argparse
import os
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--test_only', type=int, default=0, help='fast mode for testing')

        self.parser.add_argument('--iters', type=int, default=300000, help='training iterations')
        self.parser.add_argument('--name', type=str, default='default', help='experiment id')
        self.parser.add_argument('--vlnbert', type=str, default='oscar', help='oscar or prevalent')
        self.parser.add_argument('--train', type=str, default='listener')
        self.parser.add_argument('--description', type=str, default='no description\n')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')

        # Augmented Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.20)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--features", type=str, default='places365')

        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # Submision configuration
        self.parser.add_argument("--submit", type=int, default=0)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        self.parser.add_argument("--polyaxon", action="store_const", default=False, const=True)
        self.parser.add_argument("--aug_path", default=None)
        self.parser.add_argument("--aug_path2", default=None)
        self.parser.add_argument("--aug_train_path", default=None)
        self.parser.add_argument("--aug_train_path2", default=None)
        self.parser.add_argument("--aug_path_type", default="ndtw", type=str, help="[ndtw, distance, len]")
        self.parser.add_argument("--pos_thr", default=0.8, type=float)
        self.parser.add_argument("--neg_thr", default=0.65, type=float)
        self.parser.add_argument("--fna_thr", default=0.7, type=float)
        self.parser.add_argument("--con_path_loss_weight", default=0, type=float)
        self.parser.add_argument("--aug_con_path_loss_weight", default=0, type=float)
        self.parser.add_argument("--local_con_path_loss_weight", default=0, type=float)
        self.parser.add_argument("--con_neg_path_loss_weight", default=0, type=float)
        self.parser.add_argument("--con_path_loss_type", default="nce", type=str, help='[nce, circle]')
        self.parser.add_argument('--nce_t', default=0.07, type=float, help='softmax temperature (default: 0.07)')
        self.parser.add_argument('--nce_k', default=8000, type=int, help='queue size; number of negative keys')
        self.parser.add_argument("--multi_feat_dim", default=128, type=int)
        self.parser.add_argument("--aug_path_num", default=6, type=int)
        self.parser.add_argument("--circle_m", default=0.25, type=float)
        self.parser.add_argument("--circle_gamma", default=256, type=int)
        self.parser.add_argument("--circle_queue", action="store_const", default=False, const=True)
        self.parser.add_argument("--circle_mining", action="store_const", default=False, const=True)
        self.parser.add_argument("--circle_fna", action="store_const", default=False, const=True)
        self.parser.add_argument("--aug_lang", default=None)
        self.parser.add_argument("--aug_train_lang", default=None)
        self.parser.add_argument("--aug_lang_num", default=13, type=int)
        self.parser.add_argument("--lang_loss_weight", default=0, type=float)
        self.parser.add_argument("--lang_local_loss_weight", default=0, type=float)
        self.parser.add_argument("--data_percent", default=1., type=float)
        self.parser.add_argument("--workers", default=4, type=int)
        self.parser.add_argument("--shared_optimizer", action="store_const", default=False, const=True)
        self.parser.add_argument('--amsgrad', action="store_const", default=False, const=True)
        self.parser.add_argument('--cos', action="store_const", default=False, const=True, help='cosine learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
        self.parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='SGD weight decay (default: 1e-4)')
        self.parser.add_argument('--gpu_ids', type=int, default=0, nargs='+', help='GPUs to use')
        self.parser.add_argument("--dataset", default="R2R", type=str, help='[R2R,]')
        self.parser.add_argument('--multi_aug_instr', action="store_const", default=False, const=True)
        self.parser.add_argument('--aug_val', action="store_const", default=False, const=True)
        self.parser.add_argument("--aug_val_type", default="synonym", type=str, help='[synonym, substitute, insert, back_translation]')
        self.parser.add_argument("--aug_lang_type", default=None, type=str, help='[synonym, substitute, insert, back_translation]')
        self.parser.add_argument('--no_intra_negative', action="store_const", default=False, const=True)
        self.parser.add_argument('--cross', action="store_const", default=False, const=True)
        self.parser.add_argument('--local_attn', action="store_const", default=False, const=True)
        self.parser.add_argument('--local_attn_input', action="store_const", default=False, const=True)
        self.parser.add_argument('--reweight', action="store_const", default=False, const=True)
        self.parser.add_argument('--aug_lang_nearest', action="store_const", default=False, const=True)
        self.parser.add_argument('--cross_candidate_feat', action="store_const", default=False, const=True)

        self.args = self.parser.parse_args()

        if not self.args.shared_optimizer:
            if self.args.optim == 'rms':
                print("Optimizer: Using RMSProp")
                self.args.optimizer = torch.optim.RMSprop
            elif self.args.optim == 'adam':
                print("Optimizer: Using Adam")
                self.args.optimizer = torch.optim.Adam
            elif self.args.optim == 'adamW':
                print("Optimizer: Using AdamW")
                self.args.optimizer = torch.optim.AdamW
            elif self.args.optim == 'sgd':
                print("Optimizer: sgd")
                self.args.optimizer = torch.optim.SGD
            else:
                assert False

param = Param()
args = param.args

args.description = args.name
args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'

args.output_dir = ''
args.log_dir = os.path.join(args.output_dir, 'snap/%s' % args.name)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join(args.output_dir, 'snap', args.name, "debug.log"), 'w')
