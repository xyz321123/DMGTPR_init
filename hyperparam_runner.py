import uuid
import torch
import logging
import argparse
import traceback
import numpy as np
from main import main
from pathlib import Path


create_dir = lambda dir_location: Path(dir_location).mkdir(parents=True, exist_ok=True)
#直接创建一个函数create_dir,功能是创建一个新目录,地点在dir_location

def str2bool(v):
    #将输入的字符串转化成bool值
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    #输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', '-name', help="log file name", type=str)
    parser.add_argument('--dataset_name', '-dataset', help="name of the dataset", type=str)
    parser.add_argument('--noise_layer', '-noise_layer', help="True for using noise; else false (default).", type=str)
    # parser.add_argument('--eps', '-eps', help="[0.1, 0.2, 2.0, 2.4]", type=str)
    parser.add_argument('--eps_list', '-eps_list', nargs="*", help="--eps_list 0.1, 0.2, 2.0, 2.4", type=float)
    parser.add_argument('--is_adv', '-is_adv', help="True for using adv; else false (default).", type=str)
    parser.add_argument('--apply_noise_to_adv', '-apply_noise_to_adv', help="True noise on adv else no noise on adv", type=str)
    parser.add_argument('--adv_start', '-adv_s', help="start of adv scale for increment increase for ex: 0.1", type=float)
    parser.add_argument('--adv_end', '-adv_e', help="end of adv scale 1.0", type=float)
    parser.add_argument('--epochs', '-epochs', help="epochs!", type=int)
    parser.add_argument('--use_lr_schedule', '-use_lr_schedule', help="use_lr_schedule!", type=str)
    parser.add_argument('--mode_of_loss_scale', '-mode_of_loss_scale', help="constant/linear/exp", type=str)
    parser.add_argument('--seed', '-seed', nargs="*", help="1234 3567 7508", type=int)
    parser.add_argument('--diverse_adversary', '-diverse_adversary', help="True for using diverse adversary; else false (default).", type=str)
    parser.add_argument('--diverse_adv_lambda', '-diverse_adv_lambda', help="diverse_adv_lambda the orthogonal loss weight", type=float)
    parser.add_argument('--model', '-model', help="diverse_adv_lambda the orthogonal loss weight", type=str)
    parser.add_argument('--diff_scale', help="diffusion encoder loss weight", type=float)

    args = parser.parse_args()

    assert args.log_name != None
    assert args.dataset_name != None
    assert args.epochs != None

    #设置线程
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    # setting up the run
    epochs = args.epochs # epochs:45
    dataset_name = args.dataset_name # encoded_bias_in_bios
    noise_layer = str2bool(args.noise_layer) # 噪声层:True
    is_adv = str2bool(args.is_adv) # 攻击分类器adv:True
    bs = 320 # batch_size:320 64*5
    # only_perturbate = True # 是否只进行扰动，没啥用
    mode_of_loss_scale = args.mode_of_loss_scale # linear atleast for amazon!
    # mode_of_loss_scale:exp 在梯度反转模式下，adv的损失规模随 epoch 增加的方式
    # optimizer = 'sgd'
    use_lr_schedule = str2bool(args.use_lr_schedule)
    seeds = args.seed # 随机种子

    supervised_da = False

    if args.apply_noise_to_adv: # 噪声同时应用于 adv 和分类器，否则仅应用于分类器
        apply_noise_to_adv = str2bool(args.apply_noise_to_adv)
    else:
        apply_noise_to_adv = True

    lrs = [('adam', 0.001)] # 学习率

    if noise_layer: # epsilon大小
        # epss = map(float, args.eps.strip('[]').split(','))
        epss = args.eps_list
    else:
        epss = [0.0]

    if is_adv: # adv的系数lambda大小
        if args.adv_end == 0.0:
            adv_scales = args.adv_start
        else:
            adv_scales = [round(i,2) for i in np.arange(args.adv_start,args.adv_end,0.2)]
    else:
        adv_scales = [0.0]

    if args.diverse_adversary: # 是否给攻击分类器添加正交损失（不需要）
        diverse_adversary = True
    else:
        diverse_adversary = False

    if args.diverse_adv_lambda: # 正交损失的lambda（不需要）
        diverse_adv_lambda = args.diverse_adv_lambda
    else:
        diverse_adv_lambda = 0.0

    if args.model: # 攻击模型和分类器模型都是线性模型
        model = args.model
    else:
        model = 'linear_adv_encoded_emoji'

    #按照数据集计算公平分数和泄露率
    if dataset_name in ['blog', 'blog_v2']:
        fairness_score_function = 'calculate_multiple_things_blog'
    elif "amazon" in dataset_name:
        fairness_score_function = 'dummy_fairness'
        if len(dataset_name.split("_")) == 2:
            supervised_da = True
    elif "encoded_bias_in_bios" == dataset_name:
        fairness_score_function = 'grms'
    else:
        fairness_score_function = 'multiple_things'

    if dataset_name == 'encoded_emoji' or 'blog' in dataset_name or "encoded_bias_in_bios" == dataset_name :
        calculate_leakage = True
    else:
        calculate_leakage = False

    ########### 日志文件 ###########
    # create main logging dir
    logs_dir = Path('../logs')
    create_dir(logs_dir)

    # create dataset dir in logs_dir
    if supervised_da:
        dataset_dir = logs_dir / Path(args.dataset_name + '_supervised_da')
    else:
        dataset_dir = logs_dir / Path(args.dataset_name)
    create_dir(dataset_dir)
    log_file_name = str(dataset_dir / Path(args.log_name)) + '.log'

    # logger init
    logging.basicConfig(filename=log_file_name,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info(f"logging for {log_file_name}")
    logger = logging.getLogger('main')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)
    ########### 日志文件结束 ###########

    for seed in seeds: # 随机种子[20,30,40,50,60]
        for eps in epss: # 隐私预算[8,10,12,14,16]
            for adv_scale in adv_scales: # 联合损失函数中的λ[1.0]
                for optimizer, lr in lrs: # 优化器adam，学习率[0.001]
                    unique_id = str(uuid.uuid4()) # 生成唯一随机数
                    try:
                        logger.info(f"start of run - {unique_id}")
                        main(emb_dim=300, # 嵌入维数[300]
                             spacy_model="en_core_web_sm", # 用于token化的spacy模型
                             seed=seed,
                             dataset_name=dataset_name,
                             batch_size=bs,
                             pad_token='<pad>',
                             unk_token='<unk>',
                             pre_trained_embeddings='../../bias-in-nlp/different_embeddings/simple_glove_vectors.vec', # 没有用
                             model_save_name=f'dummy.pt',
                             model=model,  # 'linear_adv_encoded_emoji' for diverse adv.
                             regression=False,
                             tokenizer_type='simple', # 只能选3种[tweet,spacy,simple]
                             use_clean_text=True,
                             max_length=None,
                             epochs=epochs,
                             learnable_embeddings=False, # 可学习嵌入
                             vocab_location=False, # 词汇表地址
                             is_adv=is_adv,
                             adv_loss_scale=adv_scale,
                             use_pretrained_emb=False, # 是否使用预训练嵌入
                             default_emb_dim=300, # 预训练嵌入维数
                             save_test_pred=False, # 没有用
                             noise_layer=noise_layer, # 是否添加噪声层
                             eps=eps, # epsilon
                             is_post_hoc=False, # 是否训练post-hoc分类器来推测敏感属性
                             train_main_model=True,
                             use_wandb=False,
                             config_dict="",
                             experiment_name="hyper-param-search",
                             only_perturbate=False, # 是否只进行扰动
                             mode_of_loss_scale=mode_of_loss_scale,
                             training_loop_type='three_phase_custom',
                             hidden_loss=False,
                             hidden_l1_scale=0.5,
                             hidden_l2_scale=0.5,
                             reset_classifier=False,
                             reset_adv=True,
                             encoder_learning_rate_second_phase=0.01,
                             classifier_learning_rate_second_phase=0.01,
                             trim_data=True, # 减少bias_in_bios的训练数据到15000
                             eps_scale='constant',
                             optimizer=optimizer,
                             lr=lr,
                             fair_grad=False,
                             reset_fairness=False,
                             use_adv_dataset=True,
                             use_lr_schedule=use_lr_schedule,
                             fairness_function='demographic_parity',
                             fairness_score_function=fairness_score_function,
                             sample_specific_class=True, # 只在bias_in_bios中有用
                             calculate_leakage=calculate_leakage,
                             clip_fairness=True,
                             normalize_fairness=True,
                             fairness_iterator='custom_3',
                             supervised_da=supervised_da,
                             apply_noise_to_adv=apply_noise_to_adv,
                             diverse_adversary=diverse_adversary,
                             diverse_adv_lambda=diverse_adv_lambda,
                             loss_diff_scale=args.diff_scale,
                             )
                        logger.info(f"end of run - {unique_id}")
                    except KeyboardInterrupt:
                        raise IOError
                    except Exception:
                        error = traceback.print_exc()
                        print(error)
                        logger.info(error)
                        logger.info(f"run failed for some reason - {unique_id}")
                        logger.info(f"end of run - {unique_id}")
                        continue

