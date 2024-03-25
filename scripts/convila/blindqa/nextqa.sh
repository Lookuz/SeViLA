result_dir="/home/users/nus/e0176617/SeViLA/outputs/nextqa"
exp_name='blindqa'
checkpoint='checkpoints/sevila_pretrained.pth'
cfg_path='lavis/projects/blindqa/eval/nextqa.yaml'
task='blindqa'

# Note: No spaces or dashes to be used in the data paths, as absolute paths are computed and formatted within the LAVIS code
train_path='/scratch/users/nus/e0176617/datasets/nextqa/processed/train.json'
val_path='/scratch/users/nus/e0176617/datasets/nextqa/processed/val.json'
test_path='/scratch/users/nus/e0176617/datasets/nextqa/processed/test.json'
video_path='/scratch/users/nus/e0176617/datasets/nextqa/videos'
embeddings_path='/scratch/users/nus/e0176617/datasets/nextqa/embeddings'

batch_size=16
n_frms=1

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 evaluate.py \
    --cfg-path ${cfg_path} \
    --options run.output_dir=${result_dir}_${exp_name} \
    model.num_keyframes=4 \
    model.task=${task} \
    model.finetuned=${checkpoint} \
    datasets.nextqa.vis_processor.eval.n_frms=${n_frms} \
    datasets.nextqa.build_info.annotations.train.storage=${train_path} \
    datasets.nextqa.build_info.annotations.val.storage=${val_path} \
    datasets.nextqa.build_info.annotations.test.storage=${test_path} \
    datasets.nextqa.build_info.videos.storage=${video_path} \
    run.task='videoqa' \
    run.batch_size_eval=${batch_size} 