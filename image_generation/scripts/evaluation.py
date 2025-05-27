import argparse
from valuefunction.DAgger_train_value_function_comp import guided_generation
import torch
import os

device = 'cuda'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate guided generation with different checkpoints and parameters.")
    parser.add_argument("--checkpoints", nargs='+', type=str, 
                        default=[
                            'comp_model/1746905677/ckpts/reward_predictor_overalliter_3_epoch_0.pth',
                        ],
                        help="List of checkpoint paths to evaluate.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base directory to save logs and images.")
    parser.add_argument("--etas", nargs='+', type=float, default=[6.5, 7, 8, 8.5, 9, 10], help="List of eta values for guidance.")
    parser.add_argument("--guidance_strengths", nargs='+', type=int, default=[75], help="List of guidance strength values.")
    parser.add_argument("--scale_coeff", type=float, default=0.0, help="Scale coefficient for guidance.")
    parser.add_argument("--eval_num", type=int, default=256, help="Number of images to generate for evaluation.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for generation.")

    args = parser.parse_args()

    for ck_id, ckpt_path in enumerate(args.checkpoints):
        convnet = torch.load(ckpt_path).to(device)
        convnet.eval()
        
        score_dic = {}
        image_list = []
        for eta_val in args.etas:
            for guidance_val in args.guidance_strengths:
                current_log_path = os.path.join(args.log_dir, f'ckpt_{ck_id}', f'{eta_val}_{guidance_val}')
                os.makedirs(current_log_path, exist_ok=True)
                
                images, scores = guided_generation(
                    convnet, 
                    eta_val,
                    guidance_val,
                    scale_coeff=args.scale_coeff,
                    eval_num=args.eval_num,
                    return_dis=True, 
                    random_seed=args.random_seed
                )
                image_list.extend(images)
                score_dic[eta_val * 100000 + guidance_val] = scores
                
                for img_id, image in enumerate(images):
                    image.save(os.path.join(current_log_path, f'{img_id}.png'))
                
                with open(os.path.join(current_log_path, 'scores.txt'), 'w') as f:
                    f.write('average score:\n')
                    f.write(str(scores.mean()) + '\n')
                    f.write('all scores:\n')
                    for score in scores:
                        f.write(str(score) + '\n')
                        
        print(f"Scores for checkpoint {ck_id} ({ckpt_path}):")
        for key in score_dic:
            eta_from_key = int(key / 100000)
            guidance_from_key = int(key % 100000)
            print(f"  Eta: {eta_from_key}, Guidance: {guidance_from_key}, Mean Score: {score_dic[key].mean()}")

