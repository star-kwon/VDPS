from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
import argparse

def main():
    parser = argparse.ArgumentParser(description="VDPS")
    parser.add_argument("--dist_min", type=float, default="propagation distance min")
    parser.add_argument("--dist_max", type=float, default="propagation distance max")
    parser.add_argument("--sigma_min", type=float, default="kernel width min")
    parser.add_argument("--sigma_max", type=float, default="kernel width max")
    parser.add_argument("--dataset", type=str, default="UCF101")
    args = parser.parse_args()

    model = Unet3D(
        channels=1,
        dim = 128,
        dim_mults = (1, 2, 3, 4)
    )

    diffusion = GaussianDiffusion(
        model,
        channels=1,
        image_size = 64,
        num_frames = 10,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2
    ).cuda()

    trainer = Trainer(
        diffusion,
        './data/UCF101_Test',
        train_batch_size = 4,
        train_lr = 1e-4,
        save_and_sample_every = 10000,
        train_num_steps = 1000000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                        # turn on mixed precision
        dataset=args.dataset
    )
    trainer.load(100.1)

    trainer.eval_physics_informed_blind(dist_low=args.dist_min, dist_high=args.dist_max, sigma_low=args.sigma_min, sigma_high=args.sigma_max)

if __name__ == "__main__":
    main()