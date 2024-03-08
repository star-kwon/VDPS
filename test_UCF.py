from video_through_scattering_pytorch import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(
    channels=1,
    dim = 128,
    dim_mults = (1, 2, 3, 4),
    attn_dim_head = 32
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
    folder = './scatter samples/UCF101',  # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    models_folder='./models/UCF101'
)

trainer.load()
trainer.eval_physics_informed(sigma=0.5, dist=2.5)