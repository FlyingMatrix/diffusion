# Diffusion

Building generative diffusion models with PyTorch from scratch, which features scripts and guides covering from noise schedulers and U-Net architectures to training loops.

### 🧠 Diffusion Models

**Diffusion models** are a relatively recent addition to a group of algorithms known as **‘generative models’**. The goal of generative modeling is to learn to **generate** data, such as images or audio, given a number of training examples. A good generative model will create a **diverse** set of outputs that resemble the training data without being exact copies.

The secret to diffusion models’ success is the iterative nature of the **diffusion process**. Generation begins with **random noise**, but this is gradually refined over a number of steps until an output image emerges. At each step, the model estimates how we could go from the current input to a **completely denoised version**. However, since we only make a small change at every step, any errors in this estimate at the early stages (where predicting the final output is extremely difficult) can be corrected in later updates.

<div align="center">
  <img src="img/diffusion_process.png" alt="Diffusion">
  <br>
  <p>Diffusion process, figure from DDPM paper (https://arxiv.org/abs/2006.11239).</p>
</div>

Training the model is relatively straightforward compared to some other types of generative model. We repeatedly 1) Load in some images from the training data 2) Add noise, in different amounts. Remember, we want the model to do a good job estimating how to ‘fix’ (denoise) both extremely noisy images and images that are close to perfect. 3) Feed the noisy versions of the inputs into the model 4) Evaluate how well the model does at denoising these inputs 5) Use this information to update the model weights.

To generate new images with a trained model, we begin with a completely random input and repeatedly feed it through the model, updating it each time by a small amount based on the model prediction.

## 🚰 MVP (Minimum Viable Pipeline)

The core API of Hugging Face Diffusers is divided into three main components:

1. **Pipelines**: high-level classes designed to rapidly generate samples from popular trained diffusion models in a user-friendly fashion.
2. **Models**: popular architectures for training new diffusion models, *e.g.* [UNet](https://arxiv.org/abs/1505.04597).
3. **Schedulers**: various techniques for generating images from noise during *inference* as well as to generate noisy images for *training*.
