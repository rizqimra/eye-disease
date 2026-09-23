import torch
import torch.nn.functional as F
import numpy as np
import scipy as sp
import torchvision.models as models


def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = sp.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sp.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value


def calculate_inception_score(images, model=None, splits=10, cuda=True):
    """
    Computes the Inception Score (IS) for a set of images using torchvision's InceptionV3.
    Args:
        images: torch.Tensor, shape (N, 3, H, W) - should be in [0,1] range, resized to 299x299
        model: ignored, kept for compatibility
        splits: number of splits
        cuda: use GPU
    Returns:
        mean IS, std IS
    """
    # Load pretrained InceptionV3 classifier
    inception = models.inception_v3(pretrained=True, aux_logits=True, transform_input=False)
    inception.eval()
    if cuda:
        inception = inception.cuda()
    
    N = images.size(0)
    batch_size = 32
    if cuda:
        images = images.cuda()
    
    # Denormalize from [-1,1] to [0,1] as expected by InceptionV3
    images = (images + 1) / 2
    
    # Resize images to 299x299 as required by InceptionV3
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    preds = []
    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            logits_list = []
            for j in range(batch.size(0)):
                single_image = batch[j:j+1]  # (1, 3, 299, 299)
                outputs = inception(single_image)
                logit = outputs[0]  # (1000,)
                logits_list.append(logit)
            logits = torch.stack(logits_list, dim=0)  # (batch_size, 1000)
            logits = torch.clamp(logits, -100, 100)  # Prevent inf in logits
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs)
    preds = np.concatenate(preds, axis=0)
    split_scores = []
    for k in range(splits):
        start = k * (N // splits)
        end = (k + 1) * (N // splits)
        part = preds[start:end]
        py = np.mean(part, axis=0)
        py = np.maximum(py, 1e-8)  # Avoid zeros in py
        py = py / py.sum()  # Renormalize
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i]
            score = sp.stats.entropy(pyx, py)
            scores.append(score)
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)