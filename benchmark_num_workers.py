import time
import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import RetinaDataset

def main():
    data_dir = 'dataset/'
    batch_size = 64
    image_size = 64  # or 128, as you use in your experiments

    dataset = RetinaDataset(
        image_folder=data_dir,
        transform=transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        target_class=None  # or set a class if you want
    )

    for num_workers in range(0, mp.cpu_count()+1, 2):  # test 0, 2, 4, ..., max
        if num_workers == 0:
            print("Testing with num_workers=0 (main process only)")
        else:
            print(f"Testing with num_workers={num_workers}")

        train_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        start = time.time()
        for epoch in range(3):  # 3 epochs is enough for benchmarking
            for i, data in enumerate(train_loader):
                pass  # just iterate, don't do any computation
        end = time.time()
        print(f"Finished in {end - start:.2f} seconds with num_workers={num_workers}\n")

if __name__ == "__main__":
    main()