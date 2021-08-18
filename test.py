import time
import fiftyone as fo
import fiftyone.zoo as foz

# The path to the source files that you manually downloaded
source_dir = "/media/zihao/data1"

dataset = foz.load_zoo_dataset(
    "imagenet-2012",
    split="train",
    source_dir=source_dir,
    max_samples=50,
    shuffle=True,
    seed=time.time_ns()
)

session = fo.launch_app(dataset)
session.wait()